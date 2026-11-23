"""300-Episode Continuous Fault Ramp Experiment.

Evaluates Research Questions 1 & 2:
- RQ1: Does the self-adaptive VLA policy mask wear/damage as the physical fault delta
  linearly grows from 0 to 8 degrees across 300 continuous episodes?
- RQ2: Do internal signal detectors (torque residual, forward kinematics discrepancy)
  provide advance warning (lead time) before catastrophic task failure?

Saves:
- eval_out/ramp_<tag>_episodes.csv (full per-episode telemetry and detector readings)
- eval_out/ramp_<tag>_summary.json (aggregate metrics, alarm thresholds, lead time)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if sys.platform == 'linux':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from PIL import Image

import draccus
_orig_decode = draccus.parsers.decoding.decode_dataclass
def _patched_decode_dataclass(cls, d, *args, **kwargs):
    if isinstance(d, dict):
        for k in ['enable_lora_vlm', 'enable_lora_action_expert', 'train_action_expert_only', 'model_dtype']:
            d.pop(k, None)
    return _orig_decode(cls, d, *args, **kwargs)
draccus.parsers.decoding.decode_dataclass = _patched_decode_dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

_orig_lora_target_modules = MolmoAct2Policy._lora_target_modules
def _champion_lora_target_modules(self, *, prefix: str, lm_head_path: str) -> str:
    vlm_linear_leaves = 'w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding'
    return rf'(?:{prefix}\.(transformer|vision_backbone|image_pooling_2d|image_projector)\.(?:.*\.)?({vlm_linear_leaves}))$'

from so101_nexus import OverheadCamera, WristCamera
from so101_nexus.mujoco.pick_and_place import PickAndPlaceConfig, PickAndPlaceEnv
from so101_nexus.objects import CubeObject

from molmoact2_so101_sim import realism
from eval.detectors import TorqueResidualDetector, KinematicResidualDetector

HALF = 0.015
SUB = 6
JS = np.array([1., -1., 1., 1., 1., 1.])
JO = np.array([0., 90., 90., 0., 0., 0.])
TASK = 'Pick up the red cube and place it on the blue circle.'


class EncoderOffset(gym.Wrapper):
    """Simulates physical joint coupling slippage delta.
    Command is shifted by +delta (physical arm link moves to q + delta),
    while reported state is shifted by -delta (policy sees q).
    """
    def __init__(self, env, delta):
        super().__init__(env)
        self.delta = np.asarray(delta, dtype=np.float64)
        self.n = len(self.delta)

    def set_delta(self, delta):
        self.delta = np.asarray(delta, dtype=np.float64)
        self.n = len(self.delta)

    def action(self, a):
        a = np.array(a, dtype=np.float64)
        a[: self.n] += self.delta
        lo, hi = self.env.action_space.low, self.env.action_space.high
        return np.clip(a, lo, hi).astype(np.float32)

    def _mask_obs(self, obs):
        if isinstance(obs, dict) and 'state' in obs:
            obs = dict(obs)
            st = np.array(obs['state'], dtype=np.float64)
            st[: self.n] -= self.delta
            obs['state'] = st.astype(np.float32)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._mask_obs(obs), info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(self.action(a))
        return self._mask_obs(obs), r, term, trunc, info

    def reported_qpos(self):
        q = np.array(self.env.unwrapped._get_current_qpos(), dtype=np.float64)
        q[: self.n] -= self.delta
        return q


def g2model(gdeg):
    return float(np.clip((gdeg + 10.) / 110. * 45., 0, 45))


def model2gdeg(gm):
    return float(gm / 45. * 110. - 10.)


def build_policy(spec, meta, control_mode=None, n_action_steps=10):
    if spec == 'zeroshot':
        cfg = make_policy_config(
            'molmoact2', checkpoint_path='allenai/MolmoAct2-SO100_101',
            norm_tag='so100_so101_molmoact2',
            chunk_size=30, n_action_steps=30,
            setup_type='single so100/so101 robotic arm in molmoact2',
            control_mode='absolute joint pose',
            image_keys=['observation.images.cam0', 'observation.images.cam1'],
            normalize_gripper=True, inference_action_mode='continuous', device='cuda')
        pol = make_policy(cfg, ds_meta=meta)
        pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    else:
        if 'champion' in str(spec):
            MolmoAct2Policy._lora_target_modules = _champion_lora_target_modules
        else:
            MolmoAct2Policy._lora_target_modules = _orig_lora_target_modules
        pol = MolmoAct2Policy.from_pretrained(spec)
        pol.config.inference_action_mode = 'continuous'
        if control_mode:
            pol.config.control_mode = control_mode
        pol.to('cuda')
        pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=spec)
    pol.config.per_episode_seed = True
    pol.config.n_action_steps = n_action_steps
    pol.config.eval_seed = 1000
    pol.eval()
    return pol, pre, post


def obs_to_batch(obs):
    c0 = np.ascontiguousarray(obs['overhead_camera']).astype(np.uint8)
    w = np.ascontiguousarray(obs['wrist_camera']).astype(np.uint8)
    c1 = w if w.shape[:2] == (480, 640) else np.asarray(Image.fromarray(w).resize((640, 480))).astype(np.uint8)

    def img(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.).unsqueeze(0).to('cuda')

    sd = np.rad2deg(obs['state'])
    st = (JS * sd + JO).astype(np.float32)
    st[5] = g2model(sd[5])
    batch = {'observation.images.cam0': img(c0), 'observation.images.cam1': img(c1),
             'observation.state': torch.from_numpy(st).unsqueeze(0).to('cuda'), 'task': [TASK]}
    return batch


def act_to_sim(a, lo, hi):
    arm = np.deg2rad((a[:5] - JO[:5]) / JS[:5])
    grip = np.deg2rad(model2gdeg(a[5]))
    return np.clip(np.concatenate([arm, [grip]]).astype(np.float32), lo, hi)


def run_ramp_episode(env, wrapper, pol, pre, post, seed, max_pol=360, min_reach=0.23, max_reach=0.27,
                     torque_det=None, kin_det=None):
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    obs, _ = env.reset(seed=seed)
    pol.reset()
    if hasattr(pol, '_rollout_index_for_task'):
        pol._rollout_index_for_task = -1
        pol._rollout_task_key = None

    U = env.unwrapped
    r_obj = float(np.hypot(*U._get_object_pose()[:2]))
    if r_obj > max_reach or (min_reach > 0.0 and r_obj < min_reach):
        return None

    lo, hi = env.action_space.low, env.action_space.high
    grasped = False
    lifted = 0.0
    torque_residuals = []
    kin_errors = []

    for t in range(max_pol):
        batch = obs_to_batch(obs)
        if pre is not None:
            proc = pre(dict(batch))
            with torch.inference_mode():
                a_raw = pol.select_action(proc)
            a = post(a_raw)[0].detach().cpu().numpy()
        else:
            with torch.inference_mode():
                a = pol.select_action(batch)[0].detach().cpu().numpy()

        q = act_to_sim(a, lo, hi)
        for _ in range(SUB):
            obs, _r, term, _t2, info = env.step(q)
            lh = float(info.get('lift_height', 0))
            lifted = max(lifted, lh)
            grasped |= bool(info.get('is_grasped')) or (lh > 0.02)

            # Signal Detectors tick
            rep_q = wrapper.reported_qpos()
            if torque_det is not None:
                t_res = torque_det.compute_residual(U.data, rep_q)
                torque_residuals.append(t_res['residual_norm'])
            if kin_det is not None:
                k_res = kin_det.compute_residual(U.data, rep_q)
                kin_errors.append(k_res['error_m'])

            if term:
                break
        if term:
            break

    return {
        'grasp': bool(grasped),
        'lift': float(lifted),
        'succ': bool(info.get('success', False)),
        'otd': float(info.get('obj_to_target_dist', 9.0)),
        'end_grasped': bool(info.get('is_grasped', False)),
        'mean_torque_res': float(np.mean(torque_residuals)) if torque_residuals else 0.0,
        'max_torque_res': float(np.max(torque_residuals)) if torque_residuals else 0.0,
        'mean_kin_err_m': float(np.mean(kin_errors)) if kin_errors else 0.0,
        'max_kin_err_m': float(np.max(kin_errors)) if kin_errors else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy', required=True, help="'zeroshot' or path/HF repo ID")
    parser.add_argument('--dataset-root', default='data/cube500')
    parser.add_argument('--dataset-repo-id', default='local/cube500')
    parser.add_argument('--episodes', type=int, default=300, help='Total ramp episodes (default: 300)')
    parser.add_argument('--start-deg', type=float, default=0.0, help='Initial fault in degrees')
    parser.add_argument('--max-deg', type=float, default=8.0, help='Final fault in degrees')
    parser.add_argument('--fault-joint', default='shoulder_lift',
                        choices=['shoulder_lift', 'elbow_flex', 'wrist_flex', 'all'])
    parser.add_argument('--tag', required=True, help='Experiment label tag')
    parser.add_argument('--out-dir', default='eval_out', help='Output directory')
    parser.add_argument('--seed-start', type=int, default=10000, help='Starting seed')
    parser.add_argument('--min-reach', type=float, default=0.23)
    parser.add_argument('--max-reach', type=float, default=0.27)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Environment
    realism.install()
    cfg = PickAndPlaceConfig(
        obs_mode='visual',
        observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
        objects=[CubeObject(half_size=HALF, color='red')])
    base_env = PickAndPlaceEnv(config=cfg, render_mode='rgb_array')
    U = base_env.unwrapped
    realism.tune_arm(U.model)
    realism.set_camera(base_env, 'front_low')
    
    # Grip contacts
    for gid in set(U._gripper_geom_ids) | set(U._jaw_geom_ids) | set(U._obj_geom_ids if hasattr(U, "_obj_geom_ids") else [U._obj_geom_id]):
        f = U.model.geom_friction[gid].copy()
        U.model.geom_friction[gid] = [max(f[0], 1.2), max(f[1] * 10, 0.05), max(f[2] * 10, 0.01)]

    # Wrap with dynamic fault injector
    wrapper = EncoderOffset(base_env, delta=np.zeros(6, dtype=np.float64))
    
    # Detectors
    torque_det = TorqueResidualDetector(U.model)
    kin_det = KinematicResidualDetector(U.model)

    # Load policy
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root) if args.policy == 'zeroshot' else None
    print(f'Loading policy: {args.policy}')
    pol, pre, post = build_policy(args.policy, meta)

    # Calibrate detectors on first 10 nominal seeds (delta=0)
    print(">>> Calibrating detectors on nominal baseline...")
    calib_torques = []
    calib_kins = []
    curr_seed = args.seed_start
    calib_count = 0
    while calib_count < 10:
        wrapper.set_delta(np.zeros(6, dtype=np.float64))
        res = run_ramp_episode(wrapper, wrapper, pol, pre, post, seed=curr_seed,
                               min_reach=args.min_reach, max_reach=args.max_reach,
                               torque_det=torque_det, kin_det=kin_det)
        curr_seed += 1
        if res is not None:
            calib_torques.append(res['mean_torque_res'])
            calib_kins.append(res['mean_kin_err_m'])
            calib_count += 1

    torque_det.cusum.calibrate(np.array(calib_torques))
    kin_det.cusum.calibrate(np.array(calib_kins))
    print(f"Torque Detector threshold: {torque_det.cusum.threshold:.4f} (baseline mean: {torque_det.cusum.baseline_mean:.4f})")
    print(f"Kinematic Detector threshold: {kin_det.cusum.threshold:.4f} m (baseline mean: {kin_det.cusum.baseline_mean:.4f} m)")

    # Execute Continuous Fault Ramp
    print(f"\n>>> Launching {args.episodes}-Episode Fault Ramp: {args.start_deg} deg -> {args.max_deg} deg on {args.fault_joint} <<<")
    ramp_records = []
    scored_episodes = 0
    t0 = time.time()

    torque_alarm_ep = None
    kin_alarm_ep = None
    first_fail_ep = None

    while scored_episodes < args.episodes:
        # Linear fault ramp
        prog = float(scored_episodes) / max(1, args.episodes - 1)
        deg = args.start_deg + prog * (args.max_deg - args.start_deg)
        rad = np.deg2rad(deg)
        delta_arr = np.zeros(6, dtype=np.float64)
        if args.fault_joint == 'shoulder_lift':
            delta_arr[1] = rad
        elif args.fault_joint == 'elbow_flex':
            delta_arr[2] = rad
        elif args.fault_joint == 'wrist_flex':
            delta_arr[3] = rad
        elif args.fault_joint == 'all':
            delta_arr[1] = rad
            delta_arr[2] = rad
            delta_arr[3] = rad
        wrapper.set_delta(delta_arr)

        res = run_ramp_episode(wrapper, wrapper, pol, pre, post, seed=curr_seed,
                               min_reach=args.min_reach, max_reach=args.max_reach,
                               torque_det=torque_det, kin_det=kin_det)
        curr_seed += 1
        if res is None:
            continue

        scored_episodes += 1
        t_alarm = bool(torque_det.cusum.triggered)
        k_alarm = bool(kin_det.cusum.triggered)
        
        if t_alarm and torque_alarm_ep is None:
            torque_alarm_ep = scored_episodes
        if k_alarm and kin_alarm_ep is None:
            kin_alarm_ep = scored_episodes
        if not res['succ'] and first_fail_ep is None and scored_episodes > 10:
            first_fail_ep = scored_episodes

        rec = {
            'episode': scored_episodes,
            'delta_deg': round(deg, 4),
            'grasp': int(res['grasp']),
            'lift': round(res['lift'], 4),
            'succ': int(res['succ']),
            'otd': round(res['otd'], 4),
            'mean_torque_res': round(res['mean_torque_res'], 4),
            'max_torque_res': round(res['max_torque_res'], 4),
            'torque_alarm': int(t_alarm),
            'mean_kin_err_m': round(res['mean_kin_err_m'], 5),
            'max_kin_err_m': round(res['max_kin_err_m'], 5),
            'kin_alarm': int(k_alarm),
        }
        ramp_records.append(rec)

        if scored_episodes % 25 == 0 or scored_episodes == args.episodes:
            recent = ramp_records[-25:]
            recent_grasp = sum(r['grasp'] for r in recent) / len(recent) * 100
            recent_succ = sum(r['succ'] for r in recent) / len(recent) * 100
            elapsed = time.time() - t0
            print(f"Ep [{scored_episodes:3d}/{args.episodes}] | delta: {deg:4.2f} deg | "
                  f"Grasp: {recent_grasp:5.1f}% | Succ: {recent_succ:5.1f}% | "
                  f"TorqueAlarm: {t_alarm} | KinAlarm: {k_alarm} | Time: {elapsed:.1f}s")

    # Save episode telemetry DataFrame
    df = pd.DataFrame(ramp_records)
    csv_path = out_dir / f"ramp_{args.tag}_episodes.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved episode records to: {csv_path}")

    # Compute high-level summary
    summary = {
        'tag': args.tag,
        'policy': args.policy,
        'episodes': args.episodes,
        'fault_joint': args.fault_joint,
        'start_deg': args.start_deg,
        'max_deg': args.max_deg,
        'overall_grasp_rate': float(df['grasp'].mean()),
        'overall_succ_rate': float(df['succ'].mean()),
        'torque_alarm_episode': torque_alarm_ep,
        'kin_alarm_episode': kin_alarm_ep,
        'first_fail_episode': first_fail_ep,
        'torque_lead_episodes': (first_fail_ep - torque_alarm_ep) if (first_fail_ep and torque_alarm_ep) else None,
        'kin_lead_episodes': (first_fail_ep - kin_alarm_ep) if (first_fail_ep and kin_alarm_ep) else None,
        'corr_delta_torque': float(df['delta_deg'].corr(df['mean_torque_res'])),
        'corr_delta_kin_err': float(df['delta_deg'].corr(df['mean_kin_err_m'])),
    }
    json_path = out_dir / f"ramp_{args.tag}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {json_path}")
    print("\n=== FAULT RAMP COMPLETED ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
