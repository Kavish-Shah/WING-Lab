"""VLA Internal Latent Activation Probe Trainer.

Evaluates Research Question 2 (Modality C):
Tests whether the Vision-Language-Action policy's internal latent features
encode the physical mechanical fault delta, even though internal joint encoders
report healthy nominal values (proprioceptive blindness).

Extracts transformer activations from the MolmoAct2 policy, trains a linear / ridge
regression probe to predict the fault angle, and evaluates R^2 and detection accuracy.

Saves:
- eval_out/probe_activations.npz (collected activations and true delta labels)
- eval_out/activation_probe.pt (trained probe weights and normalization parameters)
- eval_out/activation_probe_summary.json (cross-validated R^2, MAE, detection metrics)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if sys.platform == 'linux':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gymnasium as gym
import numpy as np
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
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

_orig_lora_target_modules = MolmoAct2Policy._lora_target_modules
def _champion_lora_target_modules(self, *, prefix: str, lm_head_path: str) -> str:
    vlm_linear_leaves = 'w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding'
    return rf'(?:{prefix}\.(transformer|vision_backbone|image_pooling_2d|image_projector)\.(?:.*\.)?({vlm_linear_leaves}))$'

from so101_nexus import OverheadCamera, WristCamera
from so101_nexus.mujoco.pick_and_place import PickAndPlaceConfig, PickAndPlaceEnv
from so101_nexus.objects import CubeObject

from molmoact2_so101_sim import realism
from eval.detectors import ActivationProbeDetector

HALF = 0.015
SUB = 6
JS = np.array([1., -1., 1., 1., 1., 1.])
JO = np.array([0., 90., 90., 0., 0., 0.])
TASK = 'Pick up the red cube and place it on the blue circle.'


class EncoderOffset(gym.Wrapper):
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


def obs_to_batch(obs):
    c0 = np.ascontiguousarray(obs['overhead_camera']).astype(np.uint8)
    w = np.ascontiguousarray(obs['wrist_camera']).astype(np.uint8)
    c1 = w if w.shape[:2] == (480, 640) else np.asarray(Image.fromarray(w).resize((640, 480))).astype(np.uint8)

    def img(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.).unsqueeze(0).to('cuda')

    sd = np.rad2deg(obs['state'])
    st = (JS * sd + JO).astype(np.float32)
    st[5] = float(np.clip((sd[5] + 10.) / 110. * 45., 0, 45))
    return {'observation.images.cam0': img(c0), 'observation.images.cam1': img(c1),
            'observation.state': torch.from_numpy(st).unsqueeze(0).to('cuda'), 'task': [TASK]}


def act_to_sim(a, lo, hi):
    arm = np.deg2rad((a[:5] - JO[:5]) / JS[:5])
    grip = np.deg2rad(float(a[5] / 45. * 110. - 10.))
    return np.clip(np.concatenate([arm, [grip]]).astype(np.float32), lo, hi)


class ActivationHook:
    """Hooks into a PyTorch module to capture forward activations."""
    def __init__(self):
        self.features = []

    def hook_fn(self, module, input, output):
        # Extract mean-pooled feature representation from output tensor
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if isinstance(out, torch.Tensor):
            feat = out.detach().cpu().float()
            if feat.ndim > 2:
                feat = feat.mean(dim=list(range(1, feat.ndim - 1)))
            self.features.append(feat.squeeze(0).numpy())

    def clear(self):
        self.features.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, help="'zeroshot' or HF model repo ID")
    parser.add_argument('--dataset-root', default='data/cube500')
    parser.add_argument('--dataset-repo-id', default='local/cube500')
    parser.add_argument('--episodes-per-fault', type=int, default=10, help='Episodes per delta level')
    parser.add_argument('--fault-sweep', default='0,1,2,4,6,8', help='Comma-separated fault angles in degrees')
    parser.add_argument('--out-dir', default='eval_out')
    parser.add_argument('--tag', default='activation_probe')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup Environment
    realism.install()
    cfg = PickAndPlaceConfig(
        obs_mode='visual',
        observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
        objects=[CubeObject(half_size=HALF, color='red')])
    base_env = PickAndPlaceEnv(config=cfg, render_mode='rgb_array')
    U = base_env.unwrapped
    realism.tune_arm(U.model)
    realism.set_camera(base_env, 'front_low')
    wrapper = EncoderOffset(base_env, delta=np.zeros(6))

    # Load policy
    if 'champion' in str(args.policy):
        MolmoAct2Policy._lora_target_modules = _champion_lora_target_modules
    else:
        MolmoAct2Policy._lora_target_modules = _orig_lora_target_modules

    pol = MolmoAct2Policy.from_pretrained(args.policy)
    pol.config.inference_action_mode = 'continuous'
    pol.to('cuda')
    pol.eval()
    pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=args.policy)

    # Attach hook to action expert or final transformer layer
    hook = ActivationHook()
    target_layer = None
    if hasattr(pol.model, 'action_expert'):
        target_layer = pol.model.action_expert
        print("Hooked into policy.model.action_expert")
    elif hasattr(pol.model, 'transformer'):
        target_layer = pol.model.transformer
        print("Hooked into policy.model.transformer")
    else:
        for name, m in pol.model.named_modules():
            if 'action' in name or 'head' in name:
                target_layer = m
                print(f"Hooked into fallback module: {name}")
                break

    if target_layer is not None:
        target_layer.register_forward_hook(hook.hook_fn)

    fault_levels = [float(x) for x in args.fault_sweep.split(',')]
    collected_acts = []
    collected_deltas = []

    print(f"\n>>> Collecting activations across fault levels: {fault_levels} <<<")
    seed_idx = 20000
    lo, hi = base_env.action_space.low, base_env.action_space.high

    for delta_deg in fault_levels:
        rad = np.deg2rad(delta_deg)
        delta_arr = np.zeros(6)
        delta_arr[1] = rad  # shoulder_lift
        wrapper.set_delta(delta_arr)

        scored = 0
        while scored < args.episodes_per_fault:
            obs, _ = wrapper.reset(seed=seed_idx)
            seed_idx += 1
            r_obj = float(np.hypot(*U._get_object_pose()[:2]))
            if r_obj > 0.27 or r_obj < 0.23:
                continue

            scored += 1
            pol.reset()
            hook.clear()

            # Execute 15 steps to collect latent trajectory
            for _ in range(15):
                batch = obs_to_batch(obs)
                proc = pre(dict(batch)) if pre is not None else batch
                with torch.inference_mode():
                    a_raw = pol.select_action(proc)
                a = post(a_raw)[0].detach().cpu().numpy() if post is not None else a_raw[0].detach().cpu().numpy()
                q = act_to_sim(a, lo, hi)
                obs, _r, term, _t2, _info = wrapper.step(q)
                if term:
                    break

            if hook.features:
                ep_act = np.mean(hook.features, axis=0)
                collected_acts.append(ep_act)
                collected_deltas.append(delta_deg)

        print(f"Finished delta = {delta_deg:4.1f} deg | Collected {scored} episodes")

    X = np.array(collected_acts)
    y = np.array(collected_deltas)
    print(f"\nFeature matrix X shape: {X.shape}, Targets y shape: {y.shape}")

    # Train-test split (80/20)
    indices = np.random.RandomState(42).permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit Ridge Regression Probe
    probe = ActivationProbeDetector()
    train_metrics = probe.fit(X_train, y_train, alpha=10.0)

    # Evaluate on test split
    test_preds = np.array([probe.predict(x)[0] for x in X_test])
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_res = np.sum((y_test - test_preds) ** 2)
    test_r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    test_mae = float(np.mean(np.abs(y_test - test_preds)))

    print("\n=== ACTIVATION PROBE RESULTS ===")
    print(f"Train R^2: {train_metrics['r2']:.4f} | Train MAE: {train_metrics['mae_deg']:.3f} deg")
    print(f"Test R^2:  {test_r2:.4f} | Test MAE:  {test_mae:.3f} deg")

    # Save summary and probe
    summary = {
        'tag': args.tag,
        'policy': args.policy,
        'num_samples': len(X),
        'feature_dim': int(X.shape[1]),
        'train_r2': train_metrics['r2'],
        'train_mae_deg': train_metrics['mae_deg'],
        'test_r2': test_r2,
        'test_mae_deg': test_mae,
    }
    with open(out_dir / f"{args.tag}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(out_dir / f"{args.tag}_dataset.npz", X=X, y=y)
    torch.save({'weights': probe.weights, 'bias': probe.bias, 'x_mean': probe.x_mean},
               out_dir / f"{args.tag}_model.pt")
    print(f"Saved probe model and metrics to {out_dir}/")


if __name__ == '__main__':
    main()
