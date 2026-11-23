"""Internal signal detectors for physical wear and damage masking.

Investigates Research Question 2 (RQ2):
What sensing modalities expose hidden mechanical damage (joint coupling slippage)
when motor encoders report normal, healthy readings (proprioceptive blindness)?

Three complementary detection modalities:
1. TorqueResidualDetector:
   Discrepancy between actual measured actuator torques and expected
   rigid-body dynamics / gravity compensation computed on reported encoder angles.
2. KinematicResidualDetector:
   Discrepancy between visual / physical end-effector pose and forward-kinematics
   projection from reported joint positions.
3. ActivationProbeDetector:
   Linear / ridge regression probe on VLA transformer latent representations
   to test if the neural network internally encodes the physical fault.
4. CUSUMDetector:
   Online cumulative sum change-point detector with calibrated alarm thresholds.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np


class CUSUMDetector:
    """Two-sided or one-sided Cumulative Sum (CUSUM) change-point detector.
    
    Accumulates evidence of a positive shift in signal mean above baseline noise.
    Alarm fires when cumulative deviation exceeds threshold h.
    """
    def __init__(self, drift: float = 0.05, threshold: float = 1.0, baseline_mean: float = 0.0):
        self.drift = float(drift)
        self.threshold = float(threshold)
        self.baseline_mean = float(baseline_mean)
        self.reset()

    def reset(self):
        self.S = 0.0
        self.triggered = False
        self.history: List[float] = []
        self.cusum_history: List[float] = []

    def calibrate(self, baseline_signals: np.ndarray, false_alarm_rate: float = 0.01):
        """Calibrate baseline mean, drift (allowance k), and threshold h from nominal data."""
        clean = np.asarray(baseline_signals, dtype=np.float64)
        self.baseline_mean = float(np.mean(clean))
        sigma = float(np.std(clean)) if len(clean) > 1 else 0.01
        self.drift = max(0.5 * sigma, 1e-4)
        
        # Test candidate thresholds on nominal data to guarantee low false alarm rate
        s_vals = []
        s = 0.0
        for x in clean:
            s = max(0.0, s + (x - self.baseline_mean) - self.drift)
            s_vals.append(s)
        
        pct = (1.0 - false_alarm_rate) * 100.0
        max_nominal_s = float(np.percentile(s_vals, pct)) if len(s_vals) > 0 else 3.0 * sigma
        self.threshold = max(max_nominal_s * 1.5, 3.0 * sigma)
        return self

    def update(self, val: float) -> Tuple[bool, float]:
        """Process a single measurement tick. Returns (is_alarm, current_cusum_statistic)."""
        val = float(val)
        self.history.append(val)
        self.S = max(0.0, self.S + (val - self.baseline_mean) - self.drift)
        self.cusum_history.append(self.S)
        if self.S >= self.threshold and not self.triggered:
            self.triggered = True
        return self.triggered, self.S


class TorqueResidualDetector:
    """Detects mechanical coupling fault by comparing physical actuator torque
    against the analytical rigid-body gravity compensation model evaluated at
    the reported (masked) joint angles.
    
    When an arm link has a physical offset delta, the required gravity torque
    tau_grav(q + delta) differs from expected tau_grav(q).
    """
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data_nominal = mujoco.MjData(model)
        self.cusum = CUSUMDetector(drift=0.02, threshold=0.3)
        self.residuals: List[float] = []

    def reset(self):
        self.cusum.reset()
        self.residuals.clear()

    def compute_residual(self, data_physical: mujoco.MjData, reported_qpos: np.ndarray) -> Dict[str, Any]:
        """Calculate torque residual between actual actuator force and expected static model."""
        # 1. Measured actuator torques (first 5 arm joints)
        tau_measured = np.array(data_physical.qfrc_actuator[:5], dtype=np.float64)
        
        # 2. Expected gravity torque at reported (encoder) posture
        self.data_nominal.qpos[:5] = reported_qpos[:5]
        self.data_nominal.qvel[:] = 0.0
        self.data_nominal.qacc[:] = 0.0
        # Inverse dynamics at zero vel/accel computes exact static gravity compensation
        mujoco.mj_inverse(self.model, self.data_nominal)
        tau_expected = np.array(self.data_nominal.qfrc_inverse[:5], dtype=np.float64)

        # Residual vector and L2 norm
        delta_tau = tau_measured - tau_expected
        res_norm = float(np.linalg.norm(delta_tau))
        self.residuals.append(res_norm)
        
        alarm, cusum_stat = self.cusum.update(res_norm)
        return {
            'tau_measured': tau_measured.tolist(),
            'tau_expected': tau_expected.tolist(),
            'residual_vector': delta_tau.tolist(),
            'residual_norm': res_norm,
            'alarm': alarm,
            'cusum_stat': cusum_stat
        }


class KinematicResidualDetector:
    """Detects fault by comparing the true physical end-effector 3D position
    (obtained in real life from visual tracking / external camera) against the
    forward kinematics projection computed from reported encoder angles.
    """
    def __init__(self, model: mujoco.MjModel, ee_site_name: str = 'gripperframe'):
        self.model = model
        self.data_nominal = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.site_id < 0:
            # Fallback to body or gripper site
            self.site_id = 1
        self.cusum = CUSUMDetector(drift=0.002, threshold=0.015) # meters
        self.residuals: List[float] = []

    def reset(self):
        self.cusum.reset()
        self.residuals.clear()

    def compute_residual(self, data_physical: mujoco.MjData, reported_qpos: np.ndarray) -> Dict[str, Any]:
        """Compute 3D position error between true physical end-effector and reported FK."""
        # Physical EE position
        if self.site_id > 0:
            p_true = np.array(data_physical.site_xpos[self.site_id], dtype=np.float64)
        else:
            p_true = np.array(data_physical.xpos[-1], dtype=np.float64)

        # Nominal FK position at reported angles
        self.data_nominal.qpos[: len(reported_qpos)] = reported_qpos
        mujoco.mj_kinematics(self.model, self.data_nominal)
        if self.site_id > 0:
            p_expected = np.array(self.data_nominal.site_xpos[self.site_id], dtype=np.float64)
        else:
            p_expected = np.array(self.data_nominal.xpos[-1], dtype=np.float64)

        pos_error = float(np.linalg.norm(p_true - p_expected))
        self.residuals.append(pos_error)
        
        alarm, cusum_stat = self.cusum.update(pos_error)
        return {
            'pos_true': p_true.tolist(),
            'pos_expected': p_expected.tolist(),
            'error_m': pos_error,
            'alarm': alarm,
            'cusum_stat': cusum_stat
        }


class ActivationProbeDetector:
    """Linear regression probe on internal transformer activations of the VLA policy.
    Tests whether latent embeddings internally represent the fault magnitude delta.
    """
    def __init__(self, weights: Optional[np.ndarray] = None, bias: float = 0.0, alarm_threshold_deg: float = 2.0):
        self.weights = weights
        self.bias = float(bias)
        self.alarm_threshold_deg = float(alarm_threshold_deg)
        self.predictions: List[float] = []

    def fit(self, activations: np.ndarray, targets_deg: np.ndarray, alpha: float = 1.0):
        """Fit ridge regression probe W to map feature vector -> fault delta in degrees."""
        X = np.asarray(activations, dtype=np.float64)
        y = np.asarray(targets_deg, dtype=np.float64)
        N, D = X.shape
        
        # Center features
        self.x_mean = np.mean(X, axis=0)
        Xc = X - self.x_mean
        y_mean = np.mean(y)
        yc = y - y_mean
        
        # Ridge regression solve: (X^T X + alpha * I)^{-1} X^T y
        reg = alpha * np.eye(D)
        self.weights = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ yc)
        self.bias = float(y_mean)
        
        # Training R^2 score
        y_pred = Xc @ self.weights + self.bias
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        return {'r2': float(r2), 'mae_deg': float(np.mean(np.abs(y - y_pred)))}

    def predict(self, activation_vector: np.ndarray) -> Tuple[float, bool]:
        """Predict fault angle delta in degrees from activation vector. Returns (pred_deg, is_alarm)."""
        if self.weights is None:
            return 0.0, False
        vec = np.asarray(activation_vector, dtype=np.float64)
        if hasattr(self, 'x_mean'):
            vec = vec - self.x_mean
        pred_deg = float(np.dot(vec, self.weights) + self.bias)
        self.predictions.append(pred_deg)
        is_alarm = abs(pred_deg) >= self.alarm_threshold_deg
        return pred_deg, is_alarm
