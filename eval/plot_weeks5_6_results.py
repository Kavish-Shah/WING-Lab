#!/usr/bin/env python3
"""
Weeks 5-6 Results Visualizer & Paper Figures Generator
======================================================
Plots publication-ready figures for:
- RQ1: Physical wear masking curves (Grasp & Success vs. Fault Angle delta)
- RQ2: Multi-modal residual detectors (Kinematic & Torque residuals with CUSUM alarms)
- RQ2: Latent activation probe predictions (True vs. Predicted delta)
- Aggregate comparison metrics across Baseline and Masker tiers.
"""

import os
import sys
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration for clean publication figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

POLICY_COLORS = {
    'baseline': '#d9534f',       # Red
    'masker_2deg': '#337ab7',    # Blue
    'masker_5deg': '#f0ad4e',    # Orange / Amber
    'masker_10deg': '#5cb85c',   # Green
}

POLICY_LABELS = {
    'baseline': 'Baseline Champion (0° relabel)',
    'masker_2deg': 'Weak Masker (±2° relabel)',
    'masker_5deg': 'Medium Masker (±5° relabel)',
    'masker_10deg': 'Strong Masker (±10° relabel)',
}

def load_data(eval_dir: pathlib.Path):
    data = {}
    for tag in ['baseline', 'masker_2deg', 'masker_5deg', 'masker_10deg']:
        csv_file = eval_dir / f"ramp_{tag}_episodes.csv"
        json_file = eval_dir / f"ramp_{tag}_summary.json"
        if csv_file.exists() and json_file.exists():
            df = pd.read_csv(csv_file)
            with open(json_file, 'r') as f:
                summary = json.load(f)
            data[tag] = {'df': df, 'summary': summary}
            print(f"Loaded {tag}: {len(df)} episodes")
        else:
            print(f"Notice: {tag} not yet fully available in {eval_dir}")
    return data

def plot_rq1_wear_masking(data: dict, out_dir: pathlib.Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    window = 5
    for tag, d in data.items():
        df = d['df']
        color = POLICY_COLORS.get(tag, '#555555')
        label = POLICY_LABELS.get(tag, tag)
        
        # Smoothed grasp rate
        roll_grasp = df['grasp'].rolling(window, min_periods=1, center=True).mean() * 100
        roll_succ = df['succ'].rolling(window, min_periods=1, center=True).mean() * 100
        
        ax1.plot(df['delta_deg'], roll_grasp, color=color, linewidth=2.2, label=f"{label} (Grasp)")
        ax1.plot(df['delta_deg'], roll_succ, color=color, linewidth=1.5, linestyle='--', alpha=0.85, label=f"{label} (Success)")
        
    ax1.set_ylabel('Success / Grasp Rate (%)')
    ax1.set_ylim(-5, 105)
    ax1.grid(True)
    ax1.set_title('RQ1: Task Performance Under Progressive Joint Fault Ramp (0° → 8°)')
    ax1.legend(loc='lower left', ncol=2, framealpha=0.9)

    # Telemetry Error (Wear Masking Discrepancy)
    # The policy reports nominal proprioception (0 error), while actual joint offset is delta_deg
    sample_tag = next(iter(data.keys()))
    delta_vals = data[sample_tag]['df']['delta_deg']
    reported_error = np.zeros_like(delta_vals)
    
    ax2.plot(delta_vals, delta_vals, color='#d9534f', linewidth=2.0, label='True Physical Joint Wear (δ)')
    ax2.plot(delta_vals, reported_error, color='#222222', linewidth=2.0, linestyle=':', label='Internal Telemetry Reported Error (0.00°)')
    ax2.fill_between(delta_vals, reported_error, delta_vals, color='#f2dede', alpha=0.6, label='Silent Wear Masking Zone')
    
    ax2.set_xlabel('Fault Joint Angle δ (deg on shoulder_lift)')
    ax2.set_ylabel('Joint Error (deg)')
    ax2.set_ylim(-0.5, 9.0)
    ax2.grid(True)
    ax2.set_title('Proprioceptive Blindness: Reported vs. Actual Joint Fault')
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_path = out_dir / "fig1_rq1_wear_masking_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_rq2_detectors(data: dict, out_dir: pathlib.Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for tag, d in data.items():
        df = d['df']
        color = POLICY_COLORS.get(tag, '#555555')
        label = POLICY_LABELS.get(tag, tag)
        
        # Kinematic residual (mm)
        kin_mm = df['mean_kin_err_m'] * 1000.0
        ax1.plot(df['delta_deg'], kin_mm, color=color, linewidth=2.0, marker='o', markersize=3.5, label=label)
        
        # Torque residual (Nm)
        torque_nm = df['mean_torque_res']
        ax2.plot(df['delta_deg'], torque_nm, color=color, linewidth=2.0, marker='s', markersize=3.5, label=label)

    ax1.set_xlabel('Physical Fault Angle δ (deg)')
    ax1.set_ylabel('Kinematic End-Effector Error (mm)')
    ax1.set_title('(a) Kinematic Visual Residual (FK Discrepancy)')
    ax1.grid(True)
    ax1.legend(loc='upper left', framealpha=0.9)

    ax2.set_xlabel('Physical Fault Angle δ (deg)')
    ax2.set_ylabel('Mean Torque Residual ‖τ_meas - τ_model‖ (Nm)')
    ax2.set_title('(b) Actuator Dynamics Torque Residual')
    ax2.grid(True)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.suptitle('RQ2: Multi-Modal Detection Signals Tracking Joint Damage', y=1.02)
    plt.tight_layout()
    save_path = out_dir / "fig2_rq2_residual_detectors_cusum.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_rq2_activation_probe(eval_dir: pathlib.Path, out_dir: pathlib.Path):
    probe_summary_file = None
    for candidate in [eval_dir / "probe_masker_10deg_summary.json", eval_dir / "activation_probe_summary.json"]:
        if candidate.exists():
            probe_summary_file = candidate
            break

    if not probe_summary_file:
        print("Notice: Activation probe summary not yet available.")
        return

    with open(probe_summary_file, 'r') as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ['Train R²', 'Test R²', 'Train MAE (deg)', 'Test MAE (deg)']
    values = [
        summary.get('train_r2', 0.0),
        summary.get('test_r2', 0.0),
        summary.get('train_mae_deg', 0.0),
        summary.get('test_mae_deg', 0.0)
    ]
    colors = ['#5cb85c', '#337ab7', '#f0ad4e', '#d9534f']

    bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor='#333333', linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel('Metric Score')
    ax.set_title('Latent Activation Probe: Decoding Hidden Joint Fault from VLM Embeddings')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = out_dir / "fig3_rq2_activation_probe.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_aggregate_bar_comparison(data: dict, out_dir: pathlib.Path):
    tags = list(data.keys())
    if not tags:
        return

    labels = [POLICY_LABELS.get(t, t).split(' (')[0] for t in tags]
    grasp_rates = [data[t]['summary']['overall_grasp_rate'] * 100 for t in tags]
    succ_rates = [data[t]['summary']['overall_succ_rate'] * 100 for t in tags]
    kin_corrs = [data[t]['summary']['corr_delta_kin_err'] for t in tags]
    torque_corrs = [data[t]['summary']['corr_delta_torque'] for t in tags]

    x = np.arange(len(tags))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5*width, grasp_rates, width, label='Overall Grasp Rate (%)', color='#337ab7')
    ax.bar(x - 0.5*width, succ_rates, width, label='Overall Strict Success (%)', color='#5cb85c')
    ax.bar(x + 0.5*width, [c * 100 for c in kin_corrs], width, label='Kinematic Discrepancy Corr (r × 100)', color='#f0ad4e')
    ax.bar(x + 1.5*width, [c * 100 for c in torque_corrs], width, label='Torque Residual Corr (r × 100)', color='#d9534f')

    ax.set_ylabel('Percentage / Scaled Metric')
    ax.set_title('Aggregate Performance & Modality Correlation Across Evaluated Policies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save_path = out_dir / "fig4_aggregate_policy_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "eval_out"
    fig_dir = eval_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading evaluation data from: {eval_dir}")
    data = load_data(eval_dir)

    if not data:
        print("No evaluation data found. Exiting.")
        sys.exit(1)

    plot_rq1_wear_masking(data, fig_dir)
    plot_rq2_detectors(data, fig_dir)
    plot_rq2_activation_probe(eval_dir, fig_dir)
    plot_aggregate_bar_comparison(data, fig_dir)
    print("\nAll figures generated successfully in:", fig_dir)

if __name__ == '__main__':
    main()
