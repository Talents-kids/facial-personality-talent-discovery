#!/usr/bin/env python3
"""
Generate Figure 2: Human Expert Baseline Comparison
For Frontiers in Psychology submission.

This script generates the 2-panel figure showing:
- Panel A: AI vs Human Experts prediction accuracy (correlation with self-report)
- Panel B: Inter-rater reliability (ICC) between two clinical psychologists

Author: Talents.kids Research Team
Date: 2026-02-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette (Nature-style)
COLORS = {
    'ai': '#E64B35',           # Red - AI
    'expert_avg': '#3C5488',   # Dark blue - Expert average
}

# Big Five trait names
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
TRAIT_ABBREV = ['O', 'C', 'E', 'A', 'N']


def load_data():
    """Load the human expert baseline dataset."""
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / 'human_expert_baseline_complete.csv')
    return df


def calculate_correlations(df):
    """Calculate correlations between predictions and self-report.

    NOTE: Expert Avg is calculated as arithmetic mean of individual expert correlations:
    Expert Avg = (r_exp1 + r_exp2) / 2

    This is consistent with Table 4 in the manuscript.
    """
    results = []

    for trait, abbrev in zip(TRAITS, TRAIT_ABBREV):
        self_col = f'self_{abbrev}'
        exp1_col = f'exp1_{abbrev}'
        exp2_col = f'exp2_{abbrev}'
        ai_col = f'ai_{abbrev}'

        # Correlations with self-report
        r_exp1 = df[exp1_col].corr(df[self_col])
        r_exp2 = df[exp2_col].corr(df[self_col])

        # Expert Avg = arithmetic mean of two expert correlations
        # (NOT correlation of averaged ratings)
        r_exp_avg = (r_exp1 + r_exp2) / 2

        r_ai = df[ai_col].corr(df[self_col])

        results.append({
            'trait': trait,
            'abbrev': abbrev,
            'Expert 1': r_exp1,
            'Expert 2': r_exp2,
            'Expert Avg': r_exp_avg,
            'AI': r_ai
        })

    return pd.DataFrame(results)


def calculate_icc(df):
    """Calculate ICC between two experts for each trait."""
    icc_results = []

    for trait, abbrev in zip(TRAITS, TRAIT_ABBREV):
        exp1 = df[f'exp1_{abbrev}'].values
        exp2 = df[f'exp2_{abbrev}'].values

        # ICC(2,1) - two-way random effects, single measures
        n = len(exp1)
        mean_all = (exp1.mean() + exp2.mean()) / 2

        # Between-subjects variance
        subject_means = (exp1 + exp2) / 2
        ss_subjects = 2 * np.sum((subject_means - mean_all) ** 2)
        ms_subjects = ss_subjects / (n - 1)

        # Within-subjects variance (rater effect)
        rater_means = np.array([exp1.mean(), exp2.mean()])
        ss_raters = n * np.sum((rater_means - mean_all) ** 2)
        ms_raters = ss_raters / 1  # df = k-1 = 1

        # Residual variance
        ss_total = np.sum((exp1 - mean_all) ** 2) + np.sum((exp2 - mean_all) ** 2)
        ss_residual = ss_total - ss_subjects - ss_raters
        ms_residual = ss_residual / (n - 1)

        # ICC(2,1)
        icc = (ms_subjects - ms_residual) / (ms_subjects + ms_residual + 2 * (ms_raters - ms_residual) / n)

        icc_results.append({
            'trait': trait,
            'abbrev': abbrev,
            'ICC': icc
        })

    return pd.DataFrame(icc_results)


def generate_figure2(df, output_dir):
    """
    Generate Figure 2: 2-panel summary figure for main manuscript.
    Panel A: Prediction accuracy (AI vs Human Experts)
    Panel B: Inter-rater reliability (ICC)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate metrics
    corr_df = calculate_correlations(df)
    icc_df = calculate_icc(df)

    # =========================================================================
    # Panel A: Correlation comparison (AI vs Human Experts)
    # =========================================================================
    x = np.arange(len(TRAITS))
    width = 0.35

    bars1 = ax1.bar(x - width/2, corr_df['Expert Avg'], width,
                    label='Human Experts', color=COLORS['expert_avg'], edgecolor='white')
    bars2 = ax1.bar(x + width/2, corr_df['AI'], width,
                    label='AI System', color=COLORS['ai'], edgecolor='white')

    ax1.set_ylabel('Correlation with Self-Report (r)')
    ax1.set_xlabel('Personality Trait')
    ax1.set_title('A. Prediction Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(TRAIT_ABBREV)
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_ylim(0, 0.5)

    # Add mean dashed lines - CORRECTED VALUES
    mean_exp = corr_df['Expert Avg'].mean()  # Should be ~0.291
    mean_ai = corr_df['AI'].mean()           # Should be ~0.351

    ax1.axhline(y=mean_exp, color=COLORS['expert_avg'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=mean_ai, color=COLORS['ai'], linestyle='--', alpha=0.7, linewidth=1.5)

    # Print verification
    print(f"\nPanel A verification:")
    print(f"  Mean Expert Avg: {mean_exp:.3f}")
    print(f"  Mean AI: {mean_ai:.3f}")
    print(f"  Improvement: +{(mean_ai - mean_exp)*100:.1f}%")

    # =========================================================================
    # Panel B: ICC (Inter-Rater Reliability)
    # =========================================================================
    y_pos = np.arange(len(TRAITS))
    bars = ax2.barh(y_pos, icc_df['ICC'], color=COLORS['expert_avg'], edgecolor='white', height=0.6)

    # Add value labels
    for i, icc in enumerate(icc_df['ICC']):
        ax2.text(icc + 0.01, i, f'{icc:.2f}', va='center', fontsize=9)

    # Add 0.70 threshold line
    ax2.axvline(x=0.70, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Good (0.70)')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(TRAIT_ABBREV)
    ax2.set_xlabel('ICC(2,1)')
    ax2.set_title('B. Inter-Rater Reliability')
    ax2.set_xlim(0, 1)
    ax2.legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    # Save to figures directory
    plt.savefig(output_dir / 'figure_human_expert_baseline.png', dpi=300)
    plt.savefig(output_dir / 'figure_human_expert_baseline.pdf')
    plt.close()

    print(f"\n✓ Figure 2 saved to: {output_dir}")

    return corr_df, icc_df


def main():
    """Generate Figure 2."""
    print("=" * 60)
    print("Figure 2: Human Expert Baseline Comparison")
    print("For Frontiers in Psychology")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"\n✓ Loaded {len(df)} records")

    # Output directory (figures folder)
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Generate figure
    corr_df, icc_df = generate_figure2(df, output_dir)

    # Print summary tables
    print("\n" + "=" * 60)
    print("CORRELATION WITH SELF-REPORT")
    print("=" * 60)
    print("\nTrait\t\tExpert 1\tExpert 2\tExpert Avg\tAI")
    print("-" * 70)
    for _, row in corr_df.iterrows():
        print(f"{row['trait']:<15}\t{row['Expert 1']:.3f}\t\t{row['Expert 2']:.3f}\t\t{row['Expert Avg']:.3f}\t\t{row['AI']:.3f}")

    print("-" * 70)
    print(f"{'MEAN':<15}\t{corr_df['Expert 1'].mean():.3f}\t\t{corr_df['Expert 2'].mean():.3f}\t\t{corr_df['Expert Avg'].mean():.3f}\t\t{corr_df['AI'].mean():.3f}")

    # Verify Expert Avg is (Expert1 + Expert2) / 2
    print("\n" + "=" * 60)
    print("VERIFICATION: Expert Avg = (Expert1 + Expert2) / 2")
    print("=" * 60)
    for _, row in corr_df.iterrows():
        calculated = (row['Expert 1'] + row['Expert 2']) / 2
        match = "✓" if abs(calculated - row['Expert Avg']) < 0.001 else "✗"
        print(f"{row['trait']:<15}: ({row['Expert 1']:.3f} + {row['Expert 2']:.3f})/2 = {calculated:.3f} {match}")

    print("\n" + "=" * 60)
    print("ICC VALUES")
    print("=" * 60)
    for _, row in icc_df.iterrows():
        print(f"{row['trait']:<15}: {row['ICC']:.3f}")
    print(f"{'MEAN':<15}: {icc_df['ICC'].mean():.3f}")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
