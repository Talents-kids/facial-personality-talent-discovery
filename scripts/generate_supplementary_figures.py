#!/usr/bin/env python3
"""
Generate Supplementary Figures for Human Expert Baseline Comparison.
For Frontiers in Psychology submission.

Generates:
- Figure S1 (figure_a): AI vs Human Experts Correlation Comparison
- Figure S2 (figure_b): Inter-rater Reliability (ICC)
- Figure S3 (figure_c): Prediction Error Distribution
- Figure S4 (figure_d): Expert Agreement Scatter Plots

Author: Talents.kids Research Team
Date: 2026-01-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

# Color palette
COLORS = {
    'ai': '#E64B35',           # Red - AI
    'expert1': '#4DBBD5',      # Cyan - Expert 1
    'expert2': '#00A087',      # Teal - Expert 2
    'expert_avg': '#3C5488',   # Dark blue - Expert average
    'neutral': '#B09C85'       # Beige - neutral
}

# Big Five trait names
TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
TRAIT_ABBREV = ['O', 'C', 'E', 'A', 'N']


def load_data():
    """Load the human expert baseline dataset."""
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir.parent / 'data' / 'human_expert_baseline' / 'human_expert_baseline_complete.csv')
    return df


def calculate_correlations(df):
    """Calculate correlations - Expert Avg as mean of correlations."""
    results = []

    for trait, abbrev in zip(TRAITS, TRAIT_ABBREV):
        self_col = f'self_{abbrev}'
        exp1_col = f'exp1_{abbrev}'
        exp2_col = f'exp2_{abbrev}'
        ai_col = f'ai_{abbrev}'

        r_exp1 = df[exp1_col].corr(df[self_col])
        r_exp2 = df[exp2_col].corr(df[self_col])
        r_exp_avg = (r_exp1 + r_exp2) / 2  # Arithmetic mean of correlations
        r_ai = df[ai_col].corr(df[self_col])

        results.append({
            'trait': trait,
            'Expert 1': r_exp1,
            'Expert 2': r_exp2,
            'Expert Avg': r_exp_avg,
            'AI': r_ai
        })

    return pd.DataFrame(results)


def calculate_icc(df):
    """Calculate ICC between two experts."""
    icc_results = []

    for trait, abbrev in zip(TRAITS, TRAIT_ABBREV):
        exp1 = df[f'exp1_{abbrev}'].values
        exp2 = df[f'exp2_{abbrev}'].values

        n = len(exp1)
        mean_all = (exp1.mean() + exp2.mean()) / 2

        subject_means = (exp1 + exp2) / 2
        ss_subjects = 2 * np.sum((subject_means - mean_all) ** 2)
        ms_subjects = ss_subjects / (n - 1)

        rater_means = np.array([exp1.mean(), exp2.mean()])
        ss_raters = n * np.sum((rater_means - mean_all) ** 2)
        ms_raters = ss_raters / 1

        ss_total = np.sum((exp1 - mean_all) ** 2) + np.sum((exp2 - mean_all) ** 2)
        ss_residual = ss_total - ss_subjects - ss_raters
        ms_residual = ss_residual / (n - 1)

        icc = (ms_subjects - ms_residual) / (ms_subjects + ms_residual + 2 * (ms_raters - ms_residual) / n)

        icc_results.append({
            'trait': trait,
            'ICC': icc
        })

    return pd.DataFrame(icc_results)


def figure_a_correlation_comparison(df, output_dir):
    """Figure S1: Bar chart comparing AI vs Human Expert correlations."""
    corr_df = calculate_correlations(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(TRAITS))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, corr_df['Expert 1'], width,
                   label='Expert 1', color=COLORS['expert1'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, corr_df['Expert 2'], width,
                   label='Expert 2', color=COLORS['expert2'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, corr_df['Expert Avg'], width,
                   label='Expert Avg', color=COLORS['expert_avg'], edgecolor='white', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, corr_df['AI'], width,
                   label='AI', color=COLORS['ai'], edgecolor='white', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Correlation with Self-Report (r)')
    ax.set_xlabel('Big Five Personality Trait')
    ax.set_title('A. Prediction Accuracy: AI vs Human Experts')
    ax.set_xticks(x)
    ax.set_xticklabels(TRAITS, rotation=15, ha='right')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, 0.5)

    # AI advantage annotation - CORRECTED
    avg_ai = corr_df['AI'].mean()
    avg_exp = corr_df['Expert Avg'].mean()
    improvement = (avg_ai - avg_exp) / avg_exp * 100
    ax.text(0.98, 0.02, f'AI advantage: +{improvement:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, fontstyle='italic', color=COLORS['ai'])

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_a_correlation_comparison.png', dpi=300)
    plt.savefig(output_dir / 'figure_a_correlation_comparison.pdf')
    plt.close()

    print(f"✓ Figure S1 (A) saved")
    return corr_df


def figure_b_icc_visualization(df, output_dir):
    """Figure S2: ICC visualization."""
    icc_df = calculate_icc(df)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(TRAITS))
    colors = [COLORS['expert_avg'] if icc > 0.7 else COLORS['neutral'] for icc in icc_df['ICC']]

    bars = ax.barh(y_pos, icc_df['ICC'], color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for i, (bar, icc) in enumerate(zip(bars, icc_df['ICC'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{icc:.3f}', va='center', ha='left', fontsize=10)

    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate (0.50)')
    ax.axvline(x=0.75, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Good (0.75)')
    ax.axvline(x=0.9, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.90)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(TRAITS)
    ax.set_xlabel('Intraclass Correlation Coefficient ICC(2,1)')
    ax.set_title('B. Inter-Rater Reliability Between Clinical Psychologists')
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_b_icc_reliability.png', dpi=300)
    plt.savefig(output_dir / 'figure_b_icc_reliability.pdf')
    plt.close()

    print(f"✓ Figure S2 (B) saved")
    return icc_df


def figure_c_error_distribution(df, output_dir):
    """Figure S3: Violin plots showing prediction error distribution."""
    error_data = []

    for trait, abbrev in zip(TRAITS, TRAIT_ABBREV):
        self_col = f'self_{abbrev}'
        exp1_col = f'exp1_{abbrev}'
        exp2_col = f'exp2_{abbrev}'
        ai_col = f'ai_{abbrev}'

        exp_avg = (df[exp1_col] + df[exp2_col]) / 2

        for i in range(len(df)):
            error_data.append({
                'Trait': trait,
                'Method': 'Expert Avg',
                'Absolute Error': abs(exp_avg.iloc[i] - df[self_col].iloc[i])
            })
            error_data.append({
                'Trait': trait,
                'Method': 'AI',
                'Absolute Error': abs(df[ai_col].iloc[i] - df[self_col].iloc[i])
            })

    error_df = pd.DataFrame(error_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    palette = {'Expert Avg': COLORS['expert_avg'], 'AI': COLORS['ai']}

    sns.violinplot(data=error_df, x='Trait', y='Absolute Error', hue='Method',
                   split=True, inner='quart', palette=palette, ax=ax)

    ax.set_ylabel('Absolute Prediction Error')
    ax.set_xlabel('Big Five Personality Trait')
    ax.set_title('C. Prediction Error Distribution: AI vs Human Experts')
    ax.legend(title='Method', loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_c_error_distribution.png', dpi=300)
    plt.savefig(output_dir / 'figure_c_error_distribution.pdf')
    plt.close()

    print(f"✓ Figure S3 (C) saved")


def figure_d_scatter_agreement(df, output_dir):
    """Figure S4: Scatter plots showing Expert 1 vs Expert 2 agreement."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for ax, (trait, abbrev) in zip(axes, zip(TRAITS, TRAIT_ABBREV)):
        exp1 = df[f'exp1_{abbrev}']
        exp2 = df[f'exp2_{abbrev}']

        ax.scatter(exp1, exp2, alpha=0.5, s=20, color=COLORS['expert_avg'], edgecolor='white', linewidth=0.3)

        lims = [0, 10]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect agreement')

        slope, intercept, r, p, se = stats.linregress(exp1, exp2)
        x_line = np.array(lims)
        ax.plot(x_line, intercept + slope * x_line, color=COLORS['ai'], linewidth=1.5,
               label=f'r = {r:.2f}')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Expert 1')
        ax.set_ylabel('Expert 2')
        ax.set_title(f'{trait}\n(r={r:.2f})')
        ax.set_aspect('equal')

    fig.suptitle('D. Inter-Rater Agreement: Expert 1 vs Expert 2', fontsize=12, y=1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_d_expert_agreement.png', dpi=300)
    plt.savefig(output_dir / 'figure_d_expert_agreement.pdf')
    plt.close()

    print(f"✓ Figure S4 (D) saved")


def main():
    """Generate all supplementary figures."""
    print("=" * 60)
    print("Supplementary Figures: Human Expert Baseline")
    print("For Frontiers in Psychology")
    print("=" * 60)

    df = load_data()
    print(f"\n✓ Loaded {len(df)} records")

    # Output to figures directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating figures to: {output_dir}")
    print("-" * 40)

    corr_df = figure_a_correlation_comparison(df, output_dir)
    icc_df = figure_b_icc_visualization(df, output_dir)
    figure_c_error_distribution(df, output_dir)
    figure_d_scatter_agreement(df, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nMean AI correlation: {corr_df['AI'].mean():.3f}")
    print(f"Mean Expert Avg correlation: {corr_df['Expert Avg'].mean():.3f}")
    print(f"AI improvement: +{(corr_df['AI'].mean() - corr_df['Expert Avg'].mean())*100:.1f}%")
    print(f"\nMean ICC: {icc_df['ICC'].mean():.3f}")

    print("\n✓ All supplementary figures generated!")


if __name__ == '__main__':
    main()
