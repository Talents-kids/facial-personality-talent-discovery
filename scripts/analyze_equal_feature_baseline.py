#!/usr/bin/env python3
"""
Issue #6: Equal-Feature Baseline Comparison

Compare 21 facial features vs 21 questionnaire features for Big Five personality prediction.
Uses N=428 users with self-reported Big Five scores.

Analysis:
1. Train model on 21 facial features only → AUC_facial
2. Train model on 21 questionnaire features only → AUC_quest
3. Train model on all 42 features combined → AUC_combined
4. Compare incremental value: Δ = AUC_combined - max(AUC_facial, AUC_quest)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'equal_feature_baseline', 'TEMPLATE_6_equal_feature_N428_FILLED.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

# Big Five traits
TRAITS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
TRAIT_LABELS = {'openness': 'O', 'conscientiousness': 'C', 'extraversion': 'E',
                'agreeableness': 'A', 'neuroticism': 'N'}


def load_and_prepare_data():
    """Load data and identify feature columns."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify facial and questionnaire columns
    facial_cols = [c for c in df.columns if c.startswith('facial_')]
    quest_cols = [c for c in df.columns if c.startswith('quest_')]

    print(f"  Facial features: {len(facial_cols)}")
    print(f"  Questionnaire features: {len(quest_cols)}")
    print(f"  Big Five traits: {TRAITS}")

    return df, facial_cols, quest_cols


def preprocess_features(df, feature_cols):
    """Preprocess features: handle missing values, encode categoricals, scale."""
    X = df[feature_cols].copy()

    # Handle categorical columns (encode as integers)
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values with median
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def run_cv_analysis(X, y, n_splits=10):
    """Run cross-validated AUC analysis."""
    # Remove samples with missing y
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(np.unique(y_clean)) < 2:
        return np.nan, np.nan

    # Binarize y by median
    y_binary = (y_clean > np.median(y_clean)).astype(int)

    # Check class balance
    if np.sum(y_binary) < 10 or np.sum(1 - y_binary) < 10:
        return np.nan, np.nan

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    try:
        scores = cross_val_score(model, X_clean, y_binary, cv=cv, scoring='roc_auc')
        return np.mean(scores), np.std(scores)
    except Exception as e:
        print(f"    Error: {e}")
        return np.nan, np.nan


def analyze_all_traits(df, facial_cols, quest_cols):
    """Run analysis for all Big Five traits."""
    results = []

    # Prepare feature matrices
    X_facial = preprocess_features(df, facial_cols)
    X_quest = preprocess_features(df, quest_cols)
    X_combined = np.hstack([X_facial, X_quest])

    print("\n" + "=" * 70)
    print("EQUAL-FEATURE BASELINE COMPARISON")
    print("=" * 70)
    print(f"\nFeatures: {len(facial_cols)} facial vs {len(quest_cols)} questionnaire")
    print(f"Samples: N={len(df)}")
    print(f"Method: 10-fold CV, Gradient Boosting, AUC-ROC")
    print("\n" + "-" * 70)

    for trait in TRAITS:
        print(f"\n{trait.upper()}:")
        y = df[trait].values

        # Facial only
        auc_facial, std_facial = run_cv_analysis(X_facial, y)
        print(f"  Facial only:      AUC = {auc_facial:.3f} ± {std_facial:.3f}")

        # Questionnaire only
        auc_quest, std_quest = run_cv_analysis(X_quest, y)
        print(f"  Questionnaire:    AUC = {auc_quest:.3f} ± {std_quest:.3f}")

        # Combined
        auc_combined, std_combined = run_cv_analysis(X_combined, y)
        print(f"  Combined (42):    AUC = {auc_combined:.3f} ± {std_combined:.3f}")

        # Incremental value
        if not np.isnan(auc_combined) and not np.isnan(auc_facial) and not np.isnan(auc_quest):
            baseline = max(auc_facial, auc_quest)
            delta = auc_combined - baseline
            print(f"  Δ (incremental):  {delta:+.3f} ({'+' if delta > 0 else ''}{delta*100:.1f}%)")
        else:
            delta = np.nan

        results.append({
            'trait': trait,
            'auc_facial': auc_facial,
            'std_facial': std_facial,
            'auc_quest': auc_quest,
            'std_quest': std_quest,
            'auc_combined': auc_combined,
            'std_combined': std_combined,
            'delta': delta
        })

    return pd.DataFrame(results)


def generate_figure(results_df):
    """Generate publication-quality comparison figure."""
    print("\nGenerating figure...")

    fig, ax = plt.subplots(figsize=(10, 6))

    traits = results_df['trait'].values
    x = np.arange(len(traits))
    width = 0.25

    # Bar positions
    facial_bars = ax.bar(x - width, results_df['auc_facial'], width,
                         label='Facial (21 features)', color='#2E86AB', alpha=0.85,
                         yerr=results_df['std_facial'], capsize=3)
    quest_bars = ax.bar(x, results_df['auc_quest'], width,
                        label='Questionnaire (21 features)', color='#A23B72', alpha=0.85,
                        yerr=results_df['std_quest'], capsize=3)
    combined_bars = ax.bar(x + width, results_df['auc_combined'], width,
                          label='Combined (42 features)', color='#F18F01', alpha=0.85,
                          yerr=results_df['std_combined'], capsize=3)

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance (0.5)')

    # Labels
    ax.set_xlabel('Big Five Personality Trait', fontsize=12)
    ax.set_ylabel('AUC-ROC (10-fold CV)', fontsize=12)
    ax.set_title('Equal-Feature Baseline Comparison:\nFacial vs Questionnaire Features for Personality Prediction',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in traits], fontsize=11)
    ax.set_ylim(0.4, 0.75)
    ax.legend(loc='upper right', fontsize=10)

    # Add delta annotations
    for i, row in results_df.iterrows():
        if not np.isnan(row['delta']):
            delta_text = f"Δ={row['delta']:+.2f}"
            ax.annotate(delta_text, (x[i] + width, row['auc_combined'] + row['std_combined'] + 0.02),
                       ha='center', fontsize=8, color='#F18F01')

    # Grid
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'png']:
        filepath = os.path.join(OUTPUT_DIR, f'equal_feature_comparison.{fmt}')
        plt.savefig(filepath, dpi=300, format=fmt)
        print(f"  Saved: {filepath}")

    plt.close()


def generate_latex_table(results_df):
    """Generate LaTeX table for manuscript."""
    print("\nGenerating LaTeX table...")

    latex = r"""\begin{table}[h]
\centering
\caption{Equal-Feature Baseline Comparison: Facial vs Questionnaire Features (N=428)}
\begin{tabular}{lcccc}
\toprule
\textbf{Trait} & \textbf{Facial (21)} & \textbf{Quest. (21)} & \textbf{Combined (42)} & \textbf{$\Delta$} \\
\midrule
"""

    for _, row in results_df.iterrows():
        trait = row['trait'].capitalize()
        facial = f"{row['auc_facial']:.2f} ± {row['std_facial']:.2f}"
        quest = f"{row['auc_quest']:.2f} ± {row['std_quest']:.2f}"
        combined = f"{row['auc_combined']:.2f} ± {row['std_combined']:.2f}"
        delta = f"{row['delta']:+.2f}" if not np.isnan(row['delta']) else "---"

        latex += f"{trait} & {facial} & {quest} & {combined} & {delta} \\\\\n"

    # Mean row
    mean_facial = results_df['auc_facial'].mean()
    mean_quest = results_df['auc_quest'].mean()
    mean_combined = results_df['auc_combined'].mean()
    mean_delta = results_df['delta'].mean()

    latex += r"""\midrule
\textbf{Mean} & """ + f"{mean_facial:.2f} & {mean_quest:.2f} & {mean_combined:.2f} & {mean_delta:+.2f}" + r""" \\
\bottomrule
\end{tabular}
\vspace{0.2cm}

\small \textit{Note:} 10-fold cross-validation with Gradient Boosting classifier. AUC values with standard deviation. $\Delta$ = Combined AUC minus max(Facial, Questionnaire) AUC.
\end{table}
"""

    # Save
    filepath = os.path.join(OUTPUT_DIR, 'equal_feature_table.tex')
    with open(filepath, 'w') as f:
        f.write(latex)
    print(f"  Saved: {filepath}")

    return latex


def generate_summary(results_df):
    """Generate summary statistics and interpretation."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_facial = results_df['auc_facial'].mean()
    mean_quest = results_df['auc_quest'].mean()
    mean_combined = results_df['auc_combined'].mean()
    mean_delta = results_df['delta'].mean()

    print(f"\nMean AUC across all traits:")
    print(f"  Facial features (21):      {mean_facial:.3f}")
    print(f"  Questionnaire features (21): {mean_quest:.3f}")
    print(f"  Combined features (42):    {mean_combined:.3f}")
    print(f"  Mean incremental value:    {mean_delta:+.3f}")

    # Which is better?
    if mean_facial > mean_quest:
        print(f"\n→ Facial features outperform questionnaire by {(mean_facial - mean_quest):.3f} AUC")
    else:
        print(f"\n→ Questionnaire features outperform facial by {(mean_quest - mean_facial):.3f} AUC")

    if mean_delta > 0.01:
        print(f"→ Combining both provides incremental value (+{mean_delta:.3f} AUC)")
    else:
        print(f"→ Minimal incremental value from combining ({mean_delta:+.3f} AUC)")

    # Manuscript text
    print("\n" + "-" * 70)
    print("RECOMMENDED TEXT FOR MANUSCRIPT:")
    print("-" * 70)

    text = f"""
\\textbf{{Equal-Feature Baseline Comparison:}} To assess the unique predictive
value of facial features versus traditional profile data, we compared models
trained on equal numbers of features: 21 facial geometric measurements versus
21 questionnaire/demographic variables (Table X).

Across Big Five traits (N=428), facial features achieved mean AUC = {mean_facial:.2f},
while questionnaire features achieved mean AUC = {mean_quest:.2f}. Combined models
(42 features) achieved mean AUC = {mean_combined:.2f}, representing a mean incremental
gain of $\\Delta$ = {mean_delta:+.2f} over the better single-modality baseline.

{"Facial features slightly outperform questionnaire data, suggesting unique signal" if mean_facial > mean_quest else "Questionnaire data slightly outperforms facial features, though both exceed chance"}
in personality-relevant information not captured by standard profile fields.
The modest incremental value ({mean_delta:+.2f}) from combining modalities indicates
{"complementary information across feature types" if mean_delta > 0.01 else "substantial overlap in predictive information"}.
"""
    print(text)

    # Save summary
    filepath = os.path.join(OUTPUT_DIR, 'analysis_summary.txt')
    with open(filepath, 'w') as f:
        f.write("EQUAL-FEATURE BASELINE COMPARISON - ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data: N={len(pd.read_csv(DATA_FILE))} users with Big Five self-report\n")
        f.write(f"Features: 21 facial + 21 questionnaire\n")
        f.write(f"Method: 10-fold CV, Gradient Boosting, AUC-ROC\n\n")
        f.write("RESULTS:\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\nMean Facial AUC: {mean_facial:.3f}\n")
        f.write(f"Mean Questionnaire AUC: {mean_quest:.3f}\n")
        f.write(f"Mean Combined AUC: {mean_combined:.3f}\n")
        f.write(f"Mean Incremental Value: {mean_delta:+.3f}\n")
        f.write("\nRECOMMENDED MANUSCRIPT TEXT:\n")
        f.write(text)
    print(f"\n  Saved: {filepath}")


def main():
    # Load data
    df, facial_cols, quest_cols = load_and_prepare_data()

    # Run analysis
    results_df = analyze_all_traits(df, facial_cols, quest_cols)

    # Save results CSV
    results_path = os.path.join(OUTPUT_DIR, 'equal_feature_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Generate outputs
    generate_figure(results_df)
    generate_latex_table(results_df)
    generate_summary(results_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
