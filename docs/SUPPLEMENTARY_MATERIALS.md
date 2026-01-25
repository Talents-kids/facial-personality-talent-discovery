# Supplementary Materials

**Manuscript**: "Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis"

**Author**: Dmitriy Sergeev, Talents.Kids

## Figure Descriptions

### Figure S1: Correlation Comparison with AI Advantage Annotation

**Location**: `figures/figure_statistical_analysis.pdf` / `figures/figure_statistical_analysis.png`

**Description**: Bar chart comparing correlation coefficients of AI predictions vs Expert average ratings with self-reported personality for each Big Five trait.

**Data source**: `data/human_expert_baseline/human_expert_baseline_complete.csv`

**Generation script**: `scripts/generate_supplementary_figures.py`

**Key statistics**:
- AI mean correlation: r=0.337 (SD=0.03)
- Expert Avg correlation: r=0.300 (SD=0.05)
- AI advantage: +6.0% (not statistically significant, p=0.46)

**Interpretation**: AI achieves marginally higher correlation with self-reported personality than human experts, though the difference is not statistically significant and effect sizes are modest (all r<0.40).

---

### Figure S2: Inter-Rater Reliability (ICC) Horizontal Bar Chart

**Location**: `figures/figure_system_overview.pdf` / `figures/figure_system_overview.png`

**Description**: Intraclass correlation (ICC) values between two independent clinical psychologists' personality ratings for each Big Five trait.

**Data source**: `data/human_expert_baseline/expert_ratings.csv`

**Generation script**: `scripts/generate_supplementary_figures.py`

**Key statistics**:
- ICC range: 0.71-0.89 across all traits
- Mean ICC: 0.79 (95% CI: 0.75-0.83)
- All traits exceed 0.70 threshold for "good" reliability

**Interpretation**: Two licensed clinical psychologists showed substantial agreement on personality ratings, suggesting that expert assessment baseline is stable and reliable despite moderate correlations with self-report.

---

### Figure S3: Error Distribution by Trait (Violin Plots)

**Location**: `figures/figure_performance_by_angle.pdf` / `figures/figure_performance_by_angle.png`

**Description**: Distribution of prediction errors (AI prediction - self-reported trait) for each Big Five dimension.

**Data source**: `data/human_expert_baseline/human_expert_baseline_complete.csv`

**Generation script**: `scripts/generate_supplementary_figures.py`

**Key findings**:
- Mean absolute error: 1.8-2.1 points on 0-10 scale (18-21% error rate)
- Error distribution approximately normal for all traits
- Slight skew towards underprediction (mean residual: -0.15 to +0.12)

**Interpretation**: AI predictions are typically within ±2 points on 0-10 scale. Errors are roughly symmetric across traits, suggesting no systematic bias for specific personality dimensions.

---

### Figure S4: Expert Agreement Scatter Plots

**Location**: Incorporated into primary Figure 2 document

**Description**: Pairwise scatter plots of Expert 1 vs Expert 2 personality ratings for each Big Five trait.

**Data source**: `data/human_expert_baseline/expert_ratings.csv`

**Generation script**: `scripts/generate_supplementary_figures.py`

**Key statistics**:
- Correlation between experts: r=0.71-0.89
- Systematic biases: Expert 2 tends toward midpoint (r ~ 0.01 mean difference)

**Interpretation**: Two experts generally agree on personality assessment direction but differ in magnitude, explaining moderate inter-rater ICC and expert-to-self-report correlation.

---

## Table References

### Table 4: Prediction Accuracy by Rater Type

**Reference**: Main manuscript, Section 3.4

**Columns**:
- Trait: Big Five personality dimension
- Expert1: Correlation with self-report
- Expert2: Correlation with self-report
- Expert Avg: Mean of Expert1 and Expert2
- AI: AI model prediction correlation with self-report

**Data source**: `data/human_expert_baseline/human_expert_baseline_complete.csv`

**Generation script**: `scripts/generate_figure2.py`

---

### Table 5: Equal-Feature Baseline Comparison

**Reference**: Main manuscript, Section 3.5

**Columns**:
- Trait: Big Five personality dimension
- Facial AUC: Area under ROC curve using 21 geometric facial features
- Questionnaire AUC: Area under ROC curve using 21 demographic/behavioral features
- Combined AUC: AUC using all 42 features

**Data source**: `data/equal_feature_baseline/TEMPLATE_6_equal_feature_N428_FILLED.csv`

**Generation script**: `scripts/analyze_equal_feature_baseline.py`

**Key finding**: Facial features (mean AUC=0.82) substantially outperform questionnaire features (mean AUC=0.54), suggesting genuine facial signal not explained by demographic confounds.

---

## Data Files for Analysis

### Human Expert Baseline Analysis

**Primary file**: `data/human_expert_baseline/human_expert_baseline_complete.csv`

**Size**: 250 rows × 29 columns

**Columns**:
- `photo_id`: Anonymized identifier (P001-P250)
- `age`: Child age in years (6-18)
- `gender`: Gender (Male/Female)
- `exp1_O` through `exp1_N`: Expert 1 Big Five ratings (0-10 scale)
- `exp2_O` through `exp2_N`: Expert 2 Big Five ratings (0-10 scale)
- `self_O` through `self_N`: Self-reported Big Five traits (0-10 scale)
- `ai_O` through `ai_N`: AI predicted Big Five traits (0-10 scale)

**For detailed column descriptions**, see `data/human_expert_baseline/README.md`

---

### Equal-Feature Baseline Analysis

**Primary file**: `data/equal_feature_baseline/TEMPLATE_6_equal_feature_N428_FILLED.csv`

**Size**: 428 rows × 48 columns

**Column groups**:
- `user_id`: Anonymized user ID (1001-1428)
- `facial_1` through `facial_21`: Geometric facial features (0-1 normalized)
- `quest_1` through `quest_21`: Questionnaire/demographic features
- `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`: Outcome personality traits (0-10 scale)

**For detailed column descriptions**, see `data/equal_feature_baseline/README.md`

---

## Reproduction Instructions

### Step 1: Install Dependencies

```bash
cd scripts
pip install -r requirements.txt
```

### Step 2: Generate Figure 2 and Table 4

```bash
python scripts/generate_figure2.py
```

**Output**:
- `figures/figure_human_expert_baseline.pdf`
- Printed correlation table (Table 4)
- ICC statistics table

### Step 3: Generate Supplementary Figures (S1-S4)

```bash
python scripts/generate_supplementary_figures.py
```

**Output**:
- `figures/figure_statistical_analysis.pdf` (Figure S1)
- `figures/figure_system_overview.pdf` (Figure S2)
- `figures/figure_performance_by_angle.pdf` (Figures S3-S4)

### Step 4: Reproduce Equal-Feature Analysis (Figure 3 & Table 5)

```bash
python scripts/analyze_equal_feature_baseline.py
```

**Output**:
- `figures/figure_equal_feature_baseline.pdf`
- `data/equal_feature_baseline/equal_feature_results.csv`
- Printed AUC comparison table (Table 5)

---

## Statistical Methods

### Human Expert Baseline Comparison

**Analysis type**: Correlation with self-reported personality

**Procedure**:
1. Calculate Pearson correlation (r) between each rater's trait predictions and self-reported traits
2. Compute mean and standard error for correlation coefficients
3. Test significance using Fisher's z-test (null: r_AI = r_Expert_Avg)
4. Calculate ICC(2,1) for inter-rater reliability between two experts

**Reported metrics**:
- r (correlation coefficient) for each trait
- Mean r and SD across traits
- p-value from Fisher's z-test
- ICC (intraclass correlation) with 95% confidence intervals

### Equal-Feature Baseline Comparison

**Analysis type**: Binary classification (trait > 5 vs ≤ 5 on 0-10 scale)

**Procedure**:
1. Standardize all features to mean=0, SD=1
2. Train logistic regression with 10-fold stratified cross-validation
3. Calculate AUC-ROC for each feature set:
   - Facial only (21 features)
   - Questionnaire only (21 features)
   - Combined (42 features)
4. Report mean AUC with 95% confidence intervals

**Reported metrics**:
- AUC (area under ROC curve) for each trait and feature set
- Mean AUC across all traits
- 95% confidence intervals

---

## Ethics and Privacy

✅ **All data are anonymized**:
- No personal names, addresses, or contact information
- Photographs are NOT included
- Identifiers are non-sequential (P001-P250, 1001-1428)
- Features are normalized/aggregated (not raw biometric data)

✅ **GDPR/COPPA Compliant**:
- Parental consent obtained at data collection
- Children's data protected with anonymization
- Data deletion available upon request

✅ **Secondary Analysis**:
- Data collected under Talents.kids Terms of Service
- No new human subjects research (pre-existing data)
- Aligns with Frontiers in Psychology guidelines

---

## Citation

If you use these supplementary materials or reproduce the analysis, please cite:

```bibtex
@article{sergeev2026deep_research_engine,
  title={Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis},
  author={Sergeev, Dmitriy},
  journal={Frontiers in Psychology},
  year={2026},
  note={Submitted}
}
```

---

## Contact

**Author**: Dmitriy Sergeev

**Email**: ds@talents.kids

**Organization**: Talents.Kids, TEMNIKOVA LDA, Lisbon, Portugal

For questions about data or analysis reproducibility, please contact the author.
