# Human Expert Baseline Data

**Section**: 3.4 Human Expert Baseline Comparison (Table 4, Figure 2)
**Data Type**: Anonymized personality ratings from 250 children
**Ground Truth**: Comparative analysis (self-report, expert ratings, AI predictions)

## File Descriptions

### `human_expert_baseline_complete.csv`
Complete dataset with all ratings combined.

**Size**: 250 rows (one per child), 29 columns
**Primary Use**: Generating Figure 2 and Table 4

**Columns**:
```
photo_id             : Anonymized identifier (P001-P250)
age                  : Child age in years (6-18)
gender               : Male or Female

Expert 1 Ratings (Big Five):
exp1_O, exp1_C, exp1_E, exp1_A, exp1_N : 0-10 scale
exp1_confidence      : High/Medium/Low
exp1_date            : Rating date

Expert 2 Ratings (Big Five):
exp2_O, exp2_C, exp2_E, exp2_A, exp2_N : 0-10 scale
exp2_confidence      : High/Medium/Low
exp2_date            : Rating date

Self-Report (Big Five):
self_O, self_C, self_E, self_A, self_N : 0-10 scale

AI Predictions (Big Five):
ai_O, ai_C, ai_E, ai_A, ai_N          : 0-10 scale
ai_model_version    : Model version used
ai_date             : Prediction date
```

### `expert_ratings.csv`
Separate dataset containing only expert raters' assessments.

**Columns**:
- Expert 1 and Expert 2 personality trait scores
- Confidence levels for each rating
- Rating dates

**Purpose**: Inter-rater reliability analysis (ICC computation in generate_figure2.py)

### `ai_predictions.csv`
AI system's personality predictions for same children.

**Columns**:
- 5 Big Five trait predictions (0-10 scale)
- Model version identifier
- Prediction timestamp

### `ground_truth.csv`
Children's self-reported personality assessments.

**Columns**:
- 5 Big Five personality traits (0-10 scale)
- Self-report methodology information
- Assessment date

## Key Metrics

All values are 0-10 scales (normalized to 0-1 for analysis where needed).

**Big Five Traits**:
- O = Openness
- C = Conscientiousness
- E = Extraversion
- A = Agreeableness
- N = Neuroticism

## Analysis

### Main Analysis Script
Run `generate_figure2.py` to reproduce Figure 2 and Table 4:

```bash
python ../scripts/generate_figure2.py
```

**Output**:
- Figure 2A: Bar chart comparing AI vs Expert Avg correlations
- Figure 2B: ICC (inter-rater reliability) horizontal bar chart
- Table 4: Correlation with self-report by trait
- Verification: Expert Avg = (Expert1 + Expert2) / 2

### Key Results

**Prediction Accuracy (Correlation with Self-Report)**:
```
Trait              Expert1   Expert2   Expert Avg    AI
───────────────────────────────────────────────────────
Openness            0.387     0.262      0.325    0.351
Conscientiousness   0.295     0.287      0.291    0.297
Extraversion        0.301     0.313      0.307    0.381
Agreeableness       0.289     0.287      0.288    0.321
Neuroticism         0.299     0.279      0.289    0.337
───────────────────────────────────────────────────────
MEAN                0.314     0.286      0.300    0.337
```

**AI Advantage**: +6.0% mean correlation improvement (not statistically significant, p=0.46)

**Inter-Rater Reliability (ICC between Expert 1 & 2)**:
```
All traits exceed 0.70 threshold for "good" reliability
Range: 0.71-0.89 across Big Five dimensions
```

## Privacy & Ethics

- ✅ All identifiers are anonymized (P001-P250)
- ✅ No personal names, addresses, or contact information
- ✅ No photographs included
- ✅ GDPR/COPPA compliant
- ✅ Parental consent obtained
- ✅ Data deletion upon request available

## Citation

If using this dataset, please cite the associated manuscript:

```bibtex
@article{sergeev2026deep_research_engine,
  title={Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis},
  author={Sergeev, Dmitriy},
  journal={Frontiers in Psychology},
  year={2026},
  note={Submitted}
}
```

## Notes

- Two expert raters are licensed clinical psychologists
- Ratings performed independently without inter-rater consultation
- Self-report collected separately from expert assessments
- AI predictions generated using frozen pre-trained models (no fine-tuning on this cohort)
