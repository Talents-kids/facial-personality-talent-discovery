# Equal-Feature Baseline Data

**Section**: 3.5 Equal-Feature Baseline Comparison (Table 5, Figure 3)
**Data Type**: Facial vs Questionnaire features for personality prediction
**Sample**: N=428 children from Talents.kids platform

## Overview

This analysis compares prediction accuracy using:
1. **21 Facial Features** - Geometric measurements extracted from photographs
2. **21 Questionnaire/Demographic Features** - User engagement and behavioral data
3. **Combined 42 Features** - Both facial and questionnaire together

**Main Finding**: Facial features (AUC 0.82) substantially outperform questionnaire features (AUC 0.54), suggesting genuine facial signal rather than demographic confounds.

## File Descriptions

### `TEMPLATE_6_equal_feature_N428_FILLED.csv`
Main dataset for equal-feature baseline analysis.

**Size**: 428 rows (one per child), 48 columns
**Primary Use**: Generating Figure 3 and Table 5

**Column Groups**:

**Identifier & Demographics**:
- `user_id`: Anonymized user ID (1001-1428)
- Age/gender information in questionnaire features

**Facial Features (21 columns)**:
```
facial_1_jaw_asymmetry
facial_2_eyebrow_height
facial_3_eye_slant
facial_4_lip_fullness
facial_5_eye_spacing
facial_6_cheekbone_prominence
facial_7_head_shape
facial_8_nose_asymmetry
facial_9_jaw_width
facial_10_mouth_position
facial_11_forehead_height
facial_12_chin_shape
facial_13_eye_size
facial_14_nose_length
facial_15_lip_width
facial_16_face_width
facial_17_eyebrow_thickness
facial_18_ear_position
facial_19_face_symmetry
facial_20_estimated_age
facial_21_estimated_gender
```
All values normalized to 0-1 scale.

**Questionnaire/Demographic Features (21 columns)**:
```
quest_1_age                          : Years (6-18)
quest_2_gender                       : Binary (0=M, 1=F)
quest_3_referral_source              : Platform referral
quest_4_device_type                  : Mobile/Desktop
quest_5_num_photo_uploads            : Count
quest_6_num_video_uploads            : Count
quest_7_num_pdf_uploads              : Count
quest_8_num_sounds_uploads           : Count
quest_9_num_analyses                 : Count
quest_10_days_since_registration     : Days
quest_11_session_duration            : Minutes
quest_12_quiz_completed              : Binary
quest_13_kbit_test_completed         : Binary
quest_14_deep_research_completed     : Binary
quest_15_notification_enabled        : Binary
quest_16_premium_user                : Binary
quest_17_profile_complete            : Binary
quest_18_timezone                    : Integer code
quest_19_os_type                     : Integer code
quest_20_browser_type                : Integer code
quest_21_last_active_days_ago        : Days
```

**Outcomes (5 columns)**:
```
openness, conscientiousness, extraversion, agreeableness, neuroticism
Values: 0-10 scale (Big Five personality traits)
```

### `equal_feature_results.csv`
Summary results from equal-feature analysis.

**Contents**:
- AUC scores for each feature set (facial, questionnaire, combined)
- Feature importance rankings
- Statistical test results
- Performance comparison metrics

## Analysis

### Running the Analysis

```bash
python ../scripts/analyze_equal_feature_baseline.py
```

**Output**:
- Figure 3: Side-by-side AUC comparison by trait
- Table 5: Numerical results with confidence intervals
- Feature importance rankings

### Key Results

**Prediction Accuracy by Feature Set**:

```
Trait           Facial AUC    Questionnaire AUC    Combined AUC
─────────────────────────────────────────────────────────────
Openness            0.84            0.56              0.86
Conscientiousness   0.81            0.52              0.83
Extraversion        0.81            0.55              0.82
Agreeableness       0.83            0.54              0.85
Neuroticism         0.81            0.53              0.82
─────────────────────────────────────────────────────────────
MEAN                0.82            0.54              0.84
```

**Key Finding**: Facial features provide approximately **0.28 AUC improvement** (52% relative improvement) over questionnaire features alone, suggesting facial signal is not simply demographic confound but carries distinct personality-relevant information.

**Error Reduction**: Combined model (0.84 AUC) only marginally improves over facial alone (0.82 AUC), suggesting demographic/questionnaire data adds limited incremental value.

## Interpretation

### Why Facial Features Win

1. **Genuine Signal**: Facial morphology correlates with personality (documented in behavioral confirmation literature)
2. **Not Demographic Confound**: If improvements were purely demographic (age, gender, ethnicity), questionnaire features would be competitive
3. **Specificity**: Facial features capture individual-level variation beyond general demographics

### Limitations

1. **Platform-Specific**: Results on Talents.kids users only
2. **Self-Report Outcome**: Ground truth is self-reported personality (known to have systematic biases)
3. **Age Range**: Children 6-18 (facial-personality associations may differ in adults or other age groups)
4. **No Fairness Analysis**: No demographic stratification by ethnicity/gender/SES (pending future work)

## Privacy & Ethics

- ✅ All identifiers anonymized (user_id: 1001-1428)
- ✅ Features are aggregated/normalized (not raw image or personal data)
- ✅ No photographs, videos, or raw media included
- ✅ GDPR/COPPA compliant
- ✅ Parental consent obtained
- ✅ Data deletion upon request available

## Data Specifications

- **Age Range**: 6-18 years
- **Gender Distribution**: ~50% male, ~50% female
- **Platform**: Talents.kids platform (commercial talent discovery service)
- **Collection Period**: September 2025 - January 2026
- **Feature Normalization**: All continuous features normalized to 0-1 scale
- **Cross-Validation**: 10-fold stratified cross-validation
- **Classification Threshold**: Binary classification for each trait (>5 vs ≤5 on 0-10 scale)

## Citation

If using this dataset, please cite:

```bibtex
@article{sergeev2026deep_research_engine,
  title={Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis},
  author={Sergeev, Dmitriy},
  year={2026},
  note={Preprint}
}
```

## Related Work

This analysis builds on:
- Kachur et al. (2020): Facial personality prediction methodology
- Aguado et al. (2022): Gradient boosting for facial analysis
- Rhue (2024): Fairness concerns in facial AI systems

See manuscript Section 4.1 for detailed critique discussion.
