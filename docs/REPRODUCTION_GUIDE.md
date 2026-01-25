# Reproduction Guide

Complete step-by-step guide to reproduce all results from the Frontiers in Psychology manuscript.

**Manuscript**: "Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis"

---

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for data + scripts
- **OS**: macOS, Linux, or Windows (with WSL)

---

## Quick Start (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/talents-kids/frontiers-facial-personality-analysis.git
cd frontiers-facial-personality-analysis
```

### 2. Install Dependencies

```bash
pip install -r scripts/requirements.txt
```

### 3. Run All Analyses

```bash
# Generate Figure 2 (AI vs Human Experts)
python scripts/generate_figure2.py

# Generate Supplementary Figures S1-S4
python scripts/generate_supplementary_figures.py

# Generate Figure 3 (Equal-Feature Baseline)
python scripts/analyze_equal_feature_baseline.py
```

**Expected output**: PDF and PNG figures in `figures/` directory, plus printed tables in terminal.

---

## Detailed Reproduction Steps

### Phase 1: Environment Setup (10 minutes)

#### Step 1.1: Create Virtual Environment

```bash
# Navigate to repository
cd frontiers-facial-personality-analysis

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

#### Step 1.2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r scripts/requirements.txt
```

**Verify installation**:
```bash
python -c "import pandas, numpy, matplotlib, scipy; print('✅ All packages installed')"
```

---

### Phase 2: Reproduce Human Expert Baseline (Figure 2, Table 4)

**Objective**: Compare AI predictions vs clinical psychologist ratings

**Data**: `data/human_expert_baseline/human_expert_baseline_complete.csv` (N=250)

#### Step 2.1: Verify Data File

```bash
# Check file exists and has expected columns
python -c "
import pandas as pd
df = pd.read_csv('data/human_expert_baseline/human_expert_baseline_complete.csv')
print(f'Loaded {len(df)} records')
print(f'Columns: {list(df.columns)}')
assert len(df) == 250, 'Expected 250 rows'
assert 'exp1_O' in df.columns, 'Missing Expert 1 ratings'
assert 'ai_O' in df.columns, 'Missing AI predictions'
print('✅ Data structure verified')
"
```

#### Step 2.2: Run Figure 2 Generation

```bash
python scripts/generate_figure2.py
```

**Expected output**:
```
✅ Loaded 250 expert baseline records

PREDICTION ACCURACY BY RATER TYPE
─────────────────────────────────────────────────
Trait              Expert1   Expert2   Expert_Avg      AI
─────────────────────────────────────────────────
Openness           0.387     0.262       0.325     0.351
Conscientiousness  0.295     0.287       0.291     0.297
Extraversion       0.301     0.313       0.307     0.381
Agreeableness      0.289     0.287       0.288     0.321
Neuroticism        0.299     0.279       0.289     0.337
─────────────────────────────────────────────────
MEAN               0.314     0.286       0.300     0.337

AI ADVANTAGE: +6.0% (not statistically significant, p=0.46)

INTER-RATER RELIABILITY (ICC between Expert 1 & 2)
─────────────────────────────────────────────────
Trait              ICC        95% CI
─────────────────────────────────────────────────
Openness           0.89       [0.83-0.93]
Conscientiousness  0.71       [0.59-0.80]
Extraversion       0.82       [0.71-0.89]
Agreeableness      0.74       [0.62-0.83]
Neuroticism        0.77       [0.66-0.85]
─────────────────────────────────────────────────

Figure saved: figures/figure_human_expert_baseline.pdf
```

#### Step 2.3: Verify Output

```bash
# Check figure was created
ls -lh figures/figure_human_expert_baseline.*

# Open figure (macOS)
open figures/figure_human_expert_baseline.pdf

# Open figure (Linux)
evince figures/figure_human_expert_baseline.pdf

# Open figure (Windows)
start figures/figure_human_expert_baseline.pdf
```

---

### Phase 3: Reproduce Supplementary Figures S1-S4

**Objective**: Generate supplementary analysis figures

#### Step 3.1: Run Supplementary Figure Generation

```bash
python scripts/generate_supplementary_figures.py
```

**Expected output**:
```
✅ Loaded 250 expert baseline records

Generating supplementary figures...

FIGURE S1: Correlation comparison with AI advantage annotation
FIGURE S2: ICC (inter-rater reliability) horizontal bar chart
FIGURE S3: Error distribution by trait (violin plots)
FIGURE S4: Expert agreement scatter plots (matrix)

All supplementary figures saved to figures/
```

#### Step 3.2: View Generated Figures

```bash
# List all supplementary figures
ls -lh figures/figure_statistical_analysis.*
ls -lh figures/figure_system_overview.*
ls -lh figures/figure_performance_by_angle.*
```

---

### Phase 4: Reproduce Equal-Feature Baseline (Figure 3, Table 5)

**Objective**: Compare facial vs questionnaire features for personality prediction

**Data**: `data/equal_feature_baseline/TEMPLATE_6_equal_feature_N428_FILLED.csv` (N=428)

#### Step 4.1: Verify Data File

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/equal_feature_baseline/TEMPLATE_6_equal_feature_N428_FILLED.csv')
print(f'Loaded {len(df)} records')
print(f'Total columns: {len(df.columns)}')
assert len(df) == 428, 'Expected 428 rows'
assert 'facial_1' in df.columns, 'Missing facial features'
assert 'quest_1' in df.columns, 'Missing questionnaire features'
assert 'openness' in df.columns, 'Missing personality outcomes'
print('✅ Data structure verified')
"
```

#### Step 4.2: Run Equal-Feature Analysis

```bash
python scripts/analyze_equal_feature_baseline.py
```

**Expected output**:
```
✅ Loaded 428 equal-feature baseline records

PREDICTION ACCURACY BY FEATURE SET
──────────────────────────────────────────────────────
Trait              Facial AUC    Questionnaire AUC    Combined AUC
──────────────────────────────────────────────────────
Openness               0.84            0.56              0.86
Conscientiousness      0.81            0.52              0.83
Extraversion           0.81            0.55              0.82
Agreeableness          0.83            0.54              0.85
Neuroticism            0.81            0.53              0.82
──────────────────────────────────────────────────────
MEAN                   0.82            0.54              0.84

KEY FINDING: Facial features provide +0.28 AUC improvement (52% relative gain)
over questionnaire features, indicating genuine facial signal.

FEATURE IMPORTANCE RANKING (Top 10):
1. facial_6_cheekbone_prominence    (importance: 0.089)
2. facial_19_face_symmetry          (importance: 0.078)
3. facial_12_chin_shape             (importance: 0.072)
4. facial_5_eye_spacing             (importance: 0.068)
5. quest_1_age                      (importance: 0.054)
6. facial_3_eye_slant               (importance: 0.051)
7. facial_7_head_shape              (importance: 0.048)
8. quest_10_days_since_registration (importance: 0.045)
9. facial_2_eyebrow_height          (importance: 0.043)
10. quest_12_quiz_completed         (importance: 0.041)

Figure saved: figures/figure_equal_feature_baseline.pdf
Results saved: data/equal_feature_baseline/equal_feature_results.csv
```

#### Step 4.3: Verify Output

```bash
# Check figure
ls -lh figures/figure_equal_feature_baseline.*

# Check results CSV
head -20 data/equal_feature_baseline/equal_feature_results.csv
```

---

## Troubleshooting

### Issue: Import Error (Module Not Found)

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall -r scripts/requirements.txt
```

### Issue: Data File Not Found

```
FileNotFoundError: data/human_expert_baseline/human_expert_baseline_complete.csv
```

**Solution**:
```bash
# Verify you're in correct directory
pwd  # Should end with: frontiers-facial-personality-analysis

# Check files exist
ls data/human_expert_baseline/
ls data/equal_feature_baseline/
```

### Issue: Permission Denied on Linux/macOS

```
PermissionError: [Errno 13] Permission denied: 'figures/'
```

**Solution**:
```bash
# Fix permissions
chmod -R u+w figures/
chmod -R u+w data/
```

### Issue: Out of Memory

```
MemoryError: Unable to allocate X GB for array
```

**Solution**: The analysis should use <2GB RAM. If you encounter this:
```bash
# Check available RAM
free -h  # Linux
vm_stat  # macOS

# Try closing other applications
# Or increase swap space (advanced)
```

---

## Verifying Reproducibility

### Checksum Verification

After running all scripts, verify outputs match expected values:

```bash
# Figure 2 Table 4 - Expected mean correlations
python -c "
import pandas as pd
df = pd.read_csv('data/human_expert_baseline/human_expert_baseline_complete.csv')

# Calculate AI correlations
traits = ['O', 'C', 'E', 'A', 'N']
correlations = []
for trait in traits:
    r = df[f'ai_{trait}'].corr(df[f'self_{trait}'])
    correlations.append(r)

mean_r = sum(correlations) / len(correlations)
print(f'Mean AI correlation: {mean_r:.3f}')
assert 0.33 < mean_r < 0.35, 'Mean correlation outside expected range'
print('✅ Table 4 values verified')
"

# Figure 3 Table 5 - Expected AUC values
python -c "
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/equal_feature_baseline/TEMPLATE_6_equal_feature_N428_FILLED.csv')

# Check basic statistics
print(f'Dataset shape: {df.shape}')
print(f'Facial features: facial_1 to facial_21')
print(f'Questionnaire features: quest_1 to quest_21')
print(f'Outcomes: openness, conscientiousness, extraversion, agreeableness, neuroticism')
print('✅ Table 5 data structure verified')
"
```

---

## Advanced Usage

### Running Individual Analyses

**Generate only Figure 2 (skip supplementary)**:
```python
python -c "
import sys
sys.path.insert(0, 'scripts')
from generate_figure2 import main
main()
"
```

**Generate only Table 5 (skip Table 4)**:
```python
python -c "
import sys
sys.path.insert(0, 'scripts')
from analyze_equal_feature_baseline import main
main()
"
```

### Modifying Analysis Parameters

**To change cross-validation folds in equal-feature analysis**:
```bash
# Edit scripts/analyze_equal_feature_baseline.py
# Line ~45: cv=5 → cv=10 (for 10-fold cross-validation)
# Then re-run
python scripts/analyze_equal_feature_baseline.py
```

**To change personality classification threshold (binary cutoff)**:
```bash
# Default: threshold=5 (personality > 5 vs ≤ 5 on 0-10 scale)
# Edit scripts/analyze_equal_feature_baseline.py
# Line ~40: threshold = 5 → threshold = 6
# Then re-run
python scripts/analyze_equal_feature_baseline.py
```

---

## Validation Checklist

After completing reproduction, verify:

- [ ] Figure 2 PDF generated (Figure_human_expert_baseline.pdf exists)
- [ ] Figure 2 mean AI correlation ≈ 0.337 (±0.01)
- [ ] Expert ICC values all > 0.70
- [ ] Supplementary Figures S1-S4 generated (4 PDF files in figures/)
- [ ] Figure 3 PDF generated (figure_equal_feature_baseline.pdf exists)
- [ ] Table 5 mean Facial AUC ≈ 0.82 (±0.02)
- [ ] Table 5 mean Questionnaire AUC ≈ 0.54 (±0.03)
- [ ] AUC improvement (facial - questionnaire) ≈ 0.28
- [ ] equal_feature_results.csv created with 5 rows (one per trait)
- [ ] All CSV files readable and contain expected columns
- [ ] No errors or warnings in terminal output

---

## Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| Environment setup | 5 min | One-time only |
| Figure 2 generation | 30 sec | Human expert correlation analysis |
| Supplementary figures | 2 min | ICC, error distributions, scatter plots |
| Figure 3 generation | 5 min | Equal-feature AUC computation (10-fold CV) |
| **Total** | **≈8 minutes** | First run; cached plots load faster |

---

## Data Dictionary

### human_expert_baseline_complete.csv

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `photo_id` | str | P001-P250 | Anonymized photo identifier |
| `age` | int | 6-18 | Child age in years |
| `gender` | str | M/F | Gender |
| `exp1_O`, `exp1_C`, etc. | float | 0-10 | Expert 1 Big Five traits |
| `exp2_O`, `exp2_C`, etc. | float | 0-10 | Expert 2 Big Five traits |
| `self_O`, `self_C`, etc. | float | 0-10 | Self-reported Big Five traits |
| `ai_O`, `ai_C`, etc. | float | 0-10 | AI predicted Big Five traits |

### TEMPLATE_6_equal_feature_N428_FILLED.csv

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `user_id` | int | 1001-1428 | Anonymized user ID |
| `facial_1` to `facial_21` | float | 0-1 | Normalized geometric facial features |
| `quest_1` to `quest_21` | float | varies | Questionnaire/demographic features |
| `openness`, `conscientiousness`, etc. | int | 0-10 | Outcome personality traits |

---

## Citation

If you reproduce or build upon this analysis, please cite:

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

## Support

For issues or questions:

1. Check the **Troubleshooting** section above
2. Review `data/*/README.md` for data-specific questions
3. Check `docs/SUPPLEMENTARY_MATERIALS.md` for methodology questions
4. Contact: ds@talents.kids

---

*Last updated: January 16, 2026*
