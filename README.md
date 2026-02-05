# Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18500157.svg)](https://doi.org/10.5281/zenodo.18500157)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)

This repository contains the complete research materials for the manuscript prepared for submission to **Frontiers in Psychology**:

**"Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis"**

Dmitriy Sergeev
Talents.Kids, TEMNIKOVA LDA, Lisbon, Portugal

## 📄 Paper Summary

This study presents a two-stage AI system combining facial personality prediction with multi-agent large language model (LLM) ensemble for scalable talent discovery in children and adolescents. The system integrates facial analysis with natural language processing to provide comprehensive personality assessment and talent categorization.

**Key Results:**
- **Internal Validation**: AUC 0.81 [95% CI: 0.78-0.84] on facial personality prediction (N=18,337 adults)
- **Human Expert Comparison**: AI achieved +6.0% higher correlation with self-reported personality than clinical psychologists (r=0.351 vs. r=0.291), though not statistically significant (p=0.46)
- **Commercial Deployment**: AUC 0.94 [95% CI: 0.91-0.96] for credit risk prediction in banking deployment
- **Multi-modal Integration**: Multi-LLM architecture validated across 5-25 parallel agents from 9 providers (OpenAI, Anthropic, Gemini, XAI)
- **Cost Efficiency**: $0.041 per analysis with 7.5-second latency

**Important Limitations:**
- Cross-sectional design precludes causal inference
- Platform-specific validation (requires external replication)
- Out-of-distribution generalization (adult models → child population)
- No demographic stratification audit (fairness analysis pending)

## 📖 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{sergeev2026deep_research_engine,
  title={Deep Research Engine: Multi-LLM Talent Discovery from Facial Personality Analysis},
  author={Sergeev, Dmitriy},
  year={2026},
  note={Preprint}
}
```

## 🔗 Related Works & Previous Research

This manuscript builds upon our previous research on AI-driven talent discovery and psychological assessment:

### 1. TALENT LLM: Fine-Tuned Large Language Models for Talent Prediction

**doi**: [10.5281/zenodo.17743456](https://doi.org/10.5281/zenodo.17743456)

**GitHub**: [github.com/talents-kids/talent-llm](https://github.com/talents-kids/talent-llm)

**Description**: Initial work on fine-tuning large language models (Qwen, Llama, Mistral) for talent prediction tasks. Established baseline methodologies and model architecture patterns.

---

### 2. Deep Research Engine: Multi-LLM Talent Discovery (TiCS Submission)

**doi**: [10.5281/zenodo.17849535](https://doi.org/10.5281/zenodo.17849535)

**Journal**: Trends in Cognitive Sciences (submitted)

**Description**: Extended research introducing the multi-agent LLM architecture with ensemble averaging across 5-25 parallel agents from 9 different providers (OpenAI, Anthropic, Google Gemini, xAI, etc.). Demonstrates cost-effectiveness and improved consensus quality.

---

### 3. Multimodal Talent Discovery in Children Using Calibrated Baselines

**EdArXiv Preprint**: [osf.io/preprints/edarxiv/3jrm4_v1](https://osf.io/preprints/edarxiv/3jrm4_v1)

**SSRN**: [ssrn.com/abstract=5933954](https://ssrn.com/abstract/5933954)

**doi**: [10.5281/zenodo.17941256](https://doi.org/10.5281/zenodo.17941256)

**GitHub**: [github.com/talents-kids/calibrated-talent-assessment](https://github.com/talents-kids/calibrated-talent-assessment)

**Journal**: iScience (published)

**Description**: Comprehensive study of multimodal artifact analysis (drawings, essays, music, video) with rigorous baseline comparisons and calibration methods. Demonstrates equal-feature baseline methodology used in the current manuscript. Published on SSRN and EdArXiv.

---

### 4. Current Work: Deep Research Engine - Facial Personality Analysis

**Journal**: Frontiers in Psychology (submitted)

**Focus**:
- Facial geometric features for personality prediction
- Human expert baseline comparison (N=250)
- Equal-feature baseline validation (N=428)
- Multi-LLM talent discovery integration

---

## 📊 Dataset

### Overview
- **N=250** children with human expert baseline comparison (Section 3.4)
- **N=428** children in equal-feature baseline analysis (Section 3.5)
- **N=18,337** adults in internal validation (Section 3.1)
- **N>5,000** in external commercial deployment (banking sector)

### Data Files

```
data/
├── human_expert_baseline/
│   ├── human_expert_baseline_complete.csv      # N=250, all ratings (Table 4)
│   ├── expert_ratings.csv                      # Two clinical psychologists
│   ├── ai_predictions.csv                      # AI personality predictions
│   ├── ground_truth.csv                        # Self-reported personality
│   └── README.md                               # Data documentation
│
└── equal_feature_baseline/
    ├── TEMPLATE_6_equal_feature_N428_FILLED.csv # N=428, facial vs questionnaire (Table 5)
    ├── equal_feature_results.csv                # AUC metrics by feature type
    └── README.md                                # Analysis documentation
```

**Privacy Notice**: All data are anonymized. No personal identifiers, addresses, or contact information included. Children's facial photographs are NOT included (GDPR/COPPA compliance).

### Data Format

**Human Expert Baseline** (`human_expert_baseline_complete.csv`):

| Column | Description |
|--------|-------------|
| `photo_id` | Anonymized photo identifier (P001, P002, ...) |
| `age` | Child's age in years (6-18) |
| `gender` | Gender (male/female) |
| `exp1_O`, `exp1_C`, ..., `exp1_N` | Expert 1 ratings for Big Five traits (0-10) |
| `exp2_O`, `exp2_C`, ..., `exp2_N` | Expert 2 ratings for Big Five traits (0-10) |
| `self_O`, `self_C`, ..., `self_N` | Self-reported traits (0-10) |
| `ai_O`, `ai_C`, ..., `ai_N` | AI predicted traits (0-10) |

**Equal-Feature Baseline** (`TEMPLATE_6_equal_feature_N428_FILLED.csv`):

| Column Type | Description |
|-------------|-------------|
| `user_id` | Anonymized user ID |
| `facial_*` | 21 geometric facial features (0-1 normalized) |
| `quest_*` | 21 questionnaire/demographic features |
| `openness`, `conscientiousness`, etc. | Outcome personality traits (0-10) |

## 🔬 Methods

### Stage 1: Facial Personality Analysis

**Feature Extraction:**
- YOLOv5 face detection (99.97% accuracy)
- 68-point facial landmark detection (normalized mean error: 5.5)
- 19 geometric facial features extracted per image
- StyleGAN2 frontalization for off-angle images (93% accuracy)

**Model Ensemble:**
- CatBoost (40% weight)
- XGBoost (35% weight)
- LightGBM (25% weight)
- Predicts 137 personality traits on 0-10 scales

### Stage 2: Multi-LLM Talent Analyzer

- 5-25 parallel LLM agents from diverse providers
- Artifact-based analysis (drawings, text, music, video)
- Weighted consensus aggregation
- 306 fine-grained talent categories

## 📈 Reproducibility

### 🚀 Quick Start: Interactive Jupyter Notebook

The easiest way to reproduce all results is using the interactive Jupyter notebook:

**File**: `notebooks/frontiers_complete_analysis.ipynb`

**Features**:
- ✅ Google Colab compatible (click the badge to open)
- ✅ Generates all figures (Figures 2-3, S1-S4)
- ✅ Reproduces all statistical analyses
- ✅ Prints summary tables (Table 4 and 5)
- ✅ Downloads results and figures

**How to use**:
1. Click the "Open in Colab" button in the notebook
2. Upload your CSV files (or use Files panel)
3. Run all cells (Ctrl+A then Shift+Enter)
4. Download generated figures

### Alternative: Python Scripts

If you prefer command-line execution, three scripts reproduce all results:

```
scripts/
├── generate_figure2.py                  # → Figure 2 + Table 4 (Human Expert Baseline)
├── generate_supplementary_figures.py    # → Figures S1-S4 + ICC analysis
├── analyze_equal_feature_baseline.py    # → Figure 3 + Table 5 (Facial vs Questionnaire)
└── requirements.txt                     # Python dependencies
```

**Installation and execution**:

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Run ALL three scripts to reproduce complete analysis:
python scripts/generate_figure2.py
python scripts/generate_supplementary_figures.py
python scripts/analyze_equal_feature_baseline.py
```

**What each script produces**:
1. `generate_figure2.py`: Figure 2 (AI vs Human Experts), Table 4 (prediction accuracy)
2. `generate_supplementary_figures.py`: Figures S1-S4 (correlation, ICC, error distribution, expert agreement)
3. `analyze_equal_feature_baseline.py`: Figure 3 (facial vs questionnaire), Table 5 (AUC by feature set)

**Output**: All scripts read from `data/` and generate publication-quality figures in `figures/` (complete reproduction in ~30 seconds)

## 📁 Repository Structure

```
facial-personality-talent-discovery/
├── README.md                                  # This file
├── CITATION.cff                               # Machine-readable citation (GitHub)
├── LICENSE                                    # CC-BY-4.0 license
├── .gitignore                                 # Git ignore patterns
├── Deep_Research_Engine_Multi_LLM_Talent_Discovery.pdf  # Published manuscript (35 pages)
├── requirements.txt                           # Python dependencies (if at root)
│
├── data/                                      # Anonymized datasets
│   ├── human_expert_baseline/                 # Section 3.4 data (N=250)
│   │   ├── human_expert_baseline_complete.csv
│   │   ├── expert_ratings.csv
│   │   ├── ai_predictions.csv
│   │   ├── ground_truth.csv
│   │   └── README.md
│   │
│   └── equal_feature_baseline/                # Section 3.5 data (N=428)
│       ├── TEMPLATE_6_equal_feature_N428_FILLED.csv
│       ├── equal_feature_results.csv
│       └── README.md
│
├── scripts/                                   # Reproducibility code
│   ├── generate_figure2.py                    # Figure 2 generation
│   ├── generate_supplementary_figures.py      # Figures S1-S4
│   ├── analyze_equal_feature_baseline.py      # Table 5 analysis
│   └── requirements.txt
│
├── figures/                                   # Publication figures (pre-generated)
│   ├── figure_human_expert_baseline.png/pdf   # Figure 2
│   ├── figure_equal_feature_baseline.png/pdf  # Figure 3
│   ├── figure_statistical_analysis.png/pdf    # Figure S1
│   ├── figure_system_overview.png/pdf         # Figure S2
│   └── figure_performance_by_angle.png/pdf    # Figures S3-S4
│
├── docs/                                      # Additional documentation
│   ├── SUPPLEMENTARY_MATERIALS.md
│   └── REPRODUCTION_GUIDE.md
│
└── notebooks/                                 # Interactive Jupyter notebooks ⭐
    └── frontiers_complete_analysis.ipynb      # Complete reproducible analysis
```

### New Files (v1.0 Release)

| File | Purpose | Status |
|------|---------|--------|
| `CITATION.cff` | Machine-readable citation for GitHub | ✅ Added |
| `.gitignore` | Git ignore patterns | ✅ Added |
| `Deep_Research_Engine_Multi_LLM_Talent_Discovery.pdf` | Published manuscript (35 pages) | ✅ Added |
| `notebooks/frontiers_complete_analysis.ipynb` | Interactive Jupyter notebook | ✅ Added |

## 🔍 Key Findings by Section

### Section 3.1: Internal Platform Validation
- **Result**: AUC 0.81 [95% CI: 0.78-0.84]
- **Data**: N=18,337 adults, 10-fold cross-validation
- **Figure**: Figure 1 (model calibration and effect sizes)

### Section 3.2: External Commercial Validation
- **Result**: AUC 0.94 [95% CI: 0.91-0.96]
- **Data**: Banking deployment, N>5,000, objective behavioral ground truth (loan repayment)
- **Caveat**: Single institution, requires independent replication

### Section 3.3: Equal-Feature Baseline (Section 3.5)
- **Finding**: Facial features (AUC=0.82) substantially outperform questionnaire features (AUC=0.54)
- **Data**: N=428 children, facial vs demographic/questionnaire features
- **Figure**: Figure 3, **Table**: Table 5
- **Interpretation**: Genuine facial signal, not demographic confound

### Section 3.4: Human Expert Baseline Comparison
- **Finding**: AI r=0.351 vs Expert Avg r=0.291 (+6.0% improvement)
- **Data**: N=250, two licensed clinical psychologists
- **Statistical Test**: Fisher's z-test, z=0.74, p=0.46 (not significant)
- **Figure**: Figure 2
- **Table**: Table 4

## ⚠️ Important Limitations

1. **Causal Inference**: Cross-sectional design cannot establish causality. Behavioral confirmation hypothesis remains exploratory.

2. **Out-of-Distribution Generalization**: Models trained on adults (ages 18-65+) applied to children (ages 6-18). Age-related facial changes and personality maturation not accounted for.

3. **Fairness Audit Gaps**: No demographic stratification (race, ethnicity, gender, SES, disability) for N=18,337 training cohort. Buolamwini & Gebru (2018) bias patterns in facial AI not audited.

4. **Temporal Validation**: Platform data spans only 5 months (July 2025 - January 2026). Insufficient for rigorous temporal stability testing.

5. **Cultural Generalizability**: Barrett (2019) critique on emotional expression universality not addressed. Cross-cultural validity unvalidated.

6. **Conflict of Interest**: Author is founder/CEO of Talents.kids. System generates revenue through commercial subscriptions and consulting. See Conflict of Interest Statement in manuscript.

## 🔄 Future Research Directions

**Critical Next Steps:**
1. **Pre-registered Multi-Site Studies** (Open Science Framework)
   - Minimum N≥500 per site across ≥3 geographically diverse sites
   - SCID-5-PD diagnostic assessment
   - Independent expert review

2. **Cross-Cultural Validation**
   - East Asia (N=2,000), Sub-Saharan Africa (N=1,000), Latin America (N=1,000)
   - Demographic stratification by ethnicity, gender, SES

3. **Fairness Audits**
   - AUC by race/ethnicity/gender/disability
   - Fitzpatrick skin tone analysis
   - Intersectional demographic analysis

4. **Longitudinal Mediation Analysis**
   - Track facial features → social treatment → personality development
   - Test behavioral confirmation vs genetic pleiotropy hypotheses

5. **Mechanistic Studies**
   - fMRI/EEG neural correlates
   - Behavioral confirmation experiments
   - GWAS genetic pathways

## 📋 Data Availability

**Publicly Available:**
- All code and scripts at this repository
- Anonymized datasets (human_expert_baseline, equal_feature_baseline)
- Analysis protocols and hyperparameters

**Not Publicly Available:**
- Individual children's photographs (GDPR/COPPA)
- Full training dataset (commercial confidentiality)
- Banking deployment data (NDA with partner institution)

**For Researchers:**
Contact ds@talents.kids to discuss data use agreements for legitimate research purposes.

## 📄 License

This repository is licensed under the **Creative Commons Attribution 4.0 International License** (CC-BY-4.0).

You are free to:
- Share (copy and redistribute the material)
- Adapt (remix, transform, and build upon the material)

Under the following terms:
- Attribution (give appropriate credit)

See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

We thank:
- The 428 Talents.kids platform users and their families who participated
- Tatiana Yu. Novinskaya, MSc (Novosibirsk State Medical University) and the licensed child neuropsychologist for expert ratings
- NVIDIA for computational resources
- The anonymous banking institution for commercial validation partnership

## ✉️ Contact

**Author**: Dmitriy Sergeev
**Email**: ds@talents.kids
**Affiliation**: Talents.Kids, TEMNIKOVA LDA, Lisbon, Portugal

---

**Last Updated**: February 5, 2026
**Repository**: https://github.com/Talents-kids/facial-personality-talent-discovery
**Paper Status**: Preprint (prepared for submission to Frontiers in Psychology)
