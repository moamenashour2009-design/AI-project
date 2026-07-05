# ✈️ Safety-Aware CNN-LSTM for Turbofan Engine RUL Prediction

> **This model does not just predict when an engine will fail — it tells you *why*, *how fast*, and whether trusting that prediction is safe.**

A physics-informed, explainable deep learning framework for **Remaining Useful Life (RUL)** prediction of aircraft turbofan engines. Built for maintenance engineers, not just benchmark leaderboards.

This repository contains the code, pipelines, and generated artifacts accompanying the paper:
*"AI-driven Safety-Aware RUL Prediction with Operationally-Structured SHAP Explainability for Jet Engine Maintenance Decision Support"* (Moamen Ashour, 2026).

---

## 🧠 The Core Idea

Most RUL models optimize one thing: accuracy. This framework optimizes three:

| Priority | What it means |
|---|---|
| 🔴 **Safety** | Overestimating RUL when an engine is near failure is penalized 3× during training |
| 🟡 **Accuracy** | Competitive RMSE across all 4 NASA CMAPSS datasets |
| 🟢 **Interpretability** | SHAP explains which sensors are degrading — per engine and fleet-wide |

---

## 📊 Results at a Glance

Full model (all components combined) on the held-out test set:

| Dataset | RMSE (cycles) | MAE (cycles) | NASA Score | Unsafe Predictions |
|---|---|---|---|---|
| FD001 | 18.99 | 13.94 | 580.29 | 35.0% |
| FD002 | 14.49 | 11.08 | 786.43 | 40.7% |
| FD003 | 15.86 | 12.33 | 366.03 | 33.0% |
| FD004 | 17.10 | 12.82 | 1089.95 | 39.7% |

**Compared to the baseline** (raw sensors + delta features only, no physics features, no attention, no penalty loss): average unsafe prediction rate drops from **47.1% → 37.1%** across all four datasets — a **10-percentage-point (≈21% relative) reduction** in dangerous overestimations, at the cost of higher average RMSE on single-condition datasets (FD001, FD003). This trade-off is discussed in detail in the paper's Discussion section.

---

## 🏗️ Architecture Overview

```
Raw Sensor Data (CMAPSS)
        │
        ▼
┌──────────────────────────────────┐
│  PREPROCESSING                   │
│  • Dead sensor removal (FIX-T1)  │
│  • Flight regime clustering      │
│  • Engine-specific delta features│
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  PHYSICS FEATURE ENGINEERING     │
│  • Compressor Efficiency         │
│  • Corrected Coolant-Flow Index  │
│  • Thermal Stress Index          │
│  • Combustion Efficiency         │
│  • Cumulative Thermal Fatigue    │
│  • Flow Capacity Degradation     │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  CNN-LSTM + ATTENTION            │
│  CNN Block (2× Conv1D + BN)      │
│     ↓                            │
│  2-Layer LSTM (hidden=64)        │
│     ↓                            │
│  Temporal Attention              │
│     ↓                            │
│  FC Head → RUL (cycles)          │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  OPERATIONAL OUTPUTS             │
│  • RUL prediction                │
│  • Degradation velocity          │
│  • Engine-level SHAP             │
│  • Fleet-level SHAP              │
└──────────────────────────────────┘
```

---

## ⚙️ Key Features

### 🛡️ Aviation Penalty Loss
A custom loss function that applies a **3× penalty multiplier** to dangerous overestimation errors when the true RUL is ≤ 45 cycles. The 45-cycle threshold aligns with the aviation industry's standard "watch window."

```python
# Unsafe overestimation → penalized 3×
# Safe underestimation → standard MSE
w = 3.0  if  (true_RUL ≤ 45)  and  (predicted > true_RUL)
w = 1.0  otherwise
```

Training uses a **2-phase schedule**: 10 warmup epochs with standard MSE, then penalty loss for remaining epochs.

### 🔬 Physics-Informed Features
Six features derived from the active sensor set, as implemented in `compute_physics_features()`. Two of these (marked *Adapted*) substitute proxy sensors in place of the classical gas-turbine formula inputs, due to the specific sensors retained after dead-sensor removal — see the paper's Section 2.4.2 for full discussion of these substitutions.

| Feature | Formula (as implemented) | Type |
|---|---|---|
| Compressor Efficiency | `(T3_ideal − T24) / (T50 − T24)`, where `T3_ideal = T24 · (NRc/Nc)^0.2857` | Adapted |
| Corrected Coolant-Flow Index | `W31 / √(T24 / T_ref)` | Adapted |
| Thermal Stress Index | `T50 / Ps30` | Original |
| Combustion Efficiency | `(T50 × Nc) / phi` | Original |
| Cumulative Thermal Fatigue | `expanding_mean(T50)` | Original |
| Flow Capacity Degradation | `Nf / BPR` | Original |

> ⚠️ **Note on Feature 2:** This feature uses `W31` (HPT coolant bleed) rather than physical fan speed `Nf`. It is named "Corrected Coolant-Flow Index" (not "Corrected Fan Speed") to accurately reflect what it computes. See the paper's Limitations section for discussion.

### 📈 Degradation Velocity Indicator
Smoothed first derivative of predicted RUL, classifying engine health into three actionable states:

| State | Velocity Threshold | Meaning |
|---|---|---|
| 🟢 Stable | `v > −0.2` | Engine healthy, no action needed |
| 🟡 Standard Linear Wear | `−1.2 ≤ v ≤ −0.2` | Normal degradation, monitor |
| 🚨 Accelerating Failure | `v < −1.2` | Immediate maintenance alert |

### 🔍 Dual-Level SHAP Explainability
- **Engine-level SHAP** — *"Which sensors are driving THIS engine's degradation right now?"* Computed on last 10 sequences of a single engine.
- **Fleet-level SHAP** — *"Which sensors consistently drive degradation across all engines?"* Averaged across all validation engines, using the last 5 sequences (failure zone) per engine.

---

## 📁 Project Structure

```
├── code/                             # Core model and utility scripts
├── pipelines/                        # Preprocessing and feature engineering pipelines
├── brain/                            # Saved model weights (.pt) and preprocessors (.pkl)
├── graphs generated/                 # Output figures (loss curves, SHAP plots, RUL trajectories)
│
├── M1_baseline.py / T1_baseline.py           # Ablation V1: raw sensors + delta features
├── M2_delta_features.py / T2_delta_features.py  # Ablation V2: delta features variant
├── All_features_model.py                     # Full model training pipeline (V6)
├── All_features_test_pipline.py              # Test pipeline (official CMAPSS test sets)
│
│── [Generated after training, in brain/ and graphs generated/]
├── cnn_lstm_brain_FD00X.pt           # Saved model weights per dataset
├── preprocessors_FD00X.pkl           # Scalers, KMeans, feature config
├── cnn_lstm_loss_FD00X.png           # Training loss curve
├── deg_velocity_val_FD00X.png        # Degradation velocity chart
├── cnn_lstm_shap_engine_FD00X.png    # Engine-level SHAP
└── cnn_lstm_shap_fleet_FD00X.png     # Fleet-level SHAP
```

*(One set of output files is generated per dataset: FD001–FD004. Raw CMAPSS data files are not included in this repository — see download instructions below.)*

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib shap joblib
```

### 2. Download the CMAPSS Dataset

Download from the [NASA Prognostics Data Repository](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6):

Place the following files in the project root (these are **not included** in this repository):
```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

### 3. Train

```bash
python All_features_model.py
```

The script trains independently on each dataset found. It automatically skips any dataset whose training file is missing. If a saved model (`cnn_lstm_brain_FDxxx.pt`) already exists in `brain/`, it loads the saved weights and skips retraining.

### 4. Evaluate on Official Test Sets

```bash
python All_features_test_pipline.py
```

---

## 🔧 Configuration

All key hyperparameters are defined at the top of `All_features_model.py`:

```python
RUL_MAX            = 125     # Piecewise linear RUL cap
SEQUENCE_LENGTH    = 30      # Input window (cycles)
TRAIN_EPOCHS       = 40      # Maximum training epochs
WARMUP_EPOCHS      = 10      # MSE-only warmup before penalty loss
PATIENCE_LIMIT     = 6       # Early stopping patience
BATCH_SIZE         = 64

# Penalty loss parameters
penalty_multiplier = 3.0     # Weight on dangerous overestimations
safety_threshold   = 45.0    # RUL threshold defining the "watch window"

# Physics feature constants
T_REF              = 518.67  # Standard sea-level reference temperature (°R)
GAMMA_EXP          = 0.2857  # (gamma-1)/gamma for air (gamma = 1.4)
```

---

## 🧩 CMAPSS Dataset Summary

| Sub-dataset | Training Engines | Operating Conditions | Fault Modes | Difficulty |
|---|---|---|---|---|
| FD001 | 100 | 1 | 1 (HPC) | Simplest |
| FD002 | 260 | 6 | 1 (HPC) | Medium |
| FD003 | 100 | 1 | 2 (HPC + Fan) | Medium |
| FD004 | 249 | 6 | 2 (HPC + Fan) | Hardest |

---

## 🔬 Methodology Notes

### Data Leakage Prevention
Two explicit fixes are documented in the code:

- **FIX-T1**: Dead sensor detection (`detect_dead_sensors()`) is called on the **training split only** — not the full dataset — preventing validation data from influencing feature selection.
- **FIX-T2**: The physics scaler is saved as `None` when no physics features are computed, preventing an unfitted `StandardScaler` from being serialized to the `.pkl` file.

### Multi-Dataset Regime Handling
- **FD001, FD003** (1 operating condition): All cycles assigned to a single regime; no clustering performed.
- **FD002, FD004** (6 operating conditions): K-Means clustering (`K=6`) fitted on training settings only; cluster assignments transferred to validation/test via fitted centroids.

### Model Persistence
Each dataset produces independent model weights and preprocessors. No parameters are shared across datasets. If weights already exist, the training loop is skipped — safe to re-run.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{ashour2026rul,
  title  = {AI-driven Safety-Aware RUL Prediction with Operationally-Structured
            SHAP Explainability for Jet Engine Maintenance Decision Support},
  author = {Ashour, Moamen},
  year   = {2026},
  note   = {Preprint. Code and data available via Zenodo: [DOI to be added upon upload]}
}
```

---

## 📖 Related Work

| Paper | Contribution | Limitation addressed by this work |
|---|---|---|
| Saxena et al. (2008) | CMAPSS dataset | — |
| Sherifi (2024) | BLSTM baseline; called for XAI | No physics features, no safety loss |
| Mitici et al. (2023) | Probabilistic RUL + maintenance scheduling | No sensor-level explanation, no asymmetric loss |
| Lundberg & Lee (2017) | SHAP framework | — |

---

## ⚠️ Limitations

- Trained and evaluated on **simulation data** (CMAPSS). Real engine validation is required before operational deployment.
- The penalty multiplier (`α = 3.0`) and safety threshold (45 cycles) were tuned for CMAPSS. These should be recalibrated for real-world engine data based on your organization's risk tolerance.
- Two physics features (Compressor Efficiency, Corrected Coolant-Flow Index) use adapted proxy sensors rather than the classical formula inputs, due to the active sensor set available after preprocessing. See the accompanying paper's Limitations section for details.
- Fleet-level SHAP computation is slow on CPU (~5–15 min per dataset). A GPU reduces this to ~1–2 min.

---

<div align="center">
  <sub>Built with safety in mind. Because the cost of overestimating remaining life is not a number on a leaderboard.</sub>
</div>
