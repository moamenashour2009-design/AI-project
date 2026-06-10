"""
cnn_lstm_multi_test.py
CNN-LSTM Physics-Informed v3 — Multi-Dataset Test Pipeline (FD001–FD004)

Auto-detects which datasets have all required files:
    test_FDxxx.txt + RUL_FDxxx.txt + cnn_lstm_brain_FDxxx.pt + preprocessors_FDxxx.pkl

Evaluates each found dataset using its own dedicated brain.
Prints a comparison table AND degradation velocity analysis per dataset.

FIXES APPLIED:
  FIX-P1: .count() replaced with .max() for engine cycle selection (semantic accuracy)
  FIX-P2: Guard added for empty df_res before division — prevents ZeroDivisionError
           in the impossible-but-defensive case where engine IDs don't match RUL file
  FIX-P3: physics_scaler.transform guarded against None (consistent with FIX-T2
           in training file where physics_scaler is now saved as None when unused)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
# ====================================================================
# ABLATION FLAGS — must match training flags for this variant
# ====================================================================
USE_DELTA_FEATURES   = False
USE_PHYSICS_FEATURES = False
USE_ATTENTION        = False
USE_PENALTY_LOSS     = False
# ====================================================================
# MODEL ARCHITECTURE — must be identical to training file
# ====================================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        w = torch.softmax(self.fc(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1), w


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x.permute(0, 2, 1)).permute(0, 2, 1)


class CNNLSTMSequential(nn.Module):
    def __init__(self, input_size, cnn_channels=64, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.cnn  = CNNBlock(input_size, cnn_channels)
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.attn = TemporalAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        c       = self.cnn(x)
        h0      = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0      = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _  = self.lstm(c, (h0, c0))
        if USE_ATTENTION:
            ctx, _ = self.attn(out)
        else:
            ctx = out[:, -1, :]
        return self.head(ctx)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

PHYSICS_REQUIREMENTS = {
    'Phys_Compressor_Efficiency'     : ['sensor_2', 'sensor_4', 'sensor_9',  'sensor_14'],
    'Phys_Corrected_Fan_Speed'       : ['sensor_2', 'sensor_20'],
    'Phys_Thermal_Stress_Index'      : ['sensor_4', 'sensor_11'],
    'Phys_Combustion_Efficiency'     : ['sensor_4', 'sensor_9',  'sensor_12'],
    'Phys_Cumulative_Thermal_Fatigue': ['sensor_4'],
    'Phys_Flow_Capacity_Degradation' : ['sensor_8', 'sensor_15'],
}


def nasa_score(y_true, y_pred):
    """Asymmetric NASA CMAPSS scoring. Lower is better."""
    d = y_pred - y_true
    return float(np.sum(
        np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    ))


def compute_physics_features(df, active_sensors, expected_features,
                              T_REF=518.67, GAMMA_EXP=0.2857):
    """
    Compute only the physics features that were present during training.
    Uses expected_features (from preprocessors pkl) to guarantee exact
    consistency with the training pipeline — even if active_sensors differ.
    """
    df = df.copy().sort_values(['engine_id', 'cycle'])

    for feat in expected_features:
        required = PHYSICS_REQUIREMENTS.get(feat, [])
        if not all(s in active_sensors for s in required):
            continue

        if feat == 'Phys_Compressor_Efficiency':
            T2         = df['sensor_2']
            T3         = df['sensor_4']
            ratio      = df['sensor_14'] / (df['sensor_9'] + 1e-5)
            T3_ideal   = T2 * (ratio.clip(lower=0.01) ** GAMMA_EXP)
            df[feat]   = (T3_ideal - T2) / (T3 - T2 + 1e-5)

        elif feat == 'Phys_Corrected_Fan_Speed':
            theta    = df['sensor_2'] / T_REF
            df[feat] = df['sensor_20'] / np.sqrt(theta.clip(lower=0.01))

        elif feat == 'Phys_Thermal_Stress_Index':
            df[feat] = df['sensor_4'] / (df['sensor_11'] + 1e-5)

        elif feat == 'Phys_Combustion_Efficiency':
            df[feat] = (
                (df['sensor_4'] * df['sensor_9']) / (df['sensor_12'] + 1e-5)
            )

        elif feat == 'Phys_Cumulative_Thermal_Fatigue':
            df[feat] = (
                df.groupby('engine_id')['sensor_4']
                .transform(lambda x: x.expanding().mean())
            )

        elif feat == 'Phys_Flow_Capacity_Degradation':
            df[feat] = df['sensor_8'] / (df['sensor_15'] + 1e-5)

    return df


# ====================================================================
# DATASET DETECTION
# ====================================================================

DATASET_IDS      = ['FD001', 'FD002', 'FD003', 'FD004']
index_labels     = ['engine_id', 'cycle']
sensor_labels_g  = [f'sensor_{i}' for i in range(1, 22)]

print('=' * 80)
print('  CNN-LSTM v3 ENHANCED PHYSICS — MULTI-DATASET TEST PIPELINE')
print('=' * 80)

print('\n[SCAN] Checking available datasets...\n')

available = []
for ds in DATASET_IDS:
    ablation_tag = (f"d{int(USE_DELTA_FEATURES)}"
                f"p{int(USE_PHYSICS_FEATURES)}"
                f"a{int(USE_ATTENTION)}"
                f"l{int(USE_PENALTY_LOSS)}")

    needed = [f'test_{ds}.txt', f'RUL_{ds}.txt',
           f'cnn_lstm_brain_{ds}_{ablation_tag}.pt',
           f'preprocessors_{ds}_{ablation_tag}.pkl']
    missing = [f for f in needed if not os.path.exists(f)]

    if not missing:
        print(f'  ✅  {ds} — all files found, queued for evaluation')
        available.append(ds)
    else:
        print(f'  ❌  {ds} — missing: {", ".join(missing)}')

if not available:
    raise FileNotFoundError(
        '\n[ERROR] No complete dataset found.\n'
        'Need: test_FDxxx.txt, RUL_FDxxx.txt, '
        'cnn_lstm_brain_FDxxx.pt, preprocessors_FDxxx.pkl'
    )

print(f'\n  → Evaluating: {", ".join(available)}\n')

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_summary = {}

# ====================================================================
# MAIN TEST LOOP — one iteration per available dataset
# ====================================================================

for dataset_id in available:

    print(f'\n{"=" * 80}')
    print(f'  EVALUATING {dataset_id}')
    print(f'{"=" * 80}')

    # ── STEP 1: Load preprocessors ───────────────────────────────────
    prep = joblib.load(f'preprocessors_{dataset_id}_{ablation_tag}.pkl')
    kmeans              = prep['kmeans']
    regime_scalers_pool = prep['regime_scalers_pool']
    physics_scaler      = prep['physics_scaler']   # may be None (FIX-T2)
    delta_scaler        = prep['delta_scaler']
    global_regime_mean  = prep['global_regime_mean']
    active_sensors      = prep['active_sensors']
    physics_features    = prep['physics_features']
    delta_features      = prep['delta_features']
    X_columns           = prep['X_columns']
    setting_labels      = prep['setting_labels']
    dead_sensors        = prep['dead_sensors']
    RUL_MAX             = prep['RUL_MAX']
    SEQUENCE_LENGTH     = prep['SEQUENCE_LENGTH']
    T_REF               = prep.get('T_REF',     518.67)
    GAMMA_EXP           = prep.get('GAMMA_EXP', 0.2857)
    n_regimes           = prep.get('n_regimes', 6)

    print(f'\n  [STEP 1] Preprocessors loaded')
    print(f'    Features : {len(X_columns)} total '
          f'({len(active_sensors)} sensors, '
          f'{len(physics_features)} physics, '
          f'{len(delta_features)} delta)')
    print(f'    Regimes  : K={n_regimes} '
          f'({"KMeans" if kmeans else "single regime"})')
    print(f'    Dead sensors removed: {dead_sensors}')
    # FIX-P3: inform user if physics_scaler is None
    if physics_scaler is None and physics_features:
        print(f'  ⚠️  physics_scaler is None but physics_features is non-empty. '
              f'Check training pkl consistency.')
    elif physics_scaler is None:
        print(f'    Physics scaler: None (no physics features for this dataset)')

    # ── STEP 2: Load model ───────────────────────────────────────────
    model = CNNLSTMSequential(input_size=len(X_columns)).to(device)
    model.load_state_dict(
        torch.load(f'cnn_lstm_brain_{dataset_id}_{ablation_tag}.pt',
                   map_location=device, weights_only=True)
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  [STEP 2] Brain loaded — {n_params:,} parameters | '
          f'device: {device}')

    # ── STEP 3: Load and preprocess test data ────────────────────────
    complete_headers = index_labels + setting_labels + sensor_labels_g
    test_raw  = pd.read_csv(f'test_{dataset_id}.txt',
                             sep=r'\s+', names=complete_headers)
    test_data = test_raw.drop(columns=dead_sensors).copy()

    print(f'\n  [STEP 3] Test data: '
          f'{test_data["engine_id"].nunique()} engines, '
          f'{len(test_data):,} rows')

    # Flight regime assignment
    if kmeans is None:
        test_data['flight_regime'] = 0
    else:
        test_data['flight_regime'] = kmeans.predict(test_data[setting_labels])

    # Delta features from each test engine's own first 20 cycles
    # (NOT from train baselines — using train baselines caused NASA score
    # to explode to 19000+ in earlier versions of this pipeline)
    if USE_DELTA_FEATURES:
        test_baseline = (
            test_data[test_data['cycle'] <= 20]
            .groupby(['engine_id', 'flight_regime'])[active_sensors]
            .mean()
            .reset_index()
        )
        test_baseline.columns = (
            ['engine_id', 'flight_regime'] + [f'{s}_base' for s in active_sensors]
        )
        test_data = test_data.merge(
            test_baseline, on=['engine_id', 'flight_regime'], how='left'
        )
        for s in active_sensors:
            test_data[f'{s}_base'] = test_data[f'{s}_base'].fillna(
                test_data['flight_regime'].map(global_regime_mean[s])
            )
            test_data[f'delta_{s}'] = test_data[s] - test_data[f'{s}_base']
            test_data.drop(columns=[f'{s}_base'], inplace=True)

    # Physics features — only those in expected_features from training
    if physics_features:
        test_data = compute_physics_features(
            test_data, active_sensors, physics_features, T_REF, GAMMA_EXP
        )

    # Scaling — apply saved scalers (fitted on training data only)
    test_data[active_sensors] = test_data[active_sensors].astype(float)
    for rid, sc in regime_scalers_pool.items():
        m = test_data['flight_regime'] == rid
        if m.any():
            test_data.loc[m, active_sensors] = sc.transform(
                test_data.loc[m, active_sensors]
            )

    # FIX-P3: guard physics_scaler against None before calling .transform()
    # physics_scaler is None when training produced no physics features
    # (consistent with FIX-T2 in training file)
    if physics_features and physics_scaler is not None:
        test_data[physics_features] = physics_scaler.transform(
            test_data[physics_features]
        )

    if delta_features and delta_scaler is not None:
        test_data[delta_features] = delta_scaler.transform(test_data[delta_features])
    print(f'  ✅ Preprocessing complete — features ready')

    # ── STEP 4: Predict RUL ──────────────────────────────────────────
    preds_per_engine = {}
    skipped          = []

    for eid in sorted(test_data['engine_id'].unique()):
        eng = test_data[test_data['engine_id'] == eid]
        if len(eng) < SEQUENCE_LENGTH:
            skipped.append(eid)
            continue
        seq = eng[X_columns].values[-SEQUENCE_LENGTH:]
        t   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds_per_engine[eid] = max(0.0, model(t).item())

    if skipped:
        print(f'  ⚠️  Skipped {len(skipped)} engines '
              f'(< {SEQUENCE_LENGTH} cycles): {skipped}')
    print(f'\n  [STEP 4] Predicted RUL for {len(preds_per_engine)} engines')

    # ── STEP 5: Compute metrics ──────────────────────────────────────
    true_rul = pd.read_csv(f'RUL_{dataset_id}.txt', sep=r'\s+',
                            header=None, names=['true_RUL'])
    true_rul['engine_id'] = range(1, len(true_rul) + 1)

    rows = []
    for eid, pred in preds_per_engine.items():
        row = true_rul[true_rul['engine_id'] == eid]
        if row.empty:
            continue
        tv  = float(row['true_RUL'].values[0])
        tvc = min(tv, RUL_MAX)
        rows.append({
            'engine_id'      : eid,
            'predicted_RUL'  : round(pred, 2),
            'true_RUL'       : tv,
            'true_RUL_capped': tvc,
            'error'          : round(pred - tvc, 2),
            'abs_error'      : round(abs(pred - tvc), 2),
        })

    df_res = pd.DataFrame(rows)

    # FIX-P2: Guard against empty df_res before any division or metric
    # computation. Cannot happen with standard CMAPSS files (engine IDs
    # always match) but prevents ZeroDivisionError in edge cases.
    if len(df_res) == 0:
        print(f'  ⚠️  No matching engine IDs between predictions and '
              f'RUL_{dataset_id}.txt. Skipping metrics for {dataset_id}.')
        continue

    y_true = df_res['true_RUL_capped'].values
    y_pred = df_res['predicted_RUL'].values

    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    nasa  = nasa_score(y_true, y_pred)
    over  = (df_res['error'] > 0).sum()
    under = (df_res['error'] < 0).sum()

    print(f'\n  [STEP 5] Metrics — {dataset_id} ({len(df_res)} unseen engines)')
    print(f'  {"─" * 52}')
    print(f'  RMSE       : {rmse:.4f} cycles')
    print(f'  MAE        : {mae:.4f} cycles')
    print(f'  NASA Score : {nasa:.2f}  (lower is better)')
    print(f'  {"─" * 52}')
    print(f'  Overestimated  (unsafe): {over:>4}  '
          f'({100*over/len(df_res):.1f}%)')
    print(f'  Underestimated (safe)  : {under:>4}  '
          f'({100*under/len(df_res):.1f}%)')
    print(f'  Max overestimate       : {df_res["error"].max():.2f} cycles')
    print(f'  Max underestimate      : {df_res["error"].min():.2f} cycles')
    print(f'  {"─" * 52}')

    test_summary[dataset_id] = {
        'rmse'     : rmse,
        'mae'      : mae,
        'nasa'     : nasa,
        'n_engines': len(df_res),
        'over_pct' : 100 * over / len(df_res),
    }

    df_res.to_csv(f'test_predictions_{dataset_id}.csv', index=False)
    print(f'  ✅ Predictions saved → test_predictions_{dataset_id}.csv')

    # ── STEP 6: Evaluation charts ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color='steelblue',
                edgecolors='navy', s=40)
    lim = [0, max(y_true.max(), y_pred.max()) + 10]
    ax1.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True RUL (cycles)', fontsize=11)
    ax1.set_ylabel('Predicted RUL (cycles)', fontsize=11)
    ax1.set_title(f'CNN-LSTM v3 {dataset_id}: Predicted vs True RUL',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.4)
    ax1.text(0.05, 0.92, f'RMSE = {rmse:.2f}\nMAE  = {mae:.2f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax2 = axes[1]
    ax2.hist(df_res['error'], bins=30, color='darkorange',
             edgecolor='black', alpha=0.75)
    ax2.axvline(0, color='red', ls='--', lw=2, label='Zero error')
    ax2.axvline(df_res['error'].mean(), color='blue', lw=2,
                label=f'Mean = {df_res["error"].mean():.2f}')
    ax2.set_xlabel('Error (Predicted − True)', fontsize=11)
    ax2.set_ylabel('Number of Engines', fontsize=11)
    ax2.set_title(f'{dataset_id}: Error Distribution\n'
                  '(Positive = Overestimate = Unsafe)',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(f'test_eval_{dataset_id}.png', dpi=300)
    plt.close(fig)
    print(f'  ✅ Evaluation chart → test_eval_{dataset_id}.png')

    # ── STEP 7: SHAP top 5 sensors ───────────────────────────────────
    print(f'\n  [STEP 7] SHAP attribution — {dataset_id}...')
    try:
        eligible = sorted([
            eid for eid in test_data['engine_id'].unique()
            if len(test_data[test_data['engine_id'] == eid]) >= SEQUENCE_LENGTH + 5
        ])
        if not eligible:
            eligible = sorted([
                eid for eid in test_data['engine_id'].unique()
                if len(test_data[test_data['engine_id'] == eid]) >= SEQUENCE_LENGTH
            ])
        sample_eid = eligible[0]
        eng_sl     = test_data[test_data['engine_id'] == sample_eid]
        seqs       = [
            eng_sl[X_columns].values[i: i + SEQUENCE_LENGTH]
            for i in range(len(eng_sl) - SEQUENCE_LENGTH + 1)
        ]

        bg_seqs = []
        for eid in sorted(test_data['engine_id'].unique()):
            if eid == sample_eid:
                continue
            eng_tmp = test_data[test_data['engine_id'] == eid]
            if len(eng_tmp) >= SEQUENCE_LENGTH:
                bg_seqs.append(eng_tmp[X_columns].values[-SEQUENCE_LENGTH:])
            if len(bg_seqs) >= 50:
                break

        if not bg_seqs:
            n_bg    = min(10, len(seqs))
            bg_seqs = seqs[:n_bg]
            print(f'  ⚠️  Using same-engine background ({n_bg} sequences)')

        bg_t  = torch.tensor(np.array(bg_seqs), dtype=torch.float32).to(device)
        smp_t = torch.tensor(
            np.array(seqs[-min(10, len(seqs)):]), dtype=torch.float32
        ).to(device)
        print(f'  Background: {bg_t.shape[0]} sequences | '
              f'Explaining: {smp_t.shape[0]} sequences (engine {sample_eid})')

        model.eval()
        explainer = shap.GradientExplainer(model, bg_t)
        raw       = explainer.shap_values(smp_t)
        arr       = np.array(raw[0] if isinstance(raw, list) else raw)

        if arr.ndim == 4:
            arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr.mean(-1)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        vals_2d = arr.mean(axis=1)

        shap.summary_plot(vals_2d, feature_names=X_columns,
                          plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f'test_shap_all_{dataset_id}.png', dpi=300)
        plt.close()
        print(f'  ✅ All-feature SHAP → test_shap_all_{dataset_id}.png')

        s_mask  = np.array([c in active_sensors for c in X_columns])
        s_vals  = vals_2d[:, s_mask]
        s_imp   = np.abs(s_vals).mean(axis=0)
        s_names = np.array(X_columns)[s_mask]
        top5    = np.argsort(s_imp)[::-1][:5]

        print(f'\n  🔥 Top 5 sensors driving RUL — '
              f'{dataset_id} (engine {sample_eid}):')
        for i, idx in enumerate(top5, 1):
            print(f'    {i}. {s_names[idx]:<20} | SHAP: {s_imp[idx]:.5f}')

        shap.summary_plot(
            s_vals[:, top5],
            feature_names=[s_names[i] for i in top5],
            plot_type='bar',
            show=False,
        )
        plt.title(f'Top 5 Sensor SHAP Importances — {dataset_id}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'test_shap_top5_{dataset_id}.png', dpi=300)
        plt.close()
        print(f'  ✅ Top-5 sensor SHAP → test_shap_top5_{dataset_id}.png')

    except Exception as e:
        print(f'  ⚠️  SHAP skipped ({dataset_id}): {e}')

    # ── STEP 8: Degradation velocity ─────────────────────────────────
    print(f'\n  [STEP 8] Degradation velocity analysis — {dataset_id}...')
    try:
        # FIX-P1: Use .max() instead of .count() to find engine with most
        # flight cycles. For CMAPSS sequential cycles both give identical
        # results, but .max() is semantically correct and clearer to read.
        engine_cycle_counts = test_data.groupby('engine_id')['cycle'].max()
        best_engine = engine_cycle_counts.idxmax()
        eng_sl = test_data[
            test_data['engine_id'] == best_engine
        ].sort_values('cycle')

        seqs = [
            eng_sl[X_columns].values[i: i + SEQUENCE_LENGTH]
            for i in range(len(eng_sl) - SEQUENCE_LENGTH + 1)
        ]
        if not seqs:
            raise RuntimeError('Engine has insufficient cycles for trajectory')

        seq_tensor = torch.tensor(
            np.array(seqs), dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            preds = model(seq_tensor).cpu().numpy().flatten()

        smoothed = pd.Series(preds).rolling(
            window=5, min_periods=1
        ).mean().values
        velocity = np.insert(np.diff(smoothed), 0, 0.0)

        print(f'\n  --- Degradation Velocity: Last 5 cycles '
              f'(Engine {best_engine}) ---')
        for idx in range(max(0, len(preds) - 5), len(preds)):
            cyc      = SEQUENCE_LENGTH + idx
            pred_val = preds[idx]
            vel      = velocity[idx]
            if vel < -1.2:
                status = '🚨 ACCELERATING FAILURE'
            elif vel > -0.2:
                status = '🟢 STABLE (CAPPED BOUND)'
            else:
                status = '🟡 STANDARD LINEAR WEAR'
            print(f'    Cycle {cyc:<4} | RUL: {pred_val:<6.2f} | '
                  f'd(RUL)/dt: {vel:<6.3f} | {status}')

        fig, ax = plt.subplots(figsize=(11, 5))
        x_axis = range(SEQUENCE_LENGTH, len(eng_sl) + 1)
        ax.plot(x_axis, preds, label='Raw Predicted RUL',
                color='darkorange', alpha=0.4, linestyle='--')
        ax.plot(x_axis, smoothed, label='Smoothed Health Trend',
                color='red', linewidth=3)
        ax.set_title(
            f'CNN-LSTM v3 {dataset_id} — Engine {best_engine} '
            f'Health Trajectory',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Operational Flight Cycles')
        ax.set_ylabel('Remaining Useful Life (cycles)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'deg_velocity_{dataset_id}.png', dpi=300)
        plt.close(fig)
        print(f'  ✅ Degradation chart → deg_velocity_{dataset_id}.png')

    except Exception as e:
        print(f'  ⚠️  Degradation velocity skipped ({dataset_id}): {e}')


# ====================================================================
# FINAL COMPARISON TABLE
# ====================================================================
print(f'\n\n{"=" * 75}')
print(f'  MULTI-DATASET TEST RESULTS — CNN-LSTM v3 ENHANCED PHYSICS')
print(f'{"=" * 75}')
print(f'  {"Dataset":<8} {"Engines":>8} {"RMSE":>8} {"MAE":>8} '
      f'{"NASA Score":>12} {"Unsafe %":>10}')
print(f'  {"─" * 8} {"─" * 8} {"─" * 8} {"─" * 8} '
      f'{"─" * 12} {"─" * 10}')

for ds, m in test_summary.items():
    print(f'  {ds:<8} {m["n_engines"]:>8} {m["rmse"]:>8.4f} '
          f'{m["mae"]:>8.4f} {m["nasa"]:>12.2f} '
          f'{m["over_pct"]:>9.1f}%')

print(f'{"=" * 75}')

if len(test_summary) > 1:
    avg_rmse = np.mean([m['rmse'] for m in test_summary.values()])
    avg_nasa = np.mean([m['nasa'] for m in test_summary.values()])
    print(f'\n  Average RMSE across datasets : {avg_rmse:.4f} cycles')
    print(f'  Average NASA Score           : {avg_nasa:.2f}')

print(f'\n  Output files per dataset:')
print(f'    test_predictions_FDxxx.csv    ← per-engine predictions + errors')
print(f'    test_eval_FDxxx.png           ← predicted vs true + error histogram')
print(f'    test_shap_all_FDxxx.png       ← all-feature SHAP bar chart')
print(f'    test_shap_top5_FDxxx.png      ← top 5 sensor SHAP chart')
print(f'    deg_velocity_FDxxx.png        ← degradation velocity trajectory')