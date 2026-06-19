import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ====================================================================
# GLOBAL CONSTANTS
# ====================================================================
RUL_MAX         = 125
SEQUENCE_LENGTH = 30
T_REF           = 518.67    # Standard sea-level temperature (°R)
GAMMA_EXP       = 0.2857    # (gamma-1)/gamma for air (gamma = 1.4)
TRAIN_EPOCHS    = 40
WARMUP_EPOCHS   = 10        # Pure MSE warmup before penalty loss kicks in
PATIENCE_LIMIT  = 6
BATCH_SIZE      = 64
# ====================================================================
# ABLATION FLAGS — set True/False per experiment
# ====================================================================
USE_DELTA_FEATURES   = True
USE_PHYSICS_FEATURES = True
USE_ATTENTION        = True
USE_PENALTY_LOSS     = True
# ====================================================================
# DATASET CONFIGURATIONS
# ====================================================================
DATASET_CONFIGS = {
    'FD001': {'n_regimes': 1, 'description': '1 condition, 1 fault mode  (simplest)'},
    'FD002': {'n_regimes': 6, 'description': '6 conditions, 1 fault mode'},
    'FD003': {'n_regimes': 1, 'description': '1 condition, 2 fault modes'},
    'FD004': {'n_regimes': 6, 'description': '6 conditions, 2 fault modes (hardest)'},
}

PHYSICS_REQUIREMENTS = {
    'Phys_Compressor_Efficiency'     : ['sensor_2', 'sensor_4', 'sensor_9',  'sensor_14'],
    'Phys_Corrected_Fan_Speed'       : ['sensor_2', 'sensor_20'],
    'Phys_Thermal_Stress_Index'      : ['sensor_4', 'sensor_11'],
    'Phys_Combustion_Efficiency'     : ['sensor_4', 'sensor_9',  'sensor_12'],
    'Phys_Cumulative_Thermal_Fatigue': ['sensor_4'],
    'Phys_Flow_Capacity_Degradation' : ['sensor_8', 'sensor_15'],
}

# ====================================================================
# MODEL ARCHITECTURE
# ====================================================================

class AviationPenaltyLoss(nn.Module):
    def __init__(self, penalty_multiplier=3.0, safety_threshold=45.0):
        super().__init__()
        self.alpha     = penalty_multiplier
        self.threshold = safety_threshold

    def forward(self, predictions, targets):
        sq_err= (predictions - targets) ** 2
        mask= (targets <= self.threshold) & (predictions > targets)
        w= torch.where(mask,
                             torch.full_like(predictions, self.alpha),
                             torch.ones_like(predictions))
        return torch.mean(sq_err * w)


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
            ctx = out[:, -1, :]   # just take last timestep
        return self.head(ctx)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# FIX-T1: detect_dead_sensors() takes a DataFrame argument.
# Previously called on raw full dataset (before train/val split).
# Now called on train_matrix only → no leakage from val rows into
# the dead-sensor decision.
# In practice zero-variance sensors are zero-variance everywhere, so
# results are identical — but architecturally this is now correct.
def detect_dead_sensors(df, sensor_labels, threshold=1e-8):
    """
    Detects sensors with near-zero variance.
    Must be called on the TRAINING split only — not the full dataset.
    threshold=1e-8 keeps weak signals while removing truly flat sensors.
    """
    var  = df[sensor_labels].var()
    dead = list(var[var < threshold].index)
    return dead


def assign_flight_regimes(train_df, val_df, setting_labels, n_regimes):
    train_df = train_df.copy()
    val_df   = val_df.copy()

    if n_regimes == 1:
        # Single operating condition — no clustering needed.
        # Assign regime 0 to every row so downstream code works uniformly.
        train_df['flight_regime'] = 0
        val_df['flight_regime']   = 0
        return train_df, val_df, None

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    train_df['flight_regime'] = kmeans.fit_predict(train_df[setting_labels])
    val_df['flight_regime']   = kmeans.predict(val_df[setting_labels])
    return train_df, val_df, kmeans


def compute_delta_features(train_df, val_df, active_sensors, global_regime_mean):
    """
    Engine-specific delta = sensor − healthy baseline (own first 20 cycles).
    Falls back to global_regime_mean if engine has no early cycles.
    Baseline computed on TRAIN engines only — val/test engines use their
    own first-20-cycle baseline (computed separately in their respective
    preprocessing steps).
    """
    result = []
    for df in (train_df, val_df):
        df = df.copy().sort_values(['engine_id', 'cycle'])
        baseline = (df[df['cycle'] <= 20]
                    .groupby(['engine_id', 'flight_regime'])[active_sensors]
                    .mean()
                    .reset_index())
        baseline.columns = (['engine_id', 'flight_regime'] +
                            [f'{s}_base' for s in active_sensors])
        df = df.merge(baseline, on=['engine_id', 'flight_regime'], how='left')
        for s in active_sensors:
            df[f'{s}_base'] = df[f'{s}_base'].fillna(
                df['flight_regime'].map(global_regime_mean[s]))
            df[f'delta_{s}'] = df[s] - df[f'{s}_base']
            df.drop(columns=[f'{s}_base'], inplace=True)
        result.append(df)

    delta_features = [f'delta_{s}' for s in active_sensors]
    train_baseline = (result[0][result[0]['cycle'] <= 20]
                      .groupby(['engine_id', 'flight_regime'])[active_sensors]
                      .mean().reset_index())
    return result[0], result[1], delta_features, train_baseline


def compute_physics_features(df, active_sensors):
    df = df.copy().sort_values(['engine_id', 'cycle'])
    computed = []

    for feat, required in PHYSICS_REQUIREMENTS.items():
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
            df[feat] = (df['sensor_4'] * df['sensor_9']) / (df['sensor_12'] + 1e-5)

        elif feat == 'Phys_Cumulative_Thermal_Fatigue':
            df[feat] = (
                df.groupby('engine_id')['sensor_4']
                .transform(lambda x: x.expanding().mean())
            )

        elif feat == 'Phys_Flow_Capacity_Degradation':
            df[feat] = df['sensor_8'] / (df['sensor_15'] + 1e-5)

        computed.append(feat)

    return df, computed


def create_temporal_sequences(data, feature_cols, sequence_length=30,
                               target_col='Target_RUL', max_sequences=15):
    X_seqs, y_targets = [], []

    for eid in data['engine_id'].unique():
        eng = data[data['engine_id'] == eid]
        if len(eng) < sequence_length:
            continue

        feats   = eng[feature_cols].values
        targets = eng[target_col].values
        max_s   = len(eng) - sequence_length

        if max_s == 0:
            X_seqs.append(feats[:sequence_length])
            y_targets.append(targets[sequence_length - 1])
            continue

        n_dist  = max_sequences - 1
        starts  = (list(range(max_s)) if max_s <= n_dist else
                   [int(round(i * max_s / n_dist)) for i in range(n_dist)])
        starts.append(max_s)

        seen, unique = set(), []
        for s in starts:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        for s in unique:
            X_seqs.append(feats[s: s + sequence_length])
            y_targets.append(targets[s + sequence_length - 1])

    return np.array(X_seqs), np.array(y_targets)


def run_shap_engine_level(model, val_matrix, X_columns, X_tr_t,
                          active_sensors, dataset_id, device):
    """
    Engine-level SHAP: explains the last 10 sequences of one validation engine.
    Answers: which sensors are driving THIS engine's RUL decrease right now?
    Operational use — what a maintenance engineer needs for one aircraft.
    """
    try:
        sample_eid = int(val_matrix['engine_id'].iloc[0])
        eng_sl     = val_matrix[val_matrix['engine_id'] == sample_eid]
        seqs       = [eng_sl[X_columns].values[i: i + SEQUENCE_LENGTH]
                      for i in range(len(eng_sl) - SEQUENCE_LENGTH + 1)]
        if not seqs:
            raise RuntimeError('Not enough sequences for engine-level SHAP')

        bg_idx = np.random.choice(len(X_tr_t), size=100, replace=False)
        bg_t   = X_tr_t[bg_idx].to(device)
        smp_t  = torch.tensor(
            np.array(seqs[-min(10, len(seqs)):]), dtype=torch.float32
        ).to(device)

        model.eval()
        explainer = shap.GradientExplainer(model, bg_t)
        raw       = explainer.shap_values(smp_t)
        arr       = np.array(raw[0] if isinstance(raw, list) else raw)

        if arr.ndim == 4:
            arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr.mean(-1)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        vals_2d   = arr.mean(axis=1)
        mean_shap = np.abs(vals_2d).mean(axis=0)

        print(f'  Top 10 features by |SHAP| — engine {sample_eid}:')
        ranked = sorted(zip(X_columns, mean_shap),
                        key=lambda x: x[1], reverse=True)[:10]
        for feat, v in ranked:
            tag = '[PHY]' if feat.startswith('Phys_') \
                  else '[DEL]' if feat.startswith('delta') else '[SEN]'
            print(f'    {tag} {feat:<42} {v:.5f}')

        s_mask  = np.array([c in active_sensors for c in X_columns])
        s_vals  = vals_2d[:, s_mask]
        s_imp   = np.abs(s_vals).mean(axis=0)
        s_names = np.array(X_columns)[s_mask]
        top5    = np.argsort(s_imp)[::-1][:5]

        shap.summary_plot(s_vals[:, top5],
                          feature_names=[s_names[i] for i in top5],
                          plot_type='bar', show=False)
        plt.title(f'Top 5 Sensor SHAP — Engine {sample_eid} ({dataset_id})',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'cnn_lstm_shap_engine_{dataset_id}.png', dpi=300)
        plt.close()
        print(f'  ✅ Engine SHAP → cnn_lstm_shap_engine_{dataset_id}.png')

    except Exception as e:
        print(f'  ⚠️  Engine-level SHAP skipped ({dataset_id}): {e}')


def run_shap_fleet_level(model, val_matrix, X_columns, X_tr_t,
                         active_sensors, dataset_id, device):
    """
    Fleet-level global SHAP: averages sensor attributions across ALL val engines.
    Answers: which sensors CONSISTENTLY drive RUL degradation fleet-wide?
    Research/validation use — proves model learned thermodynamically meaningful
    patterns rather than engine-specific noise.
    Runtime: ~5–15 min CPU | ~1–2 min GPU. Run once, results saved.
    """
    try:
        bg_idx           = np.random.choice(len(X_tr_t), size=100, replace=False)
        bg_t             = X_tr_t[bg_idx].to(device)
        global_explainer = shap.GradientExplainer(model, bg_t)
        all_importance   = []
        processed        = 0

        for eng_id in val_matrix['engine_id'].unique():
            eng_sl = val_matrix[val_matrix['engine_id'] == eng_id]
            if len(eng_sl) < SEQUENCE_LENGTH:
                continue

            # Use only the last 5 sequences (failure zone — highest signal)
            seqs = [eng_sl[X_columns].values[i: i + SEQUENCE_LENGTH]
                    for i in range(len(eng_sl) - SEQUENCE_LENGTH + 1)]
            n = min(5, len(seqs))
            t = torch.tensor(np.array(seqs[-n:]),
                             dtype=torch.float32).to(device)
            model.eval()
            try:
                raw = global_explainer.shap_values(t)
            except Exception:
                continue

            arr = np.array(raw[0] if isinstance(raw, list) else raw)
            if arr.ndim == 4:
                arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr.mean(-1)
            if arr.ndim == 2:
                arr = arr[np.newaxis]

            per_feature = np.abs(arr.mean(axis=1)).mean(axis=0)
            s_mask      = np.array([c in active_sensors for c in X_columns])
            all_importance.append(per_feature[s_mask])
            processed += 1

            if processed % 20 == 0:
                print(f'  [Fleet SHAP] Processed {processed} engines...')

        if not all_importance:
            print(f'  ⚠️  No engines produced valid SHAP values ({dataset_id})')
            return

        global_imp = np.mean(all_importance, axis=0)
        global_std = np.std(all_importance,  axis=0)
        s_names    = np.array(X_columns)[s_mask]
        ranked     = sorted(zip(s_names, global_imp, global_std),
                            key=lambda x: x[1], reverse=True)

        print(f'\n  Fleet-level global sensor attribution '
              f'({processed} val engines):')
        for i, (s, m, sd) in enumerate(ranked, 1):
            print(f'    {i:2}. {s:<20} | {m:.5f} ± {sd:.5f}')

        fig_g, ax_g = plt.subplots(figsize=(10, 7))
        ax_g.barh([r[0] for r in ranked[::-1]], [r[1] for r in ranked[::-1]],
                  xerr=[r[2] for r in ranked[::-1]],
                  color='steelblue', edgecolor='black', linewidth=0.5,
                  error_kw=dict(ecolor='black', capsize=4, linewidth=1.2),
                  alpha=0.85)
        ax_g.set_xlabel('Mean |SHAP Value| across val engines', fontsize=11)
        ax_g.set_title(
            f'{dataset_id} Fleet-Level Global Sensor Attribution '
            f'— {processed} Engines\n'
            'Which sensors consistently drive RUL degradation fleet-wide?',
            fontsize=12, fontweight='bold'
        )
        ax_g.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax_g.text(0.98, 0.02, f'n = {processed} engines\n±1 std dev',
                  transform=ax_g.transAxes, fontsize=9, ha='right',
                  bbox=dict(boxstyle='round', facecolor='lightyellow',
                            alpha=0.8))
        fig_g.tight_layout()
        fig_g.savefig(f'cnn_lstm_shap_fleet_{dataset_id}.png', dpi=300)
        plt.close(fig_g)
        print(f'  ✅ Fleet SHAP → cnn_lstm_shap_fleet_{dataset_id}.png')

    except Exception as e:
        print(f'  ⚠️  Fleet-level SHAP skipped ({dataset_id}): {e}')


# ====================================================================
# MAIN TRAINING LOOP
# ====================================================================
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_summary = {}

index_labels     = ['engine_id', 'cycle']
setting_labels   = ['setting_1', 'setting_2', 'setting_3']
sensor_labels    = [f'sensor_{i}' for i in range(1, 22)]
complete_headers = index_labels + setting_labels + sensor_labels

print('=' * 80)
print('  CNN-LSTM PHYSICS-INFORMED v3 — MULTI-DATASET TRAINING (FD001–FD004)')
print('=' * 80)
print(f'  Device: {device}')

for dataset_id, config in DATASET_CONFIGS.items():

    train_file = f'train_{dataset_id}.txt'
    ablation_tag = (f"d{int(USE_DELTA_FEATURES)}"
                f"p{int(USE_PHYSICS_FEATURES)}"
                f"a{int(USE_ATTENTION)}"
                f"l{int(USE_PENALTY_LOSS)}")
    brain_path = f'cnn_lstm_brain_{dataset_id}_{ablation_tag}.pt'
    prep_path  = f'preprocessors_{dataset_id}_{ablation_tag}.pkl'
    n_regimes  = config['n_regimes']

    print(f'\n{"=" * 80}')
    print(f'  [{dataset_id}]  {config["description"]}')
    print(f'{"=" * 80}')

    if not os.path.exists(train_file):
        print(f'  ⏭  Skipping — {train_file} not found')
        continue

    # ── PHASE 1: Load & split ────────────────────────────────────────
    print(f'\n  [PHASE 1] Loading {train_file}...')
    raw = pd.read_csv(train_file, sep=r'\s+', names=complete_headers)
    print(f'  ✅ {raw["engine_id"].nunique()} engines, {len(raw):,} rows')

    # Split before dead sensor detection (FIX-T1)
    engines      = raw['engine_id'].unique()
    split_idx    = int(len(engines) * 0.8)
    train_engine_ids = engines[:split_idx]
    val_engine_ids   = engines[split_idx:]

    # FIX-T1: Dead sensor detection on TRAIN split only.
    # Previously detect_dead_sensors() was called on the full raw dataset
    # before the split — technically a minor leakage since val rows
    # influenced which sensors were kept. Now called on train rows only.
    train_raw_split = raw[raw['engine_id'].isin(train_engine_ids)]
    dead_sensors    = detect_dead_sensors(train_raw_split, sensor_labels)
    active_sensors  = [s for s in sensor_labels if s not in dead_sensors]
    cleaned         = raw.drop(columns=dead_sensors)

    print(f'  Dead sensors ({len(dead_sensors)}, detected on train only): '
          f'{dead_sensors}')
    print(f'  Active sensors: {len(active_sensors)}')

    max_cyc = cleaned.groupby('engine_id')['cycle'].max().reset_index()
    max_cyc.columns = ['engine_id', 'max_life']
    data_df = cleaned.merge(max_cyc, on='engine_id')
    data_df['Target_RUL'] = (
        data_df['max_life'] - data_df['cycle']
    ).clip(upper=RUL_MAX)
    data_df.drop(columns=['max_life'], inplace=True)

    train_matrix = data_df[data_df['engine_id'].isin(train_engine_ids)].copy()
    val_matrix   = data_df[data_df['engine_id'].isin(val_engine_ids)].copy()
    print(f'  Train engines: {train_matrix["engine_id"].nunique()} | '
          f'Val engines: {val_matrix["engine_id"].nunique()}')

    # ── PHASE 2: Flight regimes ──────────────────────────────────────
    print(f'\n  [PHASE 2] Assigning flight regimes (K={n_regimes})...')
    train_matrix, val_matrix, kmeans = assign_flight_regimes(
        train_matrix, val_matrix, setting_labels, n_regimes
    )
    km_label = f'KMeans K={n_regimes}' if kmeans else 'Single regime'
    print(f'  ✅ {km_label}')

    # ── PHASE 3: Delta features ──────────────────────────────────────
    print(f'\n  [PHASE 3] Computing engine-specific delta features...')
    global_regime_mean = train_matrix.groupby(
        'flight_regime'
    )[active_sensors].mean()
    train_matrix, val_matrix, delta_features, regime_baseline = \
        compute_delta_features(
            train_matrix, val_matrix, active_sensors, global_regime_mean
        )
    print(f'  ✅ {len(delta_features)} delta features')

    # ── PHASE 4: Physics features ────────────────────────────────────
    print(f'\n  [PHASE 4] Computing physics features...')
    if USE_PHYSICS_FEATURES:
     train_matrix, physics_features = compute_physics_features(
        train_matrix, active_sensors
      )
     val_matrix, _ = compute_physics_features(val_matrix, active_sensors)
    else:
     physics_features = []
    print(f'  ✅ {len(physics_features)} physics features:')
    for f in physics_features:
        print(f'       → {f}')

    # ── PHASE 5: Scaling ─────────────────────────────────────────────
    print(f'\n  [PHASE 5] Normalizing...')
    train_matrix[active_sensors] = train_matrix[active_sensors].astype(float)
    val_matrix[active_sensors]   = val_matrix[active_sensors].astype(float)

    regime_scalers_pool = {}
    for rid in range(n_regimes):
        m_tr = train_matrix['flight_regime'] == rid
        m_vl = val_matrix['flight_regime']   == rid
        if m_tr.any():
            sc = MinMaxScaler()
            train_matrix.loc[m_tr, active_sensors] = sc.fit_transform(
                train_matrix.loc[m_tr, active_sensors])
            regime_scalers_pool[rid] = sc
            if m_vl.any():
                val_matrix.loc[m_vl, active_sensors] = sc.transform(
                    val_matrix.loc[m_vl, active_sensors])

    # FIX-T2: physics_scaler fitted and saved only when physics_features
    # is non-empty. Previously an unfitted StandardScaler was always saved
    # to the pkl, which was misleading if physics_features=[].
    # Test pipeline guards with 'if physics_features and physics_scaler'
    # so this is fully backward-compatible.
    if physics_features:
        physics_scaler = StandardScaler()
        train_matrix[physics_features] = physics_scaler.fit_transform(
            train_matrix[physics_features])
        val_matrix[physics_features] = physics_scaler.transform(
            val_matrix[physics_features])
    else:
        physics_scaler = None   # explicitly None — no unfitted object saved

    if delta_features:
     delta_scaler = StandardScaler()
     train_matrix[delta_features] = delta_scaler.fit_transform(
        train_matrix[delta_features])
     val_matrix[delta_features] = delta_scaler.transform(
        val_matrix[delta_features])
    else:
     delta_scaler = None

    X_columns = active_sensors + physics_features + delta_features
    print(f'  ✅ {len(X_columns)} total features '
          f'({len(active_sensors)} sensors + {len(physics_features)} physics '
          f'+ {len(delta_features)} delta)')

    preprocessors = {
        'kmeans'              : kmeans,
        'regime_scalers_pool' : regime_scalers_pool,
        'physics_scaler'      : physics_scaler,   # None if no physics features
        'delta_scaler'        : delta_scaler,
        'global_regime_mean'  : global_regime_mean,
        'regime_baseline'     : regime_baseline,
        'active_sensors'      : active_sensors,
        'physics_features'    : physics_features,
        'delta_features'      : delta_features,
        'X_columns'           : X_columns,
        'setting_labels'      : setting_labels,
        'dead_sensors'        : dead_sensors,
        'RUL_MAX'             : RUL_MAX,
        'SEQUENCE_LENGTH'     : SEQUENCE_LENGTH,
        'T_REF'               : T_REF,
        'GAMMA_EXP'           : GAMMA_EXP,
        'n_regimes'           : n_regimes,
        'dataset_id'          : dataset_id,
    }
    joblib.dump(preprocessors, prep_path)
    print(f'  ✅ Preprocessors saved → {prep_path}')

    # ── PHASE 6: Sequence creation ───────────────────────────────────
    print(f'\n  [PHASE 6] Creating temporal sequences...')
    X_tr, y_tr = create_temporal_sequences(
        train_matrix, X_columns, SEQUENCE_LENGTH
    )
    X_vl, y_vl = create_temporal_sequences(
        val_matrix, X_columns, SEQUENCE_LENGTH
    )
    print(f'  ✅ Train: {len(X_tr):,} | Val: {len(X_vl):,}')

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    # ── PHASE 7: Train or load ───────────────────────────────────────
    # Fresh model instance per dataset — weights never shared between datasets
    model    = CNNLSTMSequential(input_size=len(X_columns)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  [PHASE 7] Model | {n_params:,} params | '
          f'input_size={len(X_columns)}')

    if os.path.exists(brain_path):
        print(f'  ✅ Found {brain_path} — loading saved weights, '
              f'skipping training')
        model.load_state_dict(
            torch.load(brain_path, map_location=device, weights_only=True))
        model.eval()
    else:
        print(f'  🚀 Training on {dataset_id}...')
        mse_fn = nn.MSELoss()
        pen_fn = AviationPenaltyLoss()
        opt    = torch.optim.Adam(model.parameters(), lr=0.001)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     opt, mode='min', factor=0.5, patience=2)

        best_loss    = float('inf')
        patience_cnt = 0
        tr_hist, vl_hist = [], []

        for epoch in range(TRAIN_EPOCHS):
            crit= mse_fn if (not USE_PENALTY_LOSS or epoch < WARMUP_EPOCHS) else pen_fn
            crit_name = 'MSE' if epoch < WARMUP_EPOCHS else 'Penalty'

            model.train()
            ep_loss = 0.0
            for bX, by in loader:
                bX, by = bX.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(model(bX), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * bX.size(0)
            tr_loss = ep_loss / len(loader.dataset)

            model.eval()
            with torch.no_grad():
                vl_loss = crit(model(X_vl_t.to(device)),
                               y_vl_t.to(device)).item()

            tr_hist.append(tr_loss)
            vl_hist.append(vl_loss)
            sched.step(vl_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'    Epoch [{epoch+1:02d}/{TRAIN_EPOCHS}] | '
                      f'{crit_name} | Train {tr_loss:.4f} | '
                      f'Val {vl_loss:.4f}')

            if vl_loss < best_loss:
                best_loss    = vl_loss
                patience_cnt = 0
                torch.save(model.state_dict(), brain_path)
            elif epoch >= WARMUP_EPOCHS:
                patience_cnt += 1
                if patience_cnt >= PATIENCE_LIMIT:
                    print(f'    [EARLY STOP] Best val loss: {best_loss:.4f}')
                    break

        model.load_state_dict(
            torch.load(brain_path, map_location=device, weights_only=True))
        model.eval()
        print(f'  ✅ Best brain saved → {brain_path}')

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(tr_hist, label='Train Loss', color='steelblue',  lw=2)
        ax.plot(vl_hist, label='Val Loss',   color='darkorange', lw=2)
        ax.axvline(WARMUP_EPOCHS, color='gray', ls=':', alpha=0.7,
                   label=f'Warmup end (ep {WARMUP_EPOCHS})')
        ax.set_title(f'CNN-LSTM v3 {dataset_id} — Training Loss',
                     fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, ls='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(f'cnn_lstm_loss_{dataset_id}.png', dpi=300)
        plt.close(fig)
        print(f'  ✅ Loss curve → cnn_lstm_loss_{dataset_id}.png')

    # ── PHASE 8: Validation metrics ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(X_vl_t.to(device)).cpu().numpy().flatten()

    rmse = float(np.sqrt(mean_squared_error(y_vl, preds)))
    mae  = float(mean_absolute_error(y_vl, preds))
    nasa = nasa_score(y_vl, preds)

    print(f'\n  {"─" * 50}')
    print(f'  VALIDATION RESULTS — {dataset_id}')
    print(f'  {"─" * 50}')
    print(f'  RMSE       : {rmse:.4f} cycles')
    print(f'  MAE        : {mae:.4f} cycles')
    print(f'  NASA Score : {nasa:.2f}  (lower is better)')
    print(f'  {"─" * 50}')

    training_summary[dataset_id] = {
        'rmse'          : rmse,
        'mae'           : mae,
        'nasa'          : nasa,
        'n_features'    : len(X_columns),
        'physics_count' : len(physics_features),
        'n_regimes'     : n_regimes,
    }

    # ── PHASE 9: Degradation velocity ────────────────────────────────
    print(f'\n  [PHASE 9] Degradation velocity (val engine with most cycles)...')
    try:
        val_cycle_counts = val_matrix.groupby('engine_id')['cycle'].max()
        best_eng = val_cycle_counts.idxmax()
        eng_sl   = val_matrix[
            val_matrix['engine_id'] == best_eng
        ].sort_values('cycle')

        seqs = [eng_sl[X_columns].values[i: i + SEQUENCE_LENGTH]
                for i in range(len(eng_sl) - SEQUENCE_LENGTH + 1)]
        if not seqs:
            raise RuntimeError('Not enough cycles for trajectory')

        seq_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        with torch.no_grad():
            traj_preds = model(seq_t).cpu().numpy().flatten()

        smoothed = pd.Series(traj_preds).rolling(
            window=5, min_periods=1
        ).mean().values
        velocity = np.insert(np.diff(smoothed), 0, 0.0)

        print(f'  --- Final 5 cycles of engine {best_eng} (val) ---')
        for idx in range(max(0, len(traj_preds) - 5), len(traj_preds)):
            cyc  = SEQUENCE_LENGTH + idx
            pred = traj_preds[idx]
            vel  = velocity[idx]
            if vel < -1.2:
                status = '🚨 ACCELERATING FAILURE'
            elif vel > -0.2:
                status = '🟢 STABLE (CAPPED BOUND)'
            else:
                status = '🟡 STANDARD LINEAR WEAR'
            print(f'    Cycle {cyc:<4} | RUL: {pred:<6.2f} | '
                  f'd(RUL)/dt: {vel:<6.3f} | {status}')

        fig, ax = plt.subplots(figsize=(11, 5))
        x_axis = range(SEQUENCE_LENGTH, len(eng_sl) + 1)
        ax.plot(x_axis, traj_preds, label='Raw Predicted RUL',
                color='darkorange', alpha=0.4, linestyle='--')
        ax.plot(x_axis, smoothed, label='Smoothed Health Trend',
                color='red', linewidth=3)
        ax.set_title(
            f'CNN-LSTM v3 {dataset_id} — Val Engine {best_eng} '
            f'Health Trajectory',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Operational Flight Cycles')
        ax.set_ylabel('Remaining Useful Life (cycles)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'deg_velocity_val_{dataset_id}.png', dpi=300)
        plt.close(fig)
        print(f'  ✅ Velocity chart → deg_velocity_val_{dataset_id}.png')

    except Exception as e:
        print(f'  ⚠️  Degradation velocity skipped ({dataset_id}): {e}')

    # ── PHASE 10: Engine-level SHAP ───────────────────────────────────
    # Operational view: which sensors drive THIS engine's degradation NOW?
    print(f'\n  [PHASE 10] Engine-level SHAP ({dataset_id})...')
    run_shap_engine_level(
        model, val_matrix, X_columns, X_tr_t,
        active_sensors, dataset_id, device
    )

    # ── PHASE 11: Fleet-level global SHAP ────────────────────────────
    # Research view: which sensors consistently drive degradation fleet-wide?
    # Runtime: ~5–15 min CPU | ~1–2 min GPU per dataset.
    print(f'\n  [PHASE 11] Fleet-level global SHAP ({dataset_id})...')
    run_shap_fleet_level(
        model, val_matrix, X_columns, X_tr_t,
        active_sensors, dataset_id, device
    )


# ====================================================================
# FINAL TRAINING SUMMARY TABLE
# ====================================================================
print(f'\n\n{"=" * 75}')
print(f'  MULTI-DATASET TRAINING SUMMARY — CNN-LSTM v3 ENHANCED PHYSICS')
print(f'{"=" * 75}')
print(f'  {"Dataset":<8} {"RMSE":>8} {"MAE":>8} {"NASA":>12} '
      f'{"Features":>10} {"Physics":>8} {"K":>4}')
print(f'  {"─" * 8} {"─" * 8} {"─" * 8} {"─" * 12} '
      f'{"─" * 10} {"─" * 8} {"─" * 4}')

for ds, m in training_summary.items():
    print(f'  {ds:<8} {m["rmse"]:>8.4f} {m["mae"]:>8.4f} '
          f'{m["nasa"]:>12.2f} {m["n_features"]:>10} '
          f'{m["physics_count"]:>8} {m["n_regimes"]:>4}')

print(f'{"=" * 75}')
print(f'\n  Output files per dataset:')
print(f'    cnn_lstm_brain_FDxxx.pt              ← model weights')
print(f'    preprocessors_FDxxx.pkl              ← scaler/kmeans/feature config')
print(f'    cnn_lstm_loss_FDxxx.png              ← training loss curve')
print(f'    deg_velocity_val_FDxxx.png           ← val engine health trajectory')
print(f'    cnn_lstm_shap_engine_FDxxx.png       ← engine-level SHAP')
print(f'    cnn_lstm_shap_fleet_FDxxx.png        ← fleet-level global SHAP')
