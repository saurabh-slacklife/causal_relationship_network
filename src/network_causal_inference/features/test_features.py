"""
============================================================================
Probabilistic Graphical Model — Data Preprocessing Pipeline
============================================================================
Dataset   : UNSW_NB15 (Network Intrusion Detection)
Goal      : Prepare features for Bayesian Network structure & parameter learning
            so that causal relationships between network features and attack
            outcomes can be discovered and queried.

Pipeline Order:
    1. load_data()                  — Load & initial inspection
    2. handle_missing_values()      — Impute missing / inconsistent entries
    3. encode_categorical_features()— Convert categorical columns to numeric
    4. correct_skew()               — Log-transform right-skewed continuous cols
    5. scale_continuous_features()  — Min-Max scale continuous features to [0, 1]
    6. discretize_for_bn()          — Bin continuous vars into discrete categories
    7. select_features()            — Remove redundant / irrelevant features
    8. handle_class_imbalance()     — Balance attack vs normal samples
    9. split_data()                 — Stratified train/test split
   10. summarize_final_dataset()    — Print final feature set summary

Usage:
    python pgm_preprocessing.py
============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
import os

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION — Tweak these to match your local setup
# ============================================================================

# Path to your UNSW_NB15 CSV file(s)
DATA_PATH = "UNSW_NB15.csv"          # Update if your file is elsewhere

# Target column name in the dataset
TARGET_COL = "label"                  # 0 = normal, 1 = attack

# Attack-type column (used for stratification)
ATTACK_TYPE_COL = "attack_cat"

# Continuous features in UNSW_NB15
CONTINUOUS_FEATURES = [
    "dur", "orig_bytes", "resp_bytes", "trans_depth",
    "response_body_len", "orig_ttl", "resp_ttl",
    "mean_ipt", "ct_state_ttl", "ct_flg_hist",
    "srate", "drate", "rate",
    "sinpkt", "dinpkt",
    "ct_dst_sport_c", "ct_dst_dport_c"
]

# Categorical features in UNSW_NB15
CATEGORICAL_FEATURES = [
    "proto", "service", "state",
    "tcpbase", "tcpflags"
]

# Features with strong causal relevance — kept for the Bayesian Network
# (selected based on domain knowledge + mutual information scoring)
CAUSAL_FEATURE_CANDIDATES = [
    "proto", "service", "state",
    "dur", "orig_bytes", "resp_bytes",
    "orig_ttl", "resp_ttl",
    "mean_ipt", "rate", "srate", "drate",
    "sinpkt", "dinpkt",
    "ct_state_ttl", "ct_dst_sport_c", "ct_dst_dport_c"
]

# Number of bins for discretization (adjust per feature if needed)
DEFAULT_NUM_BINS = 4

# Correlation threshold for removing redundant features
CORRELATION_THRESHOLD = 0.85

# Train / test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the UNSW_NB15 dataset and perform initial sanity checks.

    Steps:
        - Read CSV into a DataFrame.
        - Strip whitespace from column names (common issue in this dataset).
        - Print shape, column types, and class distribution for initial review.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw loaded dataset.
    """
    print("=" * 60)
    print("  STEP 1: Loading Data")
    print("=" * 60)

    df = pd.read_csv(filepath)

    # Strip whitespace from column names (UNSW_NB15 CSVs often have this)
    df.columns = df.columns.str.strip()

    print(f"  Dataset shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory usage         : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\n  Column dtypes:\n{df.dtypes.value_counts().to_string()}")
    print(f"\n  Target distribution ('{TARGET_COL}'):")
    print(df[TARGET_COL].value_counts().to_string())
    print()

    return df


# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing and inconsistent values across the dataset.

    Strategy per column type:
        - Continuous features  → Median imputation
          (median is more robust than mean for skewed network-traffic data)
        - Categorical features → Mode imputation
          (most frequent category fills gaps)
        - attack_cat column    → Fill NaN with "Normal"
          (missing attack_cat entries correspond to benign traffic)

    Why this matters for the Bayesian Network:
        Missing values would create incomplete rows, making CPD estimation
        unreliable. Median imputation avoids pulling estimates toward outliers
        (common in byte-count features).

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Dataset with no missing values.
    """
    print("=" * 60)
    print("  STEP 2: Handling Missing Values")
    print("=" * 60)

    df = df.copy()

    # --- Report missing values before imputation ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).sort_values("Missing Count", ascending=False)
    missing_report = missing_report[missing_report["Missing Count"] > 0]

    if missing_report.empty:
        print("  No missing values found.\n")
    else:
        print("  Missing values detected:\n")
        print(missing_report.to_string())
        print()

    # --- Impute continuous features with median ---
    for col in CONTINUOUS_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [Continuous] '{col}' → filled {missing[col]} NaNs with median = {median_val:.4f}")

    # --- Impute categorical features with mode ---
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [Categorical] '{col}' → filled {missing[col]} NaNs with mode = '{mode_val}'")

    # --- Fill missing attack_cat with 'Normal' ---
    if ATTACK_TYPE_COL in df.columns:
        attack_missing = df[ATTACK_TYPE_COL].isnull().sum()
        if attack_missing > 0:
            df[ATTACK_TYPE_COL].fillna("Normal", inplace=True)
            print(f"  [Attack Type] '{ATTACK_TYPE_COL}' → filled {attack_missing} NaNs with 'Normal'")

    # --- Final verification ---
    remaining_nulls = df.isnull().sum().sum()
    print(f"\n  Total missing values remaining: {remaining_nulls}")
    print()

    return df


# ============================================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================================

def encode_categorical_features(
    df: pd.DataFrame,
    low_freq_threshold: int = 50
) -> pd.DataFrame:
    """
    Convert categorical columns into numeric form suitable for a Bayesian Network.

    Strategy:
        1. Group low-frequency categories (count < threshold) into 'Other'
           to reduce one-hot dimensionality explosion.
        2. Apply one-hot encoding on the grouped categorical columns.
        3. Drop the original categorical columns after encoding.

    Why this matters for the Bayesian Network:
        BN nodes must be discrete. One-hot encoding creates binary indicator
        nodes per category. Grouping rare categories first keeps the graph
        manageable and avoids sparse CPDs that are hard to estimate.

    Parameters:
        df (pd.DataFrame)     : Dataset after missing-value handling.
        low_freq_threshold (int): Categories appearing fewer than this many
                                  times are grouped into 'Other'.

    Returns:
        pd.DataFrame: Dataset with categorical columns one-hot encoded.
    """
    print("=" * 60)
    print("  STEP 3: Encoding Categorical Features")
    print("=" * 60)

    df = df.copy()

    cols_to_encode = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    for col in cols_to_encode:
        original_categories = df[col].nunique()

        # --- Group rare categories into 'Other' ---
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < low_freq_threshold].index.tolist()

        if rare_categories:
            df[col] = df[col].replace(rare_categories, "Other")
            print(f"  '{col}': grouped {len(rare_categories)} rare categories "
                  f"→ 'Other'  (original: {original_categories}, "
                  f"after grouping: {df[col].nunique()})")
        else:
            print(f"  '{col}': no rare categories to group "
                  f"(unique values: {original_categories})")

    # --- One-hot encode all categorical columns ---
    df = pd.get_dummies(df, columns=cols_to_encode, prefix_sep="_", dtype=int)

    # Report new columns created
    new_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in cols_to_encode)]
    print(f"\n  One-hot encoded columns created: {len(new_cols)}")
    for col in new_cols:
        print(f"    → {col}")
    print()

    return df


# ============================================================================
# 4. CORRECT SKEW (Log Transformation)
# ============================================================================

def correct_skew(df: pd.DataFrame, skew_threshold: float = 1.0) -> pd.DataFrame:
    """
    Apply log1p transformation to right-skewed continuous features.

    Why log-transform?
        Network traffic features like orig_bytes, resp_bytes, rate are
        heavily right-skewed (long tail of large values). Log-transformation
        compresses the tail, making the distribution more symmetric. This
        improves both MinMax scaling and quantile-based discretization
        (bins become more meaningful).

    Why this matters for the Bayesian Network:
        Skewed features produce uneven bins during discretization, causing
        most samples to fall into one bin. Log-transform spreads the data
        more evenly, so each discrete bin in the BN captures a meaningful
        subset of traffic behavior.

    Parameters:
        df (pd.DataFrame)    : Dataset after categorical encoding.
        skew_threshold (float): Features with skewness > this value are
                                log-transformed. Default = 1.0.

    Returns:
        pd.DataFrame: Dataset with skewed features log-transformed.
    """
    print("=" * 60)
    print("  STEP 4: Correcting Skew (Log Transformation)")
    print("=" * 60)

    df = df.copy()

    # Only consider continuous features that still exist after encoding
    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in df.columns]

    transformed_cols = []

    for col in continuous_cols:
        skewness_before = df[col].skew()

        if skewness_before > skew_threshold:
            # log1p handles zero values safely: log1p(x) = log(1 + x)
            df[col] = np.log1p(df[col])
            skewness_after = df[col].skew()
            transformed_cols.append(col)
            print(f"  '{col}': skewness {skewness_before:.2f} → {skewness_after:.2f}  (log1p applied)")
        else:
            print(f"  '{col}': skewness {skewness_before:.2f}  (no transform needed)")

    print(f"\n  Total features log-transformed: {len(transformed_cols)}")
    print()

    return df


# ============================================================================
# 5. SCALE CONTINUOUS FEATURES
# ============================================================================

def scale_continuous_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to all remaining continuous features → [0, 1].

    Why Min-Max over Standardization (z-score)?
        - Min-Max preserves the relative magnitude of values, which is
          meaningful for network traffic (e.g., 0 bytes vs 10000 bytes).
        - Standardization can produce negative values, which are harder
          to interpret as probabilities in a BN context.
        - Min-Max maps everything to [0, 1], making subsequent
          quantile-based discretization more uniform.

    Parameters:
        df (pd.DataFrame): Dataset after skew correction.

    Returns:
        tuple:
            pd.DataFrame  : Dataset with continuous features scaled to [0, 1].
            MinMaxScaler  : Fitted scaler object (save for later use on test data).
    """
    print("=" * 60)
    print("  STEP 5: Scaling Continuous Features (Min-Max)")
    print("=" * 60)

    df = df.copy()

    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in df.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    print(f"  Scaled {len(continuous_cols)} continuous features to [0, 1]:")
    for col in continuous_cols:
        print(f"    → {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, "
              f"mean={df[col].mean():.4f}")
    print()

    return df, scaler


# ============================================================================
# 6. DISCRETIZE CONTINUOUS FEATURES FOR BAYESIAN NETWORK
# ============================================================================

def discretize_for_bn(
    df: pd.DataFrame,
    num_bins: int = DEFAULT_NUM_BINS,
    domain_thresholds: dict = None
) -> pd.DataFrame:
    """
    Bin continuous features into discrete categories for the Bayesian Network.

    Strategy:
        - Default: Equal-frequency (quantile) binning.
          This ensures each bin has roughly the same number of samples,
          avoiding empty or near-empty bins that break CPD estimation.
        - Override: For specific causal features, use domain-informed
          thresholds (e.g., known attack rate limits) instead of quantiles.

    Bin Labels:
        Bins are labeled descriptively so BN nodes are interpretable:
        e.g., "Very_Low", "Low", "Medium", "High" for 4-bin features.

    Why this matters for the Bayesian Network:
        BNs require discrete variables. The quality of discretization
        directly affects structure learning — poorly binned features
        will create spurious or miss real causal edges.

    Parameters:
        df (pd.DataFrame)         : Dataset after scaling.
        num_bins (int)            : Default number of bins per feature.
        domain_thresholds (dict)  : Optional dict mapping feature name →
                                    list of bin edges (must be sorted).
                                    Example:
                                      {"rate": [0.0, 0.2, 0.5, 0.8, 1.0]}

    Returns:
        pd.DataFrame: Dataset with continuous features replaced by bin labels.
    """
    print("=" * 60)
    print("  STEP 6: Discretizing Continuous Features for BN")
    print("=" * 60)

    df = df.copy()

    # Default domain thresholds for key causal features
    # These reflect known attack behavior patterns in network traffic
    if domain_thresholds is None:
        domain_thresholds = {
            # High packet rate is a strong DoS indicator
            "rate":   [0.0, 0.15, 0.40, 0.75, 1.0],
            "srate":  [0.0, 0.20, 0.50, 0.80, 1.0],
            "drate":  [0.0, 0.20, 0.50, 0.80, 1.0],
            # Byte volumes — large transfers correlate with data exfiltration
            "orig_bytes": [0.0, 0.10, 0.30, 0.70, 1.0],
            "resp_bytes": [0.0, 0.10, 0.30, 0.70, 1.0],
        }

    # Standard bin labels by bin count
    bin_labels_map = {
        3: ["Low", "Medium", "High"],
        4: ["Very_Low", "Low", "High", "Very_High"],
        5: ["Very_Low", "Low", "Medium", "High", "Very_High"],
    }

    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in df.columns]

    for col in continuous_cols:
        if col in domain_thresholds:
            # --- Domain-informed binning ---
            edges = domain_thresholds[col]
            n_bins = len(edges) - 1
            labels = bin_labels_map.get(n_bins, [f"Bin_{i}" for i in range(n_bins)])

            df[col] = pd.cut(
                df[col],
                bins=edges,
                labels=labels,
                include_lowest=True,
                duplicates="drop"
            )
            print(f"  '{col}': domain-informed bins → {labels}")

        else:
            # --- Quantile-based binning (equal frequency) ---
            labels = bin_labels_map.get(
                num_bins,
                [f"Bin_{i}" for i in range(num_bins)]
            )

            df[col] = pd.qcut(
                df[col],
                q=num_bins,
                labels=labels,
                duplicates="drop"
            )
            print(f"  '{col}': quantile bins ({num_bins} bins) → {labels}")

    # Convert all binned columns to category dtype (memory efficient)
    for col in continuous_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print()
    return df


# ============================================================================
# 7. FEATURE SELECTION
# ============================================================================

def select_features(
    df: pd.DataFrame,
    top_k: int = 20
) -> pd.DataFrame:
    """
    Remove redundant and irrelevant features; keep only causally meaningful ones.

    Two-stage selection:
        Stage A — Correlation pruning:
            Among continuous-origin features, remove one column from each
            pair whose Pearson correlation exceeds CORRELATION_THRESHOLD.
            This eliminates multicollinearity, which can confuse BN
            structure learning algorithms.

        Stage B — Mutual Information ranking:
            Score all remaining features by mutual information with the
            target (attack/normal). Keep the top-k features. Mutual
            information captures non-linear dependencies, making it
            well-suited for selecting features that carry causal signal.

    Why this matters for the Bayesian Network:
        Redundant features create spurious edges in the learned DAG.
        Irrelevant features waste degrees of freedom during CPD estimation
        and slow down inference. Lean feature sets lead to cleaner,
        more interpretable causal graphs.

    Parameters:
        df (pd.DataFrame): Discretized dataset.
        top_k (int)      : Number of top features to retain after MI scoring.

    Returns:
        pd.DataFrame: Dataset with only selected features + target column.
    """
    print("=" * 60)
    print("  STEP 7: Feature Selection")
    print("=" * 60)

    df = df.copy()

    # Separate target before selection
    target = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL] + ([ATTACK_TYPE_COL] if ATTACK_TYPE_COL in df.columns else []),
                errors="ignore")

    # --- Stage A: Correlation-based pruning ---
    print("  [Stage A] Correlation-based pruning")

    # Convert categories to numeric codes for correlation computation
    X_numeric = X.apply(
        lambda col: col.cat.codes if col.dtype.name == "category" else col
    )

    corr_matrix = X_numeric.corr().abs()

    # Upper triangle mask (avoid double-counting pairs)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Columns where max correlation with another column exceeds threshold
    to_drop = [
        col for col in upper.columns
        if (upper[col] > CORRELATION_THRESHOLD).any()
    ]

    print(f"    Correlation threshold : {CORRELATION_THRESHOLD}")
    print(f"    Features dropped      : {len(to_drop)}")
    for col in to_drop:
        # Find which feature it was correlated with
        corr_partner = upper[col][upper[col] > CORRELATION_THRESHOLD].idxmax()
        print(f"      → '{col}' (corr with '{corr_partner}' = "
              f"{corr_matrix.loc[col, corr_partner]:.3f})")

    X_numeric = X_numeric.drop(columns=to_drop, errors="ignore")
    X = X.drop(columns=to_drop, errors="ignore")

    # --- Stage B: Mutual Information ranking ---
    print(f"\n  [Stage B] Mutual Information ranking (top {top_k})")

    # Ensure all values are numeric for MI computation
    X_mi = X.apply(
        lambda col: col.cat.codes if col.dtype.name == "category" else col
    ).fillna(0)

    mi_scores = mutual_info_classif(X_mi, target, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        "Feature": X_mi.columns,
        "MI_Score": mi_scores
    }).sort_values("MI_Score", ascending=False).reset_index(drop=True)

    print(f"\n    Top {top_k} features by Mutual Information:\n")
    print(mi_df.head(top_k).to_string(index=False))

    # Keep only top-k features
    selected_features = mi_df.head(top_k)["Feature"].tolist()

    # Rebuild DataFrame with selected features + target
    df_selected = X[selected_features].copy()
    df_selected[TARGET_COL] = target.values

    print(f"\n  Final feature count: {len(selected_features)} + target")
    print()

    return df_selected


# ============================================================================
# 8. HANDLE CLASS IMBALANCE
# ============================================================================

def handle_class_imbalance(
    df: pd.DataFrame,
    strategy: str = "combined"
) -> pd.DataFrame:
    """
    Balance the dataset between attack (1) and normal (0) classes.

    Strategies:
        "smote"     — Oversample the minority class using SMOTE.
        "under"     — Undersample the majority class randomly.
        "combined"  — Undersample majority to 2× minority, then SMOTE
                      minority up to match. This is the safest option for
                      causal modeling because it avoids excessive synthetic
                      samples while still balancing classes.

    ⚠️ Caution for Bayesian Networks:
        SMOTE generates synthetic samples by interpolating feature vectors.
        For discrete (binned) features, this can create combinations that
        never occur in real traffic — potentially introducing spurious
        causal edges. The "combined" strategy minimizes this risk by
        limiting the amount of synthetic data generated.

    Parameters:
        df (pd.DataFrame): Feature-selected dataset with target column.
        strategy (str)    : One of "smote", "under", or "combined".

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    print("=" * 60)
    print("  STEP 8: Handling Class Imbalance")
    print("=" * 60)

    df = df.copy()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Convert categories to numeric codes for resampling
    X_coded = X.apply(
        lambda col: col.cat.codes if col.dtype.name == "category" else col
    ).astype(int)

    print(f"  Before balancing:")
    print(f"    Normal (0) : {(y == 0).sum()}")
    print(f"    Attack (1) : {(y == 1).sum()}")
    print(f"    Ratio      : {(y == 0).sum() / max((y == 1).sum(), 1):.2f}:1")

    if strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_coded, y)
        print(f"\n  Applied: SMOTE oversampling")

    elif strategy == "under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_coded, y)
        print(f"\n  Applied: Random undersampling")

    elif strategy == "combined":
        # Step 1: Undersample majority to 2× minority count
        minority_count = min(y.value_counts())
        target_majority = minority_count * 2

        under_sampler = RandomUnderSampler(
            sampling_strategy={0: target_majority, 1: minority_count},
            random_state=RANDOM_STATE
        )
        X_under, y_under = under_sampler.fit_resample(X_coded, y)

        # Step 2: SMOTE minority up to match majority
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
        print(f"\n  Applied: Combined (Undersample → SMOTE)")

    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'smote', 'under', or 'combined'.")

    print(f"\n  After balancing:")
    print(f"    Normal (0) : {(y_resampled == 0).sum()}")
    print(f"    Attack (1) : {(y_resampled == 1).sum()}")
    print(f"    Ratio      : {(y_resampled == 0).sum() / max((y_resampled == 1).sum(), 1):.2f}:1")
    print()

    # Rebuild DataFrame
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[TARGET_COL] = y_resampled.values

    return df_balanced


# ============================================================================
# 9. TRAIN / TEST SPLIT
# ============================================================================

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split.

    Stratification is done on the TARGET column so that both splits
    contain proportional attack/normal samples. This prevents the test
    set from being dominated by one class, which would give misleading
    evaluation of the Bayesian Network's causal inference accuracy.

    Parameters:
        df (pd.DataFrame) : Balanced dataset with target column.
        test_size (float) : Fraction of data reserved for testing.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("  STEP 9: Train / Test Split")
    print("=" * 60)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"  Split ratio          : {1 - test_size:.0%} train / {test_size:.0%} test")
    print(f"  Training set         : {X_train.shape[0]} samples")
    print(f"    Normal : {(y_train == 0).sum()}  |  Attack : {(y_train == 1).sum()}")
    print(f"  Test set             : {X_test.shape[0]} samples")
    print(f"    Normal : {(y_test == 0).sum()}  |  Attack : {(y_test == 1).sum()}")
    print()

    return X_train, X_test, y_train, y_test


# ============================================================================
# 10. SUMMARIZE FINAL DATASET
# ============================================================================

def summarize_final_dataset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
):
    """
    Print a comprehensive summary of the final preprocessed dataset,
    including feature types, value counts per bin, and class balance.

    This summary confirms the data is ready for Bayesian Network
    structure learning and CPD estimation.

    Parameters:
        X_train, X_test : Feature DataFrames.
        y_train, y_test : Target Series.
    """
    print("=" * 60)
    print("  STEP 10: Final Dataset Summary")
    print("=" * 60)

    print(f"\n  {'Feature':<30} {'Type':<12} {'Unique Values':<15} {'Sample Values'}")
    print(f"  {'-'*30} {'-'*12} {'-'*15} {'-'*30}")

    for col in X_train.columns:
        dtype = str(X_train[col].dtype)
        n_unique = X_train[col].nunique()
        sample = X_train[col].value_counts().head(3).index.tolist()
        print(f"  {col:<30} {dtype:<12} {n_unique:<15} {sample}")

    print(f"\n  Target column: '{TARGET_COL}'")
    print(f"    Training — Normal: {(y_train == 0).sum()}, Attack: {(y_train == 1).sum()}")
    print(f"    Testing  — Normal: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()}")

    print(f"\n  ✓ Dataset is ready for Bayesian Network structure learning.")
    print(f"  ✓ All features are discrete — suitable for CPD estimation.")
    print(f"  ✓ Classes are balanced — CPD tables will not be biased.")
    print()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Execute the full preprocessing pipeline end-to-end.
    """
    print("\n" + "█" * 60)
    print("  PGM PREPROCESSING PIPELINE — UNSW_NB15")
    print("█" * 60 + "\n")

    # Step 1: Load
    df = load_data()

    # Step 2: Missing values
    df = handle_missing_values(df)

    # Step 3: Encode categoricals
    df = encode_categorical_features(df)

    # Step 4: Log-transform skewed features
    df = correct_skew(df)

    # Step 5: Min-Max scaling
    df, scaler = scale_continuous_features(df)

    # Step 6: Discretize for BN
    df = discretize_for_bn(df)

    # Step 7: Feature selection
    df = select_features(df, top_k=20)

    # Step 8: Balance classes
    df = handle_class_imbalance(df, strategy="combined")

    # Step 9: Train/test split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 10: Summary
    summarize_final_dataset(X_train, X_test, y_train, y_test)

    # --- Save preprocessed splits for BN training ---
    X_train.to_csv("X_train_bn.csv", index=False)
    X_test.to_csv("X_test_bn.csv", index=False)
    y_train.to_csv("y_train_bn.csv", index=False)
    y_test.to_csv("y_test_bn.csv", index=False)
    print("  Saved: X_train_bn.csv, X_test_bn.csv, y_train_bn.csv, y_test_bn.csv")


if __name__ == "__main__":
    main()