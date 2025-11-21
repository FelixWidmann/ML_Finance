import numpy as np
import pandas as pd

def purged_kfold_indices_by_date(df, date_col='Date', n_splits=5, purge_days=5):
    # Ensure sorted and datetime
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    n_samples = len(df)
    pos_idx = np.arange(n_samples)     
    dates = df[date_col].to_numpy()         

    # Unique dates in order
    unique_dates = np.unique(dates)
    date_folds = np.array_split(unique_dates, n_splits)

    folds = []
    for fold_dates in date_folds:
        if len(fold_dates) == 0:
            continue

        # --- test: positions whose date is in this fold ---
        test_mask = np.isin(dates, fold_dates)
        test_idx = pos_idx[test_mask]

        # --- purge window in calendar days ---
        min_test_date = fold_dates[0]
        max_test_date = fold_dates[-1]

        purge_start = min_test_date - pd.Timedelta(days=purge_days)
        purge_end   = max_test_date + pd.Timedelta(days=purge_days)

        purge_mask = (dates >= purge_start) & (dates <= purge_end)
        purge_idx = pos_idx[purge_mask]

        # --- train: everything except test+purge ---
        train_idx = np.setdiff1d(pos_idx, np.union1d(test_idx, purge_idx))

        folds.append((train_idx, test_idx))

    return folds
