import pandas as pd
import numpy as np


# SPLIT FEATURES / TARGET
def split_features_target(
    df: pd.DataFrame,
    target_column: str = "sales"

) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    """

    X = df[
        [
            'price',
            'marketing',
            'month_sin',
            'month_cos',
            'promo'
        ]
    ]

    y = df[target_column]

    return X, y


# SHUFFLE DATA
def shuffle_data(
    X: pd.DataFrame,
    y: pd.Series

) -> tuple[pd.DataFrame, pd.Series]:
    """
    Shuffle dataset randomly.
    """

    idx = np.random.permutation(len(X))

    return X.iloc[idx], y.iloc[idx]


# TRAIN TEST SPLIT
def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = 0.8,
    shuffle: bool = True

) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series
]:
    """
    Split dataset into train and test sets.
    """

    if shuffle:
        X, y = shuffle_data(X, y)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    split = int(len(X) * ratio)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test


# Z-SCORE NORMALIZATION
def z_score_normalization(
    X: pd.DataFrame,
    y: pd.Series | None = None

):
    """
    Normalize numerical features using z-score normalization.
    """

    X_norm = X.copy()

    cols_to_normalize = [
        'price',
        'marketing',
        'month_sin',
        'month_cos'
    ]

    mean_x = X_norm[cols_to_normalize].mean()
    std_x = X_norm[cols_to_normalize].std()

    std_x = std_x.replace(0, 1)

    X_norm[cols_to_normalize] = (
        X_norm[cols_to_normalize] - mean_x
    ) / std_x

    if y is not None:

        mean_y = y.mean()
        std_y = y.std()

        std_y = std_y if std_y != 0 else 1

        y_norm = (y - mean_y) / std_y

        return (
            X_norm,
            y_norm,
            mean_x,
            std_x,
            mean_y,
            std_y
        )

    return X_norm, mean_x, std_x