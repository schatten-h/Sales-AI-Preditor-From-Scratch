import numpy as np
def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters:
    ----------
    y_true : np.ndarray
        Real target values.

    y_pred : np.ndarray
        Predicted values.

    Returns:
    -------
    float
        Mean Squared Error.
    """

    return ((y_true - y_pred) ** 2).mean()

def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters:
    ----------
    y_true : np.ndarray
        Real target values.

    y_pred : np.ndarray
        Predicted values.

    Returns:
    -------
    float
        Root Mean Squared Error.
    """
    return mean_squared_error(y_true, y_pred) ** 0.5

def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute R^2 Score.

    Parameters:
    ----------
    y_true : np.ndarray
        Real target values.

    y_pred : np.ndarray
        Predicted values.

    Returns:
    -------
    float
        R^2 Score.
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)