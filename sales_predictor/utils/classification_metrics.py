import numpy as np

def accuracy_score( y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Accuracy Score.

    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.

    Returns:
    float
        Accuracy score.
    """
    return np.mean(y_true == y_pred)

def precision_score( y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Precision Score.

    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.

    Returns:
    float
        Precision score.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall_score( y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Recall Score.

    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.

    Returns:
    float
        Recall score.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score( y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 Score.

    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.

    Returns:
    float
        F1 score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if (precision + recall) == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)