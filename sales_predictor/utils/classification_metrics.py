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

def confusion_matrix(y_true: np.ndarray,
                     y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.                               
    Returns:
        [[TN FP]
         [FN TP]]
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def specificity_score(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Compute Specificity Score.
    Args:
        y_true : np.ndarray   The real target values.
        y_pred : np.ndarray The predicted target values.    
    Returns:
    float
        Specificity score.
    """

    cm = confusion_matrix(y_true, y_pred)

    tn = cm[0,0]
    fp = cm[0,1]

    if (tn + fp) == 0:
        return 0.0

    return tn / (tn + fp)

def false_positive_rate(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    """ Compute False Positive Rate.
    Args:
      y_true : np.ndarray   The real target values.
    y_pred : np.ndarray The predicted target values.    
    Returns:
    float
        False Positive Rate.
    """
    cm = confusion_matrix(y_true, y_pred)

    tn = cm[0,0]
    fp = cm[0,1]

    if (fp + tn) == 0:
        return 0.0

    return fp / (fp + tn)

