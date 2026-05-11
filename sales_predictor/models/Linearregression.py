import numpy as np
def cost_function(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lamb: float = 0.7) -> float:
    """ Compute cost for linear regression with L2 regularization.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
        lamb (float): The regularization parameter.
    Returns:
        float: The computed cost."""
    m = x.shape[0]
    reg_sum = 0

    f_wb  = x @ w + b
    cost_sum = np.sum((f_wb - y)**2)/(2*m)
   
    
    reg_sum = np.sum(w**2)
    reg_sum *= lamb/(2*m) 

    total_cost = cost_sum + reg_sum

    return total_cost

def compute_gradient(x : np.ndarray, y : np.ndarray, w : np.ndarray, b : float, lamb : float = 0.7):
    """ Compute the gradient for linear regression with L2 regularization.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
        lamb (float): The regularization parameter.
    Returns:
        tuple: A tuple containing the gradients with respect to weights and bias.  """
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0

    f_wb = x @ w + b

    error = f_wb - y

    dj_db = np.sum(error)/m

    dj_dw = (x.T @ error)/m

    dj_dw += (lamb/m) * w

    return dj_dw, dj_db
def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, lamb: float = 0.7, alpha: float = 0.01, num_iters: int = 1000):
    
    """Perform gradient descent to learn w and b. Updates w and b by taking num_iters gradient steps with learning rate alpha.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w_in (ndarray): The weights of shape (n,).
        b_in (float): The bias term.
        lamb (float): The regularization parameter.
    Returns:
        tuple: A tuple containing the updated weights, bias, and cost history.
    """

    cost_history = []
    for j in range(num_iters):
        dj_dw, dj_db = compute_gradient (x,y,w_in,b_in, lamb)
        w_in -= alpha*dj_dw
        b_in -= alpha*dj_db
        cost = cost_function(x,y,w_in,b_in, lamb)
        cost_history.append(cost)

        if j % 100 == 0:
            print(f"Iteration {j}: Cost {cost}")
    return w_in, b_in, cost_history

def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """ Predicts the target values using the learned weights and bias.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
    Returns:
        ndarray: The predicted target values of shape (m,).
    """
    sales_pred = x @ w + b
    return sales_pred