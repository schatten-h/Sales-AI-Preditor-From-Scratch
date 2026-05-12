import numpy as np
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function.
        Args:
            z (ndarray): The input array.
        Returns: 
            ndarray: The output of the sigmoid function applied element-wise to z.
   """
    return 1/ (1+ np.exp(-z))

def cost_function(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lamb: float = 0.7) -> float:
    """ Compute cost for logistic regression with L2 regularization.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
        lamb (float): The regularization parameter.
    Returns:
        float: The computed cost.
    """
    m = x.shape[0]
    cost = 0
    epsilon = 1e-15

    z = x @ w + b
    z = np.clip(z, -500, 500)
    f= sigmoid(z)
    f = np.clip(f, epsilon, 1 - epsilon)
    cost = (
    -np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))
) / m
    reg_cost = (lamb*np.sum(w**2))/(2*m)

    return cost + reg_cost

def compute_gradients(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lamb: float = 0.7) -> tuple[np.ndarray, float] :
    """ Compute the gradient for logistic regression with L2 regularization.
    Args: 
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
        lamb (float): The regularization parameter.
    Returns:
        tuple: A tuple containing the gradients with respect to weights and bias.
    """
    m,n = x.shape
    dw = np.zeros(n)
    db= 0

    z = x @ w + b
    f= sigmoid(z)
    dw = (x.T @ (f - y) )/m
    db = np.sum(f - y)/m
    
    dw += (lamb/m)* w

    return dw, db

def gradient_descent(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float = 0.01, lamb: float = 0.7, num_iters: int  = 1000) -> tuple[np.ndarray, float, list]:
    """Perform gradient descent to learn w and b. Updates w and b by taking num_iters gradient steps with learning rate alpha.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        y (ndarray): The target values of shape (m,).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
        alpha (float): The learning rate.
        lamb (float): The regularization parameter.
        num_iters (int): The number of iterations for gradient descent.
    Returns:
        tuple: A tuple containing the final weights, bias, and the history of cost values during gradient descent.
    """
    cost_history = []
    for iter in range(num_iters):
        dw, db = compute_gradients(x, y, w, b, lamb )
        w -= alpha * dw
        b -= alpha * db
        cost = cost_function(x, y, w, b, lamb)

        cost_history.append(cost)

        if iter% 100 == 0:
            print(f"Cost after iteration {iter} : {cost}")
    return w, b, cost_history

def predict(x: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions using the learned logistic regression parameters.
    Args:
        x (ndarray): The input dataset of shape (m, n).
        w (ndarray): The weights of shape (n,).
        b (float): The bias term.
    Returns:
        tuple: A tuple containing the predicted binary values and their probabilities.
    """
    m = x.shape[0]
    p = np.zeros(m)
    p_prob = np.zeros(m)

    z= x @ w + b
    p_prob = sigmoid(z)
    p =( p_prob >= 0.5).astype(int)
    return p, p_prob