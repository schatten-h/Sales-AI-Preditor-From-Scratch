import numpy as np
def cost_function(x,y,w,b, lamb = 0.7):
    m,n = x.shape
    cost_sum = 0
    reg_sum = 0

    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost_sum += (f_wb_i - y[i])**2
    cost_sum *= (1/(2*m))
    
    for j in range(n):
        reg_sum += w[j]**2
    reg_sum *= lamb/(2*m) 

    total_cost = cost_sum + reg_sum

    return total_cost

def compute_gradient(x,y,w,b, lamb = 0.7):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):

        f_wb_i = np.dot(x[i],w) + b

        dj_db += (f_wb_i - y[i])

        for j in range(n):
            dj_dw[j] += (f_wb_i - y[i]) * x[i][j]

    for j in range(n):
        dj_dw[j] += lamb * w[j]

    dj_db *= (1/m)
    dj_dw *= (1/m)
    return dj_dw, dj_db
def gradient_descent(x,y,w_in,b_in, lamb = 0.7, alpha = 0.01,num_iters = 1000):

    for j in range(num_iters):
        dj_dw, dj_db = compute_gradient (x,y,w_in,b_in, lamb)
        w_in = w_in - alpha*dj_dw
        b_in -= alpha*dj_db

        if j % 100 == 0:
            print(f"Iteration {j}: Cost {cost_function(x,y,w_in,b_in, lamb)}")
    return w_in, b_in

def predict(x,w,b):
    m = x.shape[0]
    p = np.zeros(m)
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        p[i] = f_wb_i
    return p