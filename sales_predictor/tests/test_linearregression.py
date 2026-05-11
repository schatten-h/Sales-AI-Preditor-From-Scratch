import numpy as np
import pandas as pd
import os

#import necessary modules
from utils.preproccessing import*
import utils.preproccessing as up
import models.Linearregression as lr
import utils.metrics as metrics
import utils.visualization as viz

# Load the trained model
df = pd.read_csv('/data/generate.csv')
 
#Standartization of the data
X ,Y = up.split_features_target(df)

x = X.to_numpy()
y = Y.to_numpy()
X_norm, Y_norm, _, _ = up.z_score_normalization(x,y)
x_train, y_train, x_test,y_test, _= up.train_test_split(X_norm,Y_norm)

w = np.zeros(x_train.shape[1])
b = 0

#Prediction 
w_final, b_final, cost_history = lr.gradient_descent(x_train, y_train, w, b, lamb = 0.7, alpha = 0.01,num_iters = 1000 )
y_pred = lr.predict(x_test,w_final,b_final)
print(f"Final weights: {w_final}")
print(f"Final bias: {b_final}")
print(f"Predicted values: {y_pred[:10]}")

#save
os.makedirs("outputs" , exist_ok= True)
np.save("outputs/weights.npy", w_final)
np.save("outputs/bias.npy", b_final)

#metrics
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", metrics.root_mean_squared_error(y_test, y_pred))
print("R^2 Score:", metrics.r2_score(y_test, y_pred))  

# Visualizations
viz.plot_cost_history(cost_history)
viz.plot_predictions(y_test, y_pred)        
viz.plot_residuals(y_test, y_pred)