import numpy as np
import pandas as pd

from utils.preproccessing import*
import utils.preproccessing as up
import models.Logisticregression as lg
import utils.classification_metrics as met
import utils.visualization as viz

# Load the trained model
df = pd.read_csv('/data/generate.csv')
 
#Standartization of the data
X ,Y = up.split_features_target(df, target_column="success")

x = X.to_numpy()
y = Y.to_numpy()
X_norm, Y_norm, _, _ = up.z_score_normalization(df,y)
x_train, y_train, x_test,y_test, _= up.train_test_split(X_norm,Y_norm)

w = np.zeros(x_train.shape[1])
b = 0

#Prediction 
w_final, b_final, cost_history = lg.gradient_descent(x_train, y_train, w, b, lamb = 0.7, alpha = 0.01,num_iters = 1000 )
y_pred, y_prob = lg.predict(x_test,w_final,b_final)
print(f"Final weights: {w_final}")
print(f"Final bias: {b_final}")
print(f"Predicted values: {y_pred[:10]}")
print(f"Predicted probabilities: {y_prob[:10]}")

#metrics
print(f"Accuracy: {met.accuracy_score(y_test, y_pred)}")
print(f"Precision: {met.precision_score(y_test, y_pred)}")
print(f"Recall: {met.recall_score(y_test, y_pred)}")
print(f"F1 Score: {met.f1_score(y_test, y_pred)}")

# Visualizations
viz.plot_cost_history(cost_history)
viz.plot_predictions(y_test, y_pred)        
viz.plot_residuals(y_test, y_pred)