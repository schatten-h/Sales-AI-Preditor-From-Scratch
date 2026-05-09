import numpy as np
import pandas as pd

from utils.preproccessing import*
import utils.preproccessing as up
import models.Linearregression as lr
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
w_final, b_final = lr.gradient_descent(x_train, y_train, w, b, lamb = 0.7, alpha = 0.01,num_iters = 1000 )
y_pred = lr.predict(x_test,w_final,b_final)
print(w_final)
print(b_final)
print(y_pred[:10])

