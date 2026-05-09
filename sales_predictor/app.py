
from utils.preproccessing import*
import utils.preproccessing as up
# Load the trained model
df = pd.read_csv('/data/generate.csv')
 
#Standartization of the data
X ,Y = up.split_features_target(df)
X_train, Y_train, _, _= up.train_test_split(X,Y)

X_train= X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_norm, Y_norm, _, _ = up.z_score_normalization(X_train,Y_train)

