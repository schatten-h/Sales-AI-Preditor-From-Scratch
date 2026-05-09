import pandas as pd

df = pd.read_csv('/data/generate.csv')
x = df[[ 'price', 'marketing']]
y = df['sales']
def z_score_normalization(x, y):
    mean_x = x.mean()
    std_x = x.std()
    mean_y = y.mean()
    std_y = y.std() 
    x_norm = (x - mean_x) / std_x
    y_norm = (y - mean_y) / std_y
    return x_norm,y_norm, mean_x, std_x, mean_y, std_y

def split_features_target(df):

    X = df[['price', 'marketing']]
    y = df['sales']

    return X, y
def train_test_split(X,y):
    split_index = int(0.8 * len(X))

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test 