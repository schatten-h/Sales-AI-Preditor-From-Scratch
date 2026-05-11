import numpy as np
import pandas as pd


n=1000
rng = np.random.default_rng(seed=42)
promo = rng.randint(0,2,n)
marketing=rng.randint(0,100,n)
price = rng.randint(1,100, n)
month = rng.randint(1,13,n)
sales = ( promo*20 -1.8*price + 2*(marketing) + 10*(np.sin(2*np.pi*month/12) + np.cos(2*np.pi*month/12)) + rng.normal(0, 5, n))
sales = np.maximum(sales, 0)
success = np.where(sales > np.mean(sales), 1, 0)


data = pd.DataFrame({
    "promo" : promo,
    "price" : price,
    "marketing" : marketing,
    "month_sin": np.sin(2*np.pi*month/12),
    "month_cos": np.cos(2*np.pi*month/12),
    "sales": sales,
    "success": success

}) 

data.describe()
data["success"].value_counts()
data.to_csv("generate.csv", index=False)
