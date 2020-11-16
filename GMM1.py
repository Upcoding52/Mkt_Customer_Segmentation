[globals().pop(var) for var in dir() if not var.startswith("__")]

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture


fc_new = pd.read_csv('transactions_n100000.csv')

item_encoder = LabelBinarizer()
item_encoder.fit(fc_new['item_name'])
transformed = item_encoder.transform(fc_new['item_name'])
ohe_df = pd.DataFrame(transformed)
fc_new = pd.concat([fc_new, ohe_df], axis=1)

for i in range(4):
    fc_new[i] = fc_new[i] * fc_new['item_count']

fc_new.rename(columns={0: 'Burger', 1: 'Fries', 2: 'Salad', 3: 'Shake'}, inplace=True)

fc_new.drop(['lat', 'long', 'item_count', 'item_name'], axis=1, inplace=True)

fc_new['order_timestamp'] = pd.to_datetime(fc_new['order_timestamp'], format='%Y-%m-%d %H:%M')

fc_new['order_timestamp'] = fc_new['order_timestamp'].apply(
    lambda x: 0 if x.hour <= 10 else (1 if x.hour < 14 else (2 if x.hour < 22 else 3)))

df = fc_new.groupby('ticket_id').agg(
    {'order_timestamp': 'mean', 'location': 'mean', 'Burger': 'sum', 'Fries': 'sum', 'Salad': 'sum', 'Shake': 'sum'})

df = pd.get_dummies(data=df, columns=['order_timestamp', 'location'])

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)

gmmModel = GaussianMixture(n_components=3, covariance_type='diag', random_state=0)
gmmModel.fit(df)
labels = gmmModel.predict(df)

reduced_data = PCA(n_components=2).fit_transform(df)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=labels, data=results)
plt.title('GMM Clustering with 3 components')
plt.show()


