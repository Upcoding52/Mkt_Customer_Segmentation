#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:56:50 2020

@author: daixinling
"""

[globals().pop(var) for var in dir() if not var.startswith("__")]

import pandas as pd
#from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

fc_new = pd.read_csv('transactions_n100000.csv')

profile = ProfileReport(fc_new)
profile.to_file("HW2_Report.html")

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

inertias = {}
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=2020)
    kmeans.fit(df)
    inertias[k] = kmeans.inertia_
print(inertias)

ax = plt.subplot()
ax.plot(inertias.keys(), inertias.values(), '-*')
ax.set_xticks(np.arange(2, 20))
ax.grid()
plt.show()

sil = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k).fit(df)
    labels = kmeans.labels_
    sil.append(silhouette_score(df, labels, metric='euclidean'))

ax = plt.subplot()
ax.plot(range(2, 7), sil, '-*')
ax.set_xticks(np.arange(2, 7))
ax.grid()
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=2020)
y_pred = kmeans.fit_predict(df)

reduced_data = PCA(n_components=2).fit_transform(df)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=y_pred, data=results)
plt.title('K-means Clustering with 4 dimensions')
plt.show()

df['cluster'] = y_pred

df.to_csv('data_with_clustering.csv')