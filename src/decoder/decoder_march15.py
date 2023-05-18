#%%
import os 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from umap import UMAP

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib import compute_stats

fn_in = './data/input/explore3_mean_responses.csv'

#%%
#Load data 
df = pd.read_csv(fn_in)

#%%
sessions = [2, 6, 7, 9]
conditions = pd.unique(df['treatment'])

#%%

# train_sessions = [2]
# test_session = 6
# test_idx = 1

train_sessions = [2, 6]
# test_session = 7
# test_idx = 2

test_session = 9
test_idx = 3

label_key = {'Reward Spout': 1, 'Unreward Spout': 0}

results = pd.DataFrame()

pca = PCA(n_components = 20)
scaler = StandardScaler()
logistic = LogisticRegression()
classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

#classifier = LogisticRegression()

t_min = 201
t_max = 240

t_val_min = 241
t_val_max = 261

for treatment in conditions:
    animals = df.loc[df.treatment == treatment, "Animal"].unique()
    for animal in animals:
        df_con = df[(df.treatment == treatment) & (df.Animal == animal)]
        _, cell_stats, _, animals = compute_stats(df, treatment)
        cell_fully_tracked = [k for k,v in cell_stats[animal].items() if (v[0] == 1 and v[1] == 1 and v[test_idx] == 1)]

        df_con = df_con.loc[(df_con.variable >= t_min) & (df_con.variable < t_max)]

        ds_train = df_con.loc[df_con.Session.isin(train_sessions)].set_index('MatchID')
        ds_train = ds_train[ds_train.index.isin(cell_fully_tracked)]
        ds_train = ds_train.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
        ds_train = pd.pivot(ds_train, columns = 'MatchID', index = ['Session', 'Region', 'time'], values = 'value')
        ds_train = ds_train.reset_index().drop(columns = ['Session', 'time'])
        ds_train = ds_train.dropna()
        X = ds_train.iloc[:,1:].values
        y = ds_train['Region'].apply(lambda x: label_key[x]).values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

        ds_test = df_con.loc[df_con.Session == test_session].set_index('MatchID')
        ds_test = ds_test[ds_test.index.isin(cell_fully_tracked)]

        ds_test = ds_test.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
        ds_test = pd.pivot(ds_test, columns = 'MatchID', index = ['Session', 'Region', 'time'], values = 'value')
        ds_test = ds_test.reset_index().drop(columns = ['Session', 'time'])
        ds_test = ds_test.dropna()
        X_test = ds_test.iloc[:,1:].values
        y_test = ds_test['Region'].apply(lambda x: label_key[x]).values

        classifier.fit(X_train, y_train)
        y_pred_val = classifier.predict(X_val)
        y_pred = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        prop_reward_pred_train = sum(y_pred)/len(y_pred)
        prop_reward_pred_val = sum(y_pred_val)/len(y_pred_val)
        prop_reward_pred_test = sum(y_pred_test)/len(y_pred_test)

        results = results.append(pd.DataFrame({'treatment': [treatment],
                                               'animal':[animal],
                                               'train_accuracy': [train_accuracy],
                                               'val_accuracy':[val_accuracy],
                                               'test_accuracy':[test_accuracy],
                                               'prop_reward_pred_train': prop_reward_pred_train,
                                               'prop_reward_pred_val': prop_reward_pred_val,
                                               'prop_reward_pred_test': prop_reward_pred_test}))

results

#%%
pca = PCA(n_components=2)
pca_embedding = pca.fit_transform(X_train)

# %%
plt.scatter(x = pca_embedding[:,0], y = pca_embedding[:,1], c = y_train)

#%%
umap_embedding = UMAP(n_components = 2).fit_transform(X_train)

# %%
plt.scatter(x = umap_embedding[:,0], y = umap_embedding[:,1], c = y_train)

# %%
