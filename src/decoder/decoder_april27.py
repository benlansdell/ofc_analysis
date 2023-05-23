#%%
import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import pickle 

import matplotlib.pyplot as plt
from umap import UMAP

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from lib import compute_stats, run_analysis

fn_in = './data/processed/full2.csv'

#%%
#Load data 
df = pd.read_csv(fn_in, dtype = {'Animal': str, 
                                 'UnSpout_pos':str,
                                 'UnSpout_neg':str,
                                 'UnMid_pos':str,
                                 'UnMid_neg':str, 
                                 'Session': str, 
                                 'Trial': str})
df = df[['Animal', 'Session', 'CellID', 'RewardStatus', 'Trial', 'recording',
         'Reward', 'MatchID', 'treatment', 'resp', 'uID', 'variable', 'value']]

#%%
sessions = ['2', '6', '7', '9']
conditions = pd.unique(df['treatment'])
reward_statuses = pd.unique(df['RewardStatus'])

#%%

# train_sessions = [2]
# test_session = 6
# test_idx = 1

label_key = {'reward_spout_events': 1, 'unreward_spout_events': 0}

pca = PCA(n_components = 20)
scaler = StandardScaler()
logistic = LogisticRegression()
lda = LinearDiscriminantAnalysis()
rf = RandomForestClassifier()

lr_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic",logistic)])
lda_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda",lda)])
rf_classifier = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

param_grid = {
    "pca__n_components": [5, 15, 30, 50]
}

param_grid_rf = {'rf__n_estimators': [50, 100]}

time_pairs = [(1, 20), (101, 140), (161,200), (201,240)]

#FPS = 20
# t = 200 = spout frame

n_folds = 5

lr_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic",logistic)])
lda_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda",lda)])
rf_classifier = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

#param_grid = {
#    "pca__n_components": [5, 15, 30, 50]
#}

param_grid = {
    "pca__n_components": [15]
}

param_grid_rf = {'rf__n_estimators': [50, 100]}

#time_pairs = [(1, 20), (101, 140), (161,200), (201,240)]
#classifiers = [lr_classifier, lda_classifier, rf_classifier]
#grids = [param_grid, param_grid, param_grid_rf]

#Earlier times
#time_pairs = [(1, 40), (41, 80), (81, 120), (121, 160), (161,200), (201,240), 
#              (1, 80), (81, 160), (161,240), (201,280)]
time_pairs = (201,240)
test_pairs = (0, 500)
classifiers = [lda_classifier]
grids = [param_grid]

#Longer times

#Test each
# classifiers = [lda_classifier]
# grids = [param_grid]

# classifiers = [rf_classifier]
# grids = [param_grid_rf]

train_sessions = ['2', '6']

test_session = '7'
test_idx = 2

# test_session = '9'
# test_idx = 3

day7results = {}
for cls, grid in zip(classifiers, grids):
    print("Using", cls)
    print("Decoding", time_pairs)
    day7results[(repr(cls), time_pairs)] = run_analysis(*time_pairs, cls, grid, train_sessions, test_session, test_idx, each_trial = True, tmin_test=test_pairs[0], tmax_test = test_pairs[1])

#Take average val accuracies
ave_resultsday7 = {k:v[0].val_accuracy.mean() for k,v in day7results.items()}

#Run for Day 9:
test_session = '9'
test_idx = 3

classifiers = [lda_classifier]
grids = [param_grid]

day9results = {}
for cls, grid in zip(classifiers, grids):
    print("Using", cls)
    print("Decoding", time_pairs)
    day9results[(repr(cls), time_pairs)] = run_analysis(*time_pairs, cls, grid, train_sessions, test_session, test_idx, each_trial = True, tmin_test=test_pairs[0], tmax_test = test_pairs[1])

#Take average val accuracies
ave_resultsday9 = {k:v[0].val_accuracy.mean() for k,v in day9results.items()}

#Split predictions by trial
results_d9 = day9results[("Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=20)),\n                ('lda', LinearDiscriminantAnalysis())])", (201, 240))]
results_d7 = day7results[("Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=20)),\n                ('lda', LinearDiscriminantAnalysis())])", (201, 240))]

fn_out_d9 = './decoder_day9_results_april27.csv'
fn_out_d7 = './decoder_day7_results_april27.csv'
fn_out_ave_d7_results = './decoder_day7_results_average_over_folds_april27.csv'
fn_out_ave_d9_results = './decoder_day7_results_average_over_folds_april27.csv'

results_d9[0].to_csv(fn_out_d9)
results_d7[0].to_csv(fn_out_d7)

#ave_resultsday7.to_csv(fn_out_ave_d7_results)
#ave_resultsday9.to_csv(fn_out_ave_d9_results)

#Save all results from runs:
with open('day9results_april27.pkl', 'wb') as f:
    pickle.dump(day9results, f)

# Save day7results to a pickle file
with open('day7results_april27.pkl', 'wb') as f:
    pickle.dump(day7results, f)

