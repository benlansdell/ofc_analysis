#%%
import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import pickle 

# CCA

# TODO 

# Confusion matrices... How to summarize them?

# DONE

# Statistical testing for baseline performance?
# Try other windows, not just 2s 
# Earlier times 

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

from lib import compute_stats

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

#%%%
def run_analysis(t_min, t_max, classifier, param_grid, train_sessions, test_session, test_idx, each_trial = False):

    results = pd.DataFrame()

    for treatment in conditions:
        animals = df.loc[df.treatment == treatment, "Animal"].unique()
        print(f"Fitting model for {treatment}")
        for animal in animals:
            print(f"Fitting model for {animal}")
            df_con = df[(df.treatment == treatment) & (df.Animal == animal)]
            _, cell_stats, _, animals = compute_stats(df, treatment, sessions = sessions)
            cell_fully_tracked = [k for k,v in cell_stats[animal].items() if (v[0] == 1 and v[1] == 1 and v[test_idx] == 1)]

            df_con = df_con.loc[(df_con.variable >= t_min) & (df_con.variable < t_max)]

            #Prepare training data
            ds_train = df_con.loc[df_con.Session.isin(train_sessions)].set_index('MatchID')
            ds_train = ds_train[ds_train.index.isin(cell_fully_tracked)]
            ds_train = ds_train.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
            ds_train = pd.pivot(ds_train, columns = 'MatchID', index = ['Trial', 'Session', 'RewardStatus', 'time'], values = 'value')
            ds_train = ds_train.reset_index()
            ds_train['Session_trial'] = ds_train['Session'] + '_' + ds_train['Trial']
            col_names = list(ds_train.columns)
            ds_train = ds_train[[col_names[-1]] + col_names[:-1]]
            ds_train = ds_train.drop(columns = ['Session', 'time', 'Trial'])
            ds_train = ds_train.dropna()
            X = ds_train.iloc[:,2:].values
            groups = ds_train['Session_trial']
            y = ds_train['RewardStatus'].apply(lambda x: label_key[x]).values

            #Prepare test data
            ds_test = df_con.loc[df_con.Session == test_session].set_index('MatchID')
            ds_test = ds_test[ds_test.index.isin(cell_fully_tracked)]
            ds_test = ds_test.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
            ds_test = pd.pivot(ds_test, columns = 'MatchID', index = ['Trial', 'Session', 'RewardStatus', 'time'], values = 'value')
            ds_test = ds_test.reset_index()
            ds_test['Session_trial'] = ds_test['Session'] + '_' + ds_test['Trial']
            col_names = list(ds_test.columns)
            ds_test = ds_test[[col_names[-1]] + col_names[:-1]]
            ds_test = ds_test.drop(columns = ['Trial', 'Session', 'time'])
            ds_test = ds_test.dropna()
            X_test = ds_test.iloc[:,2:].values
            groups_test = ds_test['Session_trial']
            y_test = ds_test['RewardStatus'].apply(lambda x: label_key[x]).values

            #Perform group level CV
            splitter = GroupKFold(n_splits = n_folds)
            #scores = cross_validate(classifier, X, y, 
            #                        groups = groups, 
            #                        cv = splitter,
            #                        return_train_score=True)
            search = GridSearchCV(classifier, param_grid, n_jobs=16, cv = splitter)
            search.fit(X, y, groups = groups)
            
            #train_accuracy = np.mean(scores['train_score'])
            val_accuracy = search.best_score_

            preds = cross_val_predict(search.best_estimator_, X, y, groups = groups, cv = splitter)

            prop_reward_pred = sum(preds)/len(preds)
            prop_reward = sum(y)/len(y)

            #Compute more stats... 
            val_f1 = f1_score(y, preds)
            val_tp = np.sum((y == 1) & (preds == 1))/len(y)
            val_fp = np.sum((y == 0) & (preds == 1))/len(y)
            val_tn = np.sum((y == 0) & (preds == 0))/len(y)
            val_fn = np.sum((y == 1) & (preds == 0))/len(y)

            #Apply for whole set of 12 trials
            search.best_estimator_.fit(X, y)
            y_pred_test = search.best_estimator_.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            prop_reward_pred_test = sum(y_pred_test)/len(y_pred_test)
            prop_reward_test = sum(y_test)/len(y_test)

            test_f1 = f1_score(y_test, y_pred_test)
            test_tp = np.sum((y_test == 1) & (y_pred_test == 1))/len(y_pred_test)
            test_fp = np.sum((y_test == 0) & (y_pred_test == 1))/len(y_pred_test)
            test_tn = np.sum((y_test == 0) & (y_pred_test == 0))/len(y_pred_test)
            test_fn = np.sum((y_test == 1) & (y_pred_test == 0))/len(y_pred_test)

            #Apply to each trial
            if each_trial:
                new_rows = {'treatment': treatment,
                        'animal': animal,
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'prop_reward_pred_val': prop_reward_pred,
                        'prop_reward_pred_test': prop_reward_pred_test,
                        'prop_reward': prop_reward,
                        'prop_reward_test': prop_reward_test,
                        'val_f1': val_f1, 
                        'test_f1': test_f1,
                        'test_tp': test_tp,
                        'test_tn': test_tn,
                        'test_fp': test_fp,
                        'test_fn': test_fn,
                        'val_tp': val_tp,
                        'val_tn': val_tn,
                        'val_fp': val_fp,
                        'val_fn': val_fn
                        }
                by_trial_test_preds = []
                by_trial_test = []
                g_test = groups_test.unique()
                for grp in g_test:
                    y_pred_test_trial = search.best_estimator_.predict(X_test[groups_test == grp])
                    y_test_trial = y_test[groups_test == grp]
                    prop_reward_pred_test_trial = sum(y_pred_test_trial)/len(y_pred_test_trial)            
                    by_trial_test_preds.append(prop_reward_pred_test_trial)
                    by_trial_test.append(sum(y_test_trial)/len(y_test_trial))
                new_rows['Session_trial'] = g_test
                new_rows['Reward'] = by_trial_test 
                new_rows['Prop_pred_reward_trial'] = by_trial_test_preds
            else:
                new_rows = {'treatment': [treatment],
                        'animal': [animal],
                        'val_accuracy': [val_accuracy],
                        'test_accuracy': [test_accuracy],
                        'prop_reward_pred_val': [prop_reward_pred],
                        'prop_reward_pred_test': [prop_reward_pred_test],
                        'prop_reward': [prop_reward],
                        'prop_reward_test': [prop_reward_test],
                        'val_f1': [val_f1], 
                        'test_f1': [test_f1],
                        'test_tp': [test_tp],
                        'test_tn': [test_tn],
                        'test_fp': [test_fp],
                        'test_fn': [test_fn],
                        'val_tp': [val_tp],
                        'val_tn': [val_tn],
                        'val_fp': [val_fp],
                        'val_fn': [val_fn]}

            results = results.append(pd.DataFrame(new_rows))

    return results

def run_baseline_scores(t_min, t_max, classifier, train_sessions, test_session, test_idx, N = 100):

    #Computed through shuffling. Return set of accuracies and f1 scores for classification performance on shuffled data
    results = pd.DataFrame()

    for treatment in conditions:
        animals = df.loc[df.treatment == treatment, "Animal"].unique()
        print(f"Fitting model for {treatment}")
        for animal in animals:
            print(f"Fitting model for {animal}")
            df_con = df[(df.treatment == treatment) & (df.Animal == animal)]
            _, cell_stats, _, animals = compute_stats(df, treatment, sessions = sessions)
            cell_fully_tracked = [k for k,v in cell_stats[animal].items() if (v[0] == 1 and v[1] == 1 and v[test_idx] == 1)]

            df_con = df_con.loc[(df_con.variable >= t_min) & (df_con.variable < t_max)]

            #Prepare training data
            ds_train = df_con.loc[df_con.Session.isin(train_sessions)].set_index('MatchID')
            ds_train = ds_train[ds_train.index.isin(cell_fully_tracked)]
            ds_train = ds_train.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
            ds_train = pd.pivot(ds_train, columns = 'MatchID', index = ['Trial', 'Session', 'RewardStatus', 'time'], values = 'value')
            ds_train = ds_train.reset_index()
            ds_train['Session_trial'] = ds_train['Session'] + '_' + ds_train['Trial']
            col_names = list(ds_train.columns)
            ds_train = ds_train[[col_names[-1]] + col_names[:-1]]
            ds_train = ds_train.drop(columns = ['Session', 'time', 'Trial'])
            ds_train = ds_train.dropna()
            X = ds_train.iloc[:,2:].values
            groups = ds_train['Session_trial']
            y = ds_train['RewardStatus'].apply(lambda x: label_key[x]).values

            #Prepare test data
            ds_test = df_con.loc[df_con.Session == test_session].set_index('MatchID')
            ds_test = ds_test[ds_test.index.isin(cell_fully_tracked)]
            ds_test = ds_test.reset_index().rename(columns = {'Unnamed: 0': 'idx', 'variable': 'time'})
            ds_test = pd.pivot(ds_test, columns = 'MatchID', index = ['Trial', 'Session', 'RewardStatus', 'time'], values = 'value')
            ds_test = ds_test.reset_index()
            ds_test['Session_trial'] = ds_test['Session'] + '_' + ds_test['Trial']
            col_names = list(ds_test.columns)
            ds_test = ds_test[[col_names[-1]] + col_names[:-1]]
            ds_test = ds_test.drop(columns = ['Trial', 'Session', 'time'])
            ds_test = ds_test.dropna()
            X_test = ds_test.iloc[:,2:].values
            groups_test = ds_test['Session_trial']
            y_test = ds_test['RewardStatus'].apply(lambda x: label_key[x]).values

            splitter = GroupKFold(n_splits = n_folds)

            for rep in tqdm(range(N)):
                np.random.shuffle(y)
                x_train, x_val, y_train, y_val = train_test_split(X, y)

                classifier.fit(x_train, y_train) 
                preds = classifier.predict(x_val)
                val_f1 = f1_score(y_val, preds)
                val_accuracy = accuracy_score(y_val, preds)

                y_pred_test = classifier.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                test_f1 = f1_score(y_test, y_pred_test)

                new_rows = {'rep': [rep],
                            'treatment': [treatment],
                            'animal': [animal],
                            'val_accuracy': [val_accuracy],
                            'test_accuracy': [test_accuracy],
                            'val_f1': [val_f1], 
                            'test_f1': [test_f1]
                            }

                results = results.append(pd.DataFrame(new_rows))

    return results

lr_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic",logistic)])
lda_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda",lda)])
rf_classifier = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

param_grid = {
    "pca__n_components": [5, 15, 30, 50]
}

param_grid_rf = {'rf__n_estimators': [50, 100]}

#time_pairs = [(1, 20), (101, 140), (161,200), (201,240)]
#classifiers = [lr_classifier, lda_classifier, rf_classifier]
#grids = [param_grid, param_grid, param_grid_rf]

#Earlier times
time_pairs = [(1, 40), (41, 80), (81, 120), (121, 160), (161,200), (201,240), 
              (1, 80), (81, 160), (161,240), (201,280)]
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
    for tp in time_pairs:
        print("Decoding", tp)
        day7results[(repr(cls), tp)] = run_analysis(*tp, cls, grid, train_sessions, test_session, test_idx, each_trial = True)

#Take average val accuracies

ave_resultsday7 = {k:v.val_accuracy.mean() for k,v in day7results.items()}

#Looks like best performance at 201-240 -- as we would expect.

#Run for Day 9:
test_session = '9'
test_idx = 3

time_pairs = [(1, 40), (41, 80), (81, 120), (121, 160), (161,200), (201,240), 
              (1, 80), (81, 160), (161,240), (201,280)]
classifiers = [lda_classifier]
grids = [param_grid]

day9results = {}
for cls, grid in zip(classifiers, grids):
    print("Using", cls)
    for tp in time_pairs:
        print("Decoding", tp)
        day9results[(repr(cls), tp)] = run_analysis(*tp, cls, grid, train_sessions, test_session, test_idx, each_trial = True)

### Shuffling results
train_sessions = ['2', '6']

test_session = '7'
test_idx = 2

day7results_shuffled = {}
for tp in time_pairs:
    print("Decoding", tp)
    day7results_shuffled[tp] = run_baseline_scores(*tp, lda_classifier, train_sessions, test_session, test_idx, N = 50)

test_session = '9'
test_idx = 3

day9results_shuffled = {}
for tp in time_pairs:
    print("Decoding", tp)
    day9results_shuffled[tp] = run_baseline_scores(*tp, lda_classifier, train_sessions, test_session, test_idx, N = 50)

with open('day9results_shuffled.pkl', 'wb') as f:
    pickle.dump(day9results_shuffled, f)

# Save day7results to a pickle file
with open('day7results_shuffled.pkl', 'wb') as f:
    pickle.dump(day7results_shuffled, f)


#Take average val accuracies
ave_resultsday9 = {k:v.val_accuracy.mean() for k,v in day9results.items()}

#Split predictions by trial
results_d9 = day9results[("Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=20)),\n                ('lda', LinearDiscriminantAnalysis())])", (201, 240))]
results_d7 = day7results[("Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=20)),\n                ('lda', LinearDiscriminantAnalysis())])", (201, 240))]

fn_out_d9 = './decoder_day9_results_april18.csv'
fn_out_d7 = './decoder_day7_results_april18.csv'
fn_out_ave_d7_results = './decoder_day7_results_average_over_folds_april18.csv'
fn_out_ave_d9_results = './decoder_day7_results_average_over_folds_april18.csv'

results_d9.to_csv(fn_out_d9)
results_d7.to_csv(fn_out_d7)

#ave_resultsday7.to_csv(fn_out_ave_d7_results)
#ave_resultsday9.to_csv(fn_out_ave_d9_results)

#Save all results from runs:
with open('day9results.pkl', 'wb') as f:
    pickle.dump(day9results, f)

# Save day7results to a pickle file
with open('day7results.pkl', 'wb') as f:
    pickle.dump(day7results, f)

# %%
