#%%
import os 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import pickle 

# TODO 

# Train on days 7 and 9, test on day 6. As a comparison. 
# Compare decoder performance on training days
# 

# DONE

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

time_pairs = [(1, 20), (101, 140), (161,200), (201,240)]

#FPS = 20
# t = 200 = spout frame

n_folds = 5

#%%%
def run_analysis(t_min, t_max, classifier, param_grid, train_sessions, test_session, test_idx, 
                 each_trial = False, 
                 tmin_test = None,
                 tmax_test = None, 
                 train_indices = [0,1]):

    results = pd.DataFrame()
    testpreds = {}
    trainpreds = {}

    for treatment in conditions:
        animals = df.loc[df.treatment == treatment, "Animal"].unique()
        print(f"Fitting model for {treatment}")
        for animal in animals:
            print(f"Fitting model for {animal}")
            df_con = df[(df.treatment == treatment) & (df.Animal == animal)]
            _, cell_stats, _, animals = compute_stats(df, treatment, sessions = sessions)
            cell_fully_tracked = [k for k,v in cell_stats[animal].items() if (v[train_indices[0]] == 1 and v[train_indices[1]] == 1 and v[test_idx] == 1)]

            #Prepare training data
            ds_train = df_con.loc[(df_con.variable >= t_min) & (df_con.variable < t_max)]
            ds_train = ds_train.loc[df_con.Session.isin(train_sessions)].set_index('MatchID')
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

            def prepare_data(df_con, tmin, tmax):
                ds_test = df_con.loc[(df_con.variable >= tmin) & (df_con.variable < tmax)]
                ds_test = ds_test.loc[df_con.Session == test_session].set_index('MatchID')
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
                return X_test, groups_test, y_test
            X_test, groups_test, y_test = prepare_data(df_con, t_min, t_max)
            if tmin_test is not None and tmax_test is not None:
                X_test_long, _, y_test_long = prepare_data(df_con, tmin_test, tmax_test)

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
            preds_proba = cross_val_predict(search.best_estimator_, X, y, groups = groups, cv = splitter, method = 'predict_proba')

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

            y_pred_test_long = search.best_estimator_.predict(X_test_long)
            y_pred_test_long_probs = search.best_estimator_.predict_proba(X_test_long)
            testpreds[(treatment, animal)] = [y_test_long, y_pred_test_long, y_pred_test_long_probs]
            trainpreds[(treatment, animal)] = [y, preds, preds_proba]

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

    return results, testpreds, trainpreds

#Test code
# run_analysis(t_min = 201, 
#              t_max = 240, 
#              classifier = lda_classifier, 
#              param_grid = param_grid, 
#              train_sessions = ['7', '9'], 
#              test_session = '2', 
#              test_idx = 0, 
#              tmin_test = 0,
#              tmax_test = 500, 
#              train_indices=[2,3])

def run_decoder(time_pairs, 
                test_pairs, 
                classifiers, 
                grids, 
                train_sessions, 
                test_session, 
                test_idx, 
                train_indices = [0,1]):

    day_results = {}
    for cls, grid in zip(classifiers, grids):
        print("Using", cls)
        print("Decoding", time_pairs)
        day_results[(repr(cls), time_pairs)] = run_analysis(*time_pairs, 
                                                            cls, 
                                                            grid, 
                                                            train_sessions, 
                                                            test_session, 
                                                            test_idx, 
                                                            each_trial = True, 
                                                            tmin_test=test_pairs[0], 
                                                            tmax_test = test_pairs[1], 
                                                            train_indices = train_indices
                                                            )

    ave_resultsday = {k:v[0].val_accuracy.mean() for k,v in day_results.items()}
    results_summary = day_results[("Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=20)),\n                ('lda', LinearDiscriminantAnalysis())])", (201, 240))]

    return ave_resultsday, day_results, results_summary

lr_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic",logistic)])
lda_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda",lda)])
rf_classifier = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

param_grid = {"pca__n_components": [15]}

time_pairs = (201,240)
test_pairs = (0, 500)
classifiers = [lda_classifier]
grids = [param_grid]

train_sessions = ['7', '9']
test_session = '2'
test_idx = 0

ave_resultsday2, day2results, results_d2 = run_decoder(time_pairs, 
                                                       test_pairs, 
                                                       classifiers, 
                                                       grids, 
                                                       train_sessions, 
                                                       test_session, 
                                                       test_idx, 
                                                       train_indices=[2,3])

train_sessions = ['7', '9']
test_session = '6'
test_idx = 1

ave_resultsday6, day6results, results_d6 = run_decoder(time_pairs, 
                                                       test_pairs, 
                                                       classifiers, 
                                                       grids, 
                                                       train_sessions, 
                                                       test_session, 
                                                       test_idx, 
                                                       train_indices=[2,3])

fn_out_d2 = './decoder_day2_results_may10.csv'
fn_out_d6 = './decoder_day6_results_may10.csv'
fn_out_ave_d2_results = './decoder_day2_results_average_over_folds_may10.csv'
fn_out_ave_d6_results = './decoder_day6_results_average_over_folds_may10.csv'
results_d2[0].to_csv(fn_out_d2)
results_d6[0].to_csv(fn_out_d6)

with open('day2results_may10.pkl', 'wb') as f:
    pickle.dump(day2results, f)

with open('day6results_may10.pkl', 'wb') as f:
    pickle.dump(day6results, f)