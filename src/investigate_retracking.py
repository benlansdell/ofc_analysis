#Set to project root directory
import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


from glob import glob
import pickle 

import pyreadr
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

sample_dirs = ['/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal 1',
        '/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal 10',
        '/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal M3']

#Train the position decoder on this many rows
N_ROWS = 200000

def load_animal(animal, animal_to_files):
    data = []
    cells = []
    for fn in animal_to_files[animal]:
        df = pyreadr.read_r(fn)[None]
        df['filename'] = fn.split('/')[-1]
        data.append(df)
        cells.append(df.MatchID.unique())
    df_all = pd.concat(data)
    
    #Only include cells that are in all files
    #Take the intersection of all cells
    cells = set(cells[0]).intersection(*cells)
    df_all = df_all[df_all.MatchID.isin(cells)]
    df_all['group'] = df_all['filename'] + '-' + df_all['Session'].astype(str) + '-' + df_all['Trial Number'].astype(str).str.zfill(2)

    #Figure out the side the mouse went on, for each trial
    trials = df_all['group'].unique()

    for trial in trials:
        df_trial = df_all[df_all['group'] == trial]
        events = df_trial['event'].unique()
        if 'lspout_exit' in events and 'rspout_exit' in events:
            reward_side = 'both'
        elif 'lspout_exit' in events:
            reward_side = 'left'
        elif 'rspout_exit' in events:
            reward_side = 'right'
        else:
            reward_side = 'none'
        df_all.loc[df_all['group'] == trial, 'maze_choice'] = reward_side
    return df_all

files = []
for d in sample_dirs:
    files.extend(glob(d + "/*.Rds"))

animal_to_files = {}
for fn in files:
    animal = fn.split('/')[-2]
    if animal not in animal_to_files:
        animal_to_files[animal] = []
    animal_to_files[animal].append(fn)

# animal = 'Control Animal 1'
# training_reward = 'left'

# animal = 'Control Animal 10'
# training_reward = 'right'

animal = 'Control Animal M3'
training_reward = 'left'

df = load_animal(animal, animal_to_files)

def train_one_side_test_other(df_all, hyper_params, train_reward = 'left', do_shuffling = True):

    n_pcs = hyper_params['n_pcs']
    n_clusters = hyper_params['n_clusters']

    #Turn into wide table, by cell type and dff signal
    index_cols = ['group', 'event', 'Reward', 'Centre position X', 'Centre position Y', 'frame', 'maze_choice']
    df_wide = df_all.pivot(index=index_cols, columns='MatchID', values='dff').reset_index()
    df_wide.columns = list(df_wide.columns[:len(index_cols)]) + [f'Cell_{i}' for i in df_wide.columns[len(index_cols):]]
    df_wide = df_wide.fillna(0)
    y = df_wide[['Centre position X', 'Centre position Y']].values
    X = df_wide.iloc[:,len(index_cols):].values
    groups = df_wide['group']
    events = df_wide['event']
    train_index = df_wide[df_wide['Reward'] == train_reward].index
    train_index = np.random.choice(train_index, N_ROWS, replace=True)
    test_index = df_wide[df_wide['Reward'] != train_reward].index

    train_accs = []
    test_accs = []
    train_accs_shuffled = []
    test_accs_shuffled = []

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    test_groups = groups[test_index].reset_index(drop = True)
    test_events = events[test_index].reset_index(drop = True)
    trial_choice = df_wide['maze_choice']
    trial_choice_test = trial_choice[test_index].reset_index(drop = True)
    test_group_names = test_groups.unique()
   
    pca = PCA(n_components=n_pcs)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(y_train)
    y_train_kmeans = kmeans.predict(y_train)
    y_test_kmeans = kmeans.predict(y_test)
    
    #try logistic regression 
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_pca, y_train_kmeans)
    
    y_pred_train = clf.predict(X_train_pca)
    y_pred_test = clf.predict(X_test_pca)
    
    #Evaluate
    train_accuracy = accuracy_score(y_train_kmeans, y_pred_train)
    test_accuracy = accuracy_score(y_test_kmeans, y_pred_test)
    
    test_accuracy_by_group = {}
    for group in test_group_names:
        test_group_index = test_groups[test_groups == group].index
        test_group_accuracy = accuracy_score(y_test_kmeans[test_group_index], y_pred_test[test_group_index])
        test_accuracy_by_group[group] = test_group_accuracy
    
    test_accs.append(test_accuracy)
    train_accs.append(train_accuracy)

    n_resamples = 50
    if do_shuffling:
        train_accs_shuffled_og = []
        test_accs_shuffled_og = []
        for i in tqdm(range(n_resamples)):
            np.random.shuffle(y_train_kmeans)
            np.random.shuffle(y_test_kmeans)
            train_accuracy = accuracy_score(y_train_kmeans, y_pred_train)
            test_accuracy = accuracy_score(y_test_kmeans, y_pred_test)
            train_accs_shuffled_og.append(train_accuracy)
            test_accs_shuffled_og.append(test_accuracy)    
            
        train_accs_shuffled.append(np.mean(train_accs_shuffled_og))
        test_accs_shuffled.append(np.mean(test_accs_shuffled_og))
        
    print("Train accuracy: ", np.mean(train_accs))
    print("Test accuracy: ", np.mean(test_accs))
    print("Test accuracy shuffled", np.mean(test_accs_shuffled))

    results = {'train_accs': train_accs, 
               'test_accs': test_accs, 
               'test_accs_shuffled': test_accs_shuffled, 
               'test_accuracy_by_group': test_accuracy_by_group,
               'y_test_kmeans': y_test_kmeans,
               'y_pred_test': y_pred_test,
               'test_groups': test_groups,
               'test_events': test_events,
               'trial_choice_test': trial_choice_test,
               'y_test_pos': y_test}
    return results

print(f"Training on {training_reward}, testing on the other side")
results_train_left_test_right = train_one_side_test_other(df, {'n_pcs': 10, 'n_clusters': 10}, train_reward=training_reward, do_shuffling=False)

#Save results for plotting
with open(f'results_animal_{animal}.pkl', 'wb') as f:
    pickle.dump(results_train_left_test_right, f)

#Do a hyperparameter search over n_pcs and n_clusters
hyper_params = {'n_pcs': [10, 50, 100], 'n_clusters': [5, 10, 20, 50]}
hyper_param_results = {}
from itertools import product
for n_pcs, n_clusters in tqdm(product(hyper_params['n_pcs'], hyper_params['n_clusters']), total = len(hyper_params['n_pcs']) * len(hyper_params['n_clusters'])):
    print(f"Training on left, testing on right, n_pcs = {n_pcs}, n_clusters = {n_clusters}")
    hyper_params = {'n_pcs': n_pcs, 'n_clusters': n_clusters}
    hyper_param_results[(n_pcs, n_clusters)] = train_one_side_test_other(df, hyper_params, train_reward='left')


with open('hyper_param_results.pkl', 'wb') as f:
    pickle.dump(hyper_param_results, f)

#Print results
for key, value in hyper_param_results.items():
    print(key, f"test acc: {value['test_accs']}, train acc: {value['train_accs']}")
    

def train_logo(df_all, reward = 'right'):

    df_cis = df_all[df_all['Reward'] == reward]
    df_cis['group'] = df_cis['filename'] + df_cis['Session'].astype(str) + df_cis['Trial Number'].astype(str)

    #Take a random sample of the data
    df = df_cis.sample(n=N_ROWS, random_state=42)

    #Turn into wide table, by cell type and dff signal
    df_wide = df.pivot(index=['group', 'Centre position X', 'Centre position Y', 'frame'], columns='MatchID', values='dff').reset_index()
    df_wide.columns = list(df_wide.columns[:4]) + [f'Cell_{i}' for i in df_wide.columns[4:]]
    #df_wide = df_wide.dropna()
    df_wide = df_wide.fillna(0)
    y = df_wide[['Centre position X', 'Centre position Y']].values
    X = df_wide.iloc[:,4:].values

    groups = df_wide['group'].values
    logo = LeaveOneGroupOut()

    train_accs = []
    test_accs = []
    train_accs_shuffled = []
    test_accs_shuffled = []

    # #For each group, do the whole split, training and evaluation 
    for train_index, test_index in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pca = PCA(n_components=100)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        kmeans = KMeans(n_clusters=10, random_state=0).fit(y_train)
        y_train_kmeans = kmeans.predict(y_train)
        y_test_kmeans = kmeans.predict(y_test)
        
        #try logistic regression 
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train_pca, y_train_kmeans)
        
        y_pred_train = clf.predict(X_train_pca)
        y_pred_test = clf.predict(X_test_pca)
        
        #Evaluate
        train_accuracy = accuracy_score(y_train_kmeans, y_pred_train)
        test_accuracy = accuracy_score(y_test_kmeans, y_pred_test)
        
        test_accs.append(test_accuracy)
        train_accs.append(train_accuracy)

        #try logistic regression with shuffled labels
        # clf = LogisticRegression(random_state=42)
        #Shuffle the labels
        # np.random.shuffle(y_train_kmeans)
        # clf.fit(X_train_pca, y_train_kmeans)
        
        n_resamples = 50

        train_accs_shuffled_og = []
        test_accs_shuffled_og = []

        for i in tqdm(range(n_resamples)):
            np.random.shuffle(y_train_kmeans)
            np.random.shuffle(y_test_kmeans)
            train_accuracy = accuracy_score(y_train_kmeans, y_pred_train)
            test_accuracy = accuracy_score(y_test_kmeans, y_pred_test)
            train_accs_shuffled_og.append(train_accuracy)
            test_accs_shuffled_og.append(test_accuracy)    
                
        train_accs_shuffled.append(np.mean(train_accs_shuffled_og))
        test_accs_shuffled.append(np.mean(test_accs_shuffled_og))
        
    print("Train accuracy: ", np.mean(train_accs))
    print("Test accuracy: ", np.mean(test_accs))
    print("Test accuracy shuffled", np.mean(test_accs_shuffled))

    results = {'train_accs': train_accs, 'test_accs': test_accs, 'test_accs_shuffled': test_accs_shuffled}
    return results

results_left = train_logo(df, reward='left')
results_left