import pandas as pd 
import pickle 
import os

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.lib import run_decoder

fn_in = './data/processed/full2.csv'
out_dir = './data/output/'

df = pd.read_csv(fn_in, dtype = {'Animal': str, 
                                 'UnSpout_pos':str,
                                 'UnSpout_neg':str,
                                 'UnMid_pos':str,
                                 'UnMid_neg':str, 
                                 'Session': str, 
                                 'Trial': str})
df = df[['Animal', 'Session', 'CellID', 'RewardStatus', 'Trial', 'recording',
         'Reward', 'MatchID', 'treatment', 'resp', 'uID', 'variable', 'value']]

sessions = ['2', '6', '7', '9']
conditions = pd.unique(df['treatment'])
reward_statuses = pd.unique(df['RewardStatus'])

label_key = {'reward_spout_events': 1, 'unreward_spout_events': 0}

pca = PCA(n_components = 20)
scaler = StandardScaler()
lda = LinearDiscriminantAnalysis()

lda_classifier = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("lda",lda)])
param_grid = {"pca__n_components": [15]}
time_pairs = (201,240)
test_pairs = (0, 500)
classifiers = [lda_classifier]
grids = [param_grid]

os.makedirs(out_dir, exist_ok=True)

def run_day(train_sessions, test_session, test_pairs = (0, 500), note = ''):

    test_idx = sessions.index(test_session)
    train_indices = [sessions.index(s) for s in train_sessions]

    _, dayresults, results = run_decoder(df, 
                                         conditions, 
                                         sessions, 
                                         label_key, 
                                         time_pairs, 
                                         test_pairs, 
                                         classifiers, 
                                         grids, 
                                         train_sessions, 
                                         test_session, 
                                         test_idx, 
                                         train_indices=train_indices)

    fn_out = os.path.join(out_dir, f'decoder_day{test_session}_results{note}.csv')
    results[0].to_csv(fn_out)

    with open(os.path.join(out_dir, f'day{test_session}results{note}.pkl'), 'wb') as f:
        pickle.dump(dayresults, f)

#run_day(['2', '6'], '7')
#run_day(['2', '6'], '9')
#run_day(['2', '6'], '6')

run_day(['2', '6'], '7', test_pairs = (201, 240), note = '_shorttest')
run_day(['2', '6'], '9', test_pairs = (201, 240), note = '_shorttest')