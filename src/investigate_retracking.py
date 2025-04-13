#Set to project root directory
import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import argparse
import pickle 

from glob import glob
from tqdm import tqdm
from itertools import product

from src.investigate_retracking_lib import train_one_side_test_other, load_animal, train_logo

sample_dirs = ['/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal 1',
        '/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal 10',
        '/home/blansdel/projects/schwarz/decoder/Retracked/Control Animal M3',
        '/home/blansdel/projects/schwarz/decoder/Retracked/Clonidine Animal F4',
        '/home/blansdel/projects/schwarz/decoder/Retracked/IDX Animal F3']

training_reward_by_animal = {
    'Control Animal 1': 'left',
    'Control Animal 10': 'right',
    'Control Animal M3': 'left',
    'IDX Animal F3': 'left',
    'Clonidine Animal F4': 'left'
}

output_dir = './retracked_results/'

available_animals = [t.replace(' ', '_') for t in training_reward_by_animal.keys()]

def main(args):

    animal = args.animal
    training_reward = training_reward_by_animal[animal]

    files = []
    for d in sample_dirs:
        files.extend(glob(d + "/*.Rds"))

    animal_to_files = {}
    for fn in files:
        animal = fn.split('/')[-2]
        if animal not in animal_to_files:
            animal_to_files[animal] = []
        animal_to_files[animal].append(fn)

    df = load_animal(animal, animal_to_files)

    print(f"Training on {training_reward}, testing on the other side")
    results_train_left_test_right = train_one_side_test_other(df, {'n_pcs': 10, 'n_clusters': 10}, train_reward=training_reward, do_shuffling=True)

    fn_out = os.path.join(output_dir, f'results_animal_{animal}.pkl')
    with open(fn_out, 'wb') as f:
        pickle.dump(results_train_left_test_right, f)

    #Do a hyperparameter search over n_pcs and n_clusters
    if args.runhyperparamsearch:
        hyper_params = {'n_pcs': [10, 50, 100], 'n_clusters': [5, 10, 20, 50]}
        hyper_param_results = {}
        for n_pcs, n_clusters in tqdm(product(hyper_params['n_pcs'], hyper_params['n_clusters']), total = len(hyper_params['n_pcs']) * len(hyper_params['n_clusters'])):
            print(f"Training on left, testing on right, n_pcs = {n_pcs}, n_clusters = {n_clusters}")
            hyper_params = {'n_pcs': n_pcs, 'n_clusters': n_clusters}
            hyper_param_results[(n_pcs, n_clusters)] = train_one_side_test_other(df, hyper_params, train_reward=training_reward, do_shuffling = False)

        fn_out = os.path.join(output_dir, f'results_animal_{animal}_hyperparam_search.pkl')
        with open(fn_out, 'wb') as f:
            pickle.dump(hyper_param_results, f)

    for key, value in hyper_param_results.items():
        print(key, f"test acc: {value['test_accs']}, train acc: {value['train_accs']}")

    if args.runlogo:
        results_left = train_logo(df, reward=training_reward)
        fn_out = os.path.join(output_dir, f'results_animal_{animal}_logo.pkl')
        with open(fn_out, 'wb') as f:
            pickle.dump(results_left, f)

parser = argparse.ArgumentParser(description='Run retracked decoder')
parser.add_argument('animal', type=str, nargs='+',
                    help='Animal to analyze')
parser.add_argument('--runlogo', action='store_true',
                    default=False,
                    help='Run leave one group out validation')
parser.add_argument('--runhyperparamsearch', action='store_true',
                    default=False,
                    help='Run hyperparameter search')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)