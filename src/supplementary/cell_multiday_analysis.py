#%%
import os 
import pandas as pd
import numpy as np
from itertools import combinations, product
from collections import defaultdict

sessions = [2, 6, 7, 9]

#%%
fn_in = './data/input/explore.csv'
df = pd.read_csv(fn_in)

#%%
#Start with control recordings only
df_con = df[df.treatment == 'Control']
# %%
#df_con.groupby('Animal')['Session'].unique()
animals = df_con.Animal.unique() 

cell_stats = {}
cell_sums = {}
for animal in animals:
    cells = defaultdict(lambda: np.zeros(4))
    sums = {}
    for idx, session in enumerate(sessions):
        m_ids = df_con.loc[(df_con.Animal == animal) & (df_con.Session == session), 'MatchID'].unique()
        for c in m_ids:
            cells[c][idx] = 1
    for c in cells.keys():
        sums[c] = cells[c].sum()
    cell_stats[animal] = cells
    cell_sums[animal] = sums

#%%
sums_list = list(cell_sums[1].values())

#Now, for each pair or recordings, what proportion of cells are matched between the recs?
animal = 1
animal_stats = np.stack(cell_stats[animal].values())
matching_matrix = np.zeros((4,4))

#Of all cells present in s1, how many are present in s2?
for s1, s2 in product(range(4), range(4)):
    a_stats = animal_stats[animal_stats[:,s1] == 1, s2]
    matching_matrix[s1, s2] = a_stats.sum()/a_stats.shape[0]
