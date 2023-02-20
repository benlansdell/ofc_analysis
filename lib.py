import numpy as np 
from collections import defaultdict

def compute_stats(df, condition = 'Control', sessions = [2, 6, 7, 9]):
    df_con = df[df.treatment == condition]
    animals = df_con.Animal.unique() 

    cell_stats = {}
    cell_sums = {}
    cell_sums_list = {}
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
        cell_sums_list[animal] = list(cell_sums[animal].values())

    return cell_sums, cell_stats, cell_sums_list, animals

def mutual_info(x, y):
    mi = 0
    N = len(x)
    if N == 0: return 0
    assert len(x) == len(y)
    x = x.astype(int)
    y = y.astype(int)
    for i in range(2):
        for j in range(2):
            pij = sum((x == i) & (y == j))/N
            pi = sum(x == i)/N
            pj = sum(y == j)/N
            if (pij == 0) or (pi == 0) or (pj == 0): continue
            mi += pij*np.log2(pij/pi/pj)
    return mi