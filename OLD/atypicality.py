from __future__ import print_function
from annoy import AnnoyIndex
import random,time,itertools,os,pickle,redis,json
from scipy.misc import comb
import numpy as np
import pandas as pd
import multiprocess as mp
from tqdm import tqdm as tq
from functools import partial

from scipy.spatial.distance import cosine

r = redis.StrictRedis(host='localhost', port=6379, db=0)



f=100
t = AnnoyIndex(f)
t.load('test.ann')
N = t.get_n_items()
procs = 24

cat_indices = pickle.load(open('cat_indices.pkl','rb'))
cat_year_indices = pickle.load(open('cat_year_indices.pkl','rb'))
cat_results_year_summary = pickle.load(open('cat_results_year_summary.pkl','rb'))

def calc_dist(doc,comparison_set,parallel=False):
    if parallel:
        pool = mp.Pool(procs)
        ds = [d for d in tq(pool.imap_unordered(lambda i: convert_distance(t.get_distance(doc,i)), comparison_set,chunksize=int(len(comparison_set)/procs)),total=len(comparison_set))]
        pool.terminate()
    else:
        ds = [convert_distance(t.get_distance(doc,i)) for i in tq(comparison_set)]
    return np.array(ds)

def convert_distance(d):
    return (d**2) /2


def gen_z_score(doc_idx,current,mean,std,n,random_samples):
    if random_samples < n:
        return (mean-np.mean([convert_distance(t.get_distance(i,doc_idx)) for i in np.random.choice(current,random_samples,replace=False)]))/std
    else:
        return (mean-np.mean([convert_distance(t.get_distance(i,doc_idx)) for i in current]))/std




#result = [np.mean([convert_distance(t.get_distance(i,j)) for i in np.random.choice(current,1000,replace=False)]) for j in tq(current)]
#result = [[i for i in np.random.choice(current,1000,replace=False)][0] for j in tq(current)]
pool = mp.Pool(procs)
with open('atypicality_results','w') as out:
    for cat in tq(range(251)):
        for year in tq(range(1991,2016)):
            current = cat_year_indices[cat][year]
            mean,std,n = cat_results_year_summary[cat][year]
            n = len(current)
            if n < 1000:
                for idx in current:
                    out.write('\t'.join(map(str,[cat,year,idx,'nan']))+'\n')
            else:
                random_samples = min(1000,n)
                #result = [z for z in tq(pool.imap(lambda x: (mean-np.mean([convert_distance(t.get_distance(i,x)) for i in np.random.choice(current,random_samples,replace=False)]))/std,current,chunksize=int(n/procs)),total=len(current))]
                #pool  = mp.Pool(procs)
                func = partial(gen_z_score,current=current,mean=mean,std=std,n=n,random_samples=random_samples)
                for idx,z in tq(zip(current,pool.imap(func,current,chunksize=int(n/procs))),total=n):
                    out.write('\t'.join(map(str,[cat,year,idx,z]))+'\n')
pool.terminate()

