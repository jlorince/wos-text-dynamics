from __future__ import print_function
from annoy import AnnoyIndex
import random,time,itertools,os,pickle,redis,json
from scipy.misc import comb
import numpy as np
import pandas as pd
import multiprocess as mp
#import graphlab as gl
from tqdm import tqdm as tq
from scipy.sparse import csr_matrix,csgraph
from scipy.spatial.distance import cosine


param = '100-5-5'

indexbuilt = False
n_procs = 24
chunksize = 100000

r = redis.StrictRedis(host='localhost', port=6379, db=0)


def convert_distance(d):
    return (d**2) /2
def convert_distance_reverse(d):
    return np.sqrt(2*d)






if __name__ == '__main__':

    # BUILD ANNOY INDEX

    
    f = int(param.split('-')[0])



    if not indexbuilt:

        features = np.load('/backup/home/jared/storage/wos-text-dynamics-data/d2v-wos/{0}/model_{0}.docvecs.doctag_syn0.npy'.format(param))
        total_docs,f = features.shape

        n = None
        if n:
            indices = np.random.choice(range(len(features)),n)
            sample = features[indices]
        else:
            sample = features
            #total_docs


        t = AnnoyIndex(f,metric='angular')  # Length of item vector that will be indexed
        for i,vec in tq(enumerate(sample),total=n):
            #v = [random.gauss(0, 1) for z in xrange(f)]
            t.add_item(i, vec)

        t.build(50) 
        t.save('index.ann')

         del features

    else:
        t = AnnoyIndex(f)
        t.load('index.ann')

    index_years = np.load('/backup/home/jared/storage/wos-text-dynamics-data/d2v-wos/index_years.npy')

   




    