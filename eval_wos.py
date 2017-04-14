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


indexbuilt = True
n_procs = 24
chunksize = 100000

r = redis.StrictRedis(host='localhost', port=6379, db=0)


def convert_distance(d):
    return (d**2) /2
def convert_distance_reverse(d):
    return np.sqrt(2*d)

def random_comps(n):
    pid = os.getpid()
    np.random.seed(int(time.time()/1000)+pid)
    ds = []
    comps = []
    compset = set()
    done = 0
    while done<n:
        if done%10000==0:
            print("{}/{} ({})".format(done,n,os.getpid()))
        a = np.random.randint(0,total_docs)
        b = np.random.randint(0,total_docs)
        if a!=b:
            if a>b:
                a,b = b,a
            if (a,b) not in compset:
                ds.append(convert_distance(t.get_distance(a,b)))
                compset.add((a,b))
                comps.append((a,b))
                done += 1
    return ds,comps

def random_comps_acc(n):
    pid = os.getpid()
    np.random.seed(int(time.time()/1000)+pid)
    ds = []
    a_ = []
    b_  = []
    cats_a_ = []
    cats_b_ = []
    compset = set()
    first_cat_agreement = []
    cat_overlap = []
    done = 0
    while done<n:
        if done%10000==0:
            print("{}/{} ({})".format(done,n,os.getpid()))
        a = np.random.randint(0,total_docs)
        b = np.random.randint(0,total_docs)
        if a!=b:
            if a>b:
                a,b = b,a
            if (a,b) not in compset:
                try:
                    h_a,sh_a,cats_a = json.loads(r.get(a))
                    nca = len(cats_a)
                    h_b,sh_b,cats_b = json.loads(r.get(b))
                    ncb = len(cats_b)
                except TypeError: # if no result in DB
                    continue
                ds.append(convert_distance(t.get_distance(a,b)))
                compset.add((a,b))
                a_.append(a)
                b_.append(b)
                cats_a_.append(cats_a)
                cats_b_.append(cats_b)
                intersection = len(set(cats_a).intersection(cats_b))
                first_cat_agreement.append(int(intersection>0))
                cat_overlap.append( intersection / min(nca,ncb))
                done += 1
    return pd.DataFrame({'a':a_,'b':b_,'dist':ds,'overlap':cat_overlap,'agree':first_cat_agreement,'cats_a':cats_a_,'cats_b':cats_b_})


def nn(seed,n=100):
    neighbors = t.get_nns_by_item(seed,100,include_distances=True)

def setup_cats(setup_redis=False):
    cat_dict = pickle.load(open('cat_dict.pkl','rb'))
    year_dict = pickle.load(open('year_dict.pkl','rb'))
    cat_indices = {i:[] for i in range(251)}
    cat_year_indices = {}
    for i in range(251):
        cat_year_indices[i] = {y:[] for y in range(1991,2016)}
    for k,v in tq(cat_dict.items()):
        if setup_redis:
            r.set(k,json.dumps(v))
        cats = v[2]
        year = year_dict[k]
        for c in cats:
            try:
                cat_indices[c].append(k)
                cat_year_indices[c][year].append(k)
            except KeyError:
                continue
    for c in tq(cat_indices):
        cat_indices[c] = np.array(cat_indices[c])
    for c in tq(cat_year_indices):
        for y in tq(cat_year_indices[c]):
            cat_year_indices[c][y] = np.array(cat_year_indices[c][y])
    return cat_indices,cat_year_indices

def incat_similarity(tup):
    cat_idx,n = tup
    indices = cat_indices[cat_idx]
    pid = os.getpid()
    np.random.seed(int(time.time()/1000)+pid)
    ds = []
    compset = set()
    done = 0
    while done<n:
        #if done%10000==0:
        #    print("{}/{} ({})".format(done,n,os.getpid()))
        a = np.random.choice(indices)
        b = np.random.choice(indices)
        if a!=b:
            if a>b:
                a,b = b,a
            if (a,b) not in compset:
                ds.append(convert_distance(t.get_distance(a,b)))
                compset.add((a,b))
                done += 1
    return ds

def incat_similarity_year(tup):
    cat_idx,year,n = tup
    indices = cat_year_indices[cat_idx][year]
    pid = os.getpid()
    np.random.seed(int(time.time()/1000)+pid)
    ds = []
    compset = set()
    done = 0
    while done<n:
        #if done%10000==0:
        #    print("{}/{} ({})".format(done,n,os.getpid()))
        a = np.random.choice(indices)
        b = np.random.choice(indices)
        if a!=b:
            if a>b:
                a,b = b,a
            if (a,b) not in compset:
                ds.append(convert_distance(t.get_distance(a,b)))
                compset.add((a,b))
                done += 1
    return ds




if __name__ == '__main__':

    # BUILD ANNOY INDEX

    features = np.load('model_100-5-5.npy.docvecs.doctag_syn0.npy')
    total_docs,f = features.shape



    if not indexbuilt:
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

        t.build(50) # 10 trees
        t.save('test.ann')
    else:
        t = AnnoyIndex(f)
        t.load('test.ann')

    del features

    cat_indices,cat_year_indices = setup_cats()

    # pool = mp.Pool(24)
    # %time acc =  pool.map(random_comps_acc,itertools.repeat(chunksize,n_procs))
    # acc_df = pd.concat(acc)
    # cat_results = {}
    # for cat in tq(range(251)):
    #     possible_comps = comb(len(cat_indices[cat]),2)
    #     if possible_comps <= (chunksize*n_procs):
    #         result = pool.map(lambda a: t.get_distance(a[0],a[1]),itertools.combinations(cat_indices[cat],2))
    #         #cat_results[cat] = np.histogram(result,bins=np.arange(0,2,.01))[0]
    #         cat_results[cat] = np.array(result)
    #     else:
    #         result = pool.map(incat_similarity,itertools.repeat((cat,chunksize),n_procs))
    #         cat_results[cat] = np.array(list(itertools.chain(*result)))
    # cat_results_binned = {}
    # for cat in range(251):
    #     cat_results_binned[cat] = np.histogram(cat_results[cat],bins=np.arange(0,2,.01))[0]

    # pool.terminate()


    ## THIS SHOULD MOVE TO NEW SCRIPT BUT STAYS HERE FOR NOW
    #chunksize = 10000
    sample_size = 500
    procs = 24
    pool = mp.Pool(procs)
    cat_results_year = {i:{} for i in range(251)}
    for cat in tq(range(251)):
        for year in tq(range(1991,2016)):
            possible_comps = comb(len(cat_year_indices[cat][year]),2)
            #if possible_comps <= (chunksize*n_procs):

            #np.random.shuffle(pairs)
            #result = [r for r in tq(pool.imap(lambda a: convert_distancet.get_distance(a[0],a[1])),itertools.combinations(cat_year_indices[cat][year],2)),total=possible_comps)]
            if len(cat_year_indices[cat][year])<=sample_size:
                chunksize = int(comb(len(cat_year_indices[cat][year]),2)/procs)
                pairs = itertools.combinations(cat_year_indices[cat][year],2)
                result = [r for r in tq(pool.imap(lambda a: convert_distance(t.get_distance(a[0],a[1])),pairs))]
            else:
                pairs = itertools.combinations(np.random.choice(cat_year_indices[cat][year],sample_size,replace=False),2)
                chunksize = int(comb(1500,2)/procs)
                result = [r for r in tq(pool.imap(lambda a: convert_distance(t.get_distance(a[0],a[1])),pairs,chunksize=chunksize))]
            del pairs
            cat_results_year[cat][year] = np.array(result)
            # else:
            #     result = pool.map(incat_similarity_year,itertools.repeat((cat,year,chunksize),n_procs))
            #     cat_results_year[cat][year] = np.array(list(itertools.chain(*result)))
    cat_results_year_binned = {i:{} for i in range(251)}
    for cat in tq(range(251)):
        for year in tq(range(1991,2016)):
            cat_results_year_binned[cat][year] = np.histogram(cat_results_year[cat][year],bins=np.arange(0,2,.01))[0]

    pickle.dump(cat_results_year_binned,open('cat_results_year_binned.pkl','wb'))
    cat_results_year_summary = {i:{} for i in range(251)}
    for cat in tq(cat_results_year):
        for year in tq(cat_results_year[cat]):
            cat_results_year_summary[cat][year] = (cat_results_year[cat][year].mean(),cat_results_year[cat][year].std(),len(cat_results_year[cat][year]))
    pickle.dump(cat_results_year_summary,open('cat_results_year_summary.pkl','wb'))

