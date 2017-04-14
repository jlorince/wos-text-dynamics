from __future__ import print_function
from annoy import AnnoyIndex
import random,time,itertools,os,pickle,redis,json,glob
from scipy.misc import comb
import numpy as np
import pandas as pd
import multiprocess as mp
#import graphlab as gl
from tqdm import tqdm as tq
from scipy.sparse import csr_matrix,csgraph
from scipy.spatial.distance import cosine


indexbuilt = False
n_procs = 24
chunksize = 100000

r = redis.StrictRedis(host='localhost', port=6379, db=0)


def convert_distance(d):
    return (d**2) /2
def convert_distance_reverse(d):
    return np.sqrt(2*d)


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

def setup_cats(idx_dict):
    cat_dict = pickle.load(open('cat_dict_complete.pkl','rb'))
    cat_indices = {i:[] for i in range(251)}
    n_bad = 0 
    for uid,i in tq(idx_dict.items()):
        try:
            v = cat_dict[uid]
        except KeyError:
            n_bad += 1
            continue
        r.set(i,json.dumps(v))
        cats = v[2]
        for c in cats:
            try:
                cat_indices[c].append(i)
            except KeyError:
                continue
    for c in tq(cat_indices):
        cat_indices[c] = np.array(cat_indices[c])

    return cat_indices

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




if __name__ == '__main__':

    # BUILD ANNOY INDEX
    idx_dict = {}
    indices = []
    i=0
    idx=0
    for fi in tq(sorted(glob.glob('indices_all/idx_*'))):
        for line in open(fi):
            uid,el_id = line.strip().split(',')
            if uid != '':
                 idx_dict[uid] = idx
                 indices.append(i)
                 idx+=1
            i+=1
    #features = np.load('model_100-5-5.docvecs.doctag_syn0.npy')
    features = np.load('model_100-10-100.docvecs.doctag_syn0.npy')
    features = features[np.array(indices)]
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

        t.build(10) # 10 trees
        t.save('test.ann')

        cat_indices = setup_cats(idx_dict)
        pickle.dump(cat_indices,open('cat_indices.pkl','wb'))
    else:
        t = AnnoyIndex(f)
        t.load('test.ann')
        cat_indices = pickle.load(open('cat_indices.pkl','rb'))

    del features

    

    pool = mp.Pool(24)
    %time acc =  pool.map(random_comps_acc,itertools.repeat(1000000,n_procs))
    acc_df = pd.concat(acc)
    cat_results = {}
    for cat in tq(range(251)):
        possible_comps = comb(len(cat_indices[cat]),2)
        if possible_comps <= (chunksize*n_procs):
            result = pool.map(lambda a: t.get_distance(a[0],a[1]),itertools.combinations(cat_indices[cat],2))
            #cat_results[cat] = np.histogram(result,bins=np.arange(0,2,.01))[0]
            cat_results[cat] = np.array(result)
        else:
            result = pool.map(incat_similarity,itertools.repeat((cat,chunksize),n_procs))
            cat_results[cat] = np.array(list(itertools.chain(*result)))
    cat_results_binned = {}
    for cat in range(251):
        cat_results_binned[cat] = np.histogram(cat_results[cat],bins=np.arange(0,2,.01))[0]


    pool.terminate()

    pickle.dump({c:pd.Series(cat_results[c]).describe() for c in cat_results},open('cat_results.pkl','wb'))
    pickle.dump(cat_results_binned,open('cat_results_binned.pkl','wb'))
    acc_df.to_pickle('acc_df.pkl')

