from __future__ import print_function
from annoy import AnnoyIndex
import random,time,itertools,os,pickle,redis,json
import numpy as np
import pandas as pd
import multiprocess as mp
#import graphlab as gl
from tqdm import tqdm as tq
from scipy.sparse import csr_matrix,csgraph
from scipy.spatial.distance import cosine
from functools import reduce

r = redis.StrictRedis(host='localhost', port=6379, db=0)

def convert_distance(d):
    return (d**2) /2
def convert_distance_reverse(d):
    return np.sqrt(2*d)


def get_top_neighbors(idx,prnt):
    endfound = False
    maxn = 1000
    while not endfound:
        #print('running for {}'.format(maxn))
        nn,ds = u.get_nns_by_item(idx, maxn,include_distances=True)
        if convert_distance(ds[-1])>maxdist:
            endfound = True
        else:
            if maxn<1000000:
                maxn *= 10
            else:
                maxn += 1000000
    y_result = []
    d_result = []
    for n,d in zip(nn[1:],ds[1:]):
        d = convert_distance(d)
        if d> maxdist:
            break
        y_result.append(n)
        d_result.append(d)
    final = len(y_result)
    print('{},{}'.format(prnt,idx,final))
    return [idx]*final, y_result, d_result
    #return csr_matrix((d_result,([idx]*final, y_result)),shape=(5000000,5000000))

def combiner(idx_range):
    x = []
    y = []
    d = []
    for i,idx in enumerate(idx_range):
        x_,y_,d_ = get_top_neighbors(idx,prnt=i)
        x+=x_
        y+=y_
        d+=d_

    return csr_matrix((d,(x,y)),shape=(5000000,5000000))
        


def random_comp(n):
    result = []
    for i in range(n):
        if i%1000==0:
            print("{}/{} ({})".format(i,n,os.getpid()))
        a = np.random.randint(0,features.shape[0])
        b = np.random.randint(0,features.shape[0])
        result.append(convert_distance(t.get_distance(a,b)))
    return result

def sample_graph(n_samples):
    pid = os.getpid()
    np.random.seed(int(time.time()/1000)+pid)
    data = []
    x = []
    y = []
    xy = set()
    for i in range(n_samples):
        if i%100000==0:
            print("{}/{} ({})".format(i,n_samples,pid))
        a = np.random.randint(0,n)
        b = np.random.randint(0,n)
        if a>b:
            a,b = b,a
        if (a,b) in xy:
            continue
        # USE SIMILARITY NOT DISTANCE
        # this is because "<" comparisons are inefficient on sparse graphs
        d = 2-convert_distance(t.get_distance(a,b))
        data.append(d)
        x.append(a)
        y.append(b)

    return csr_matrix((data,(x,y)),shape=(n,n))

# deal with distance conversions
def snowball(seed,k=11,thresh=0.1,initial_seed=None):
    if initial_seed is None:
        initial_seed = seed
    nn,ds = t.get_nns_by_item(seed,k,include_distances=True)
    nn = nn[1:]
    ds = ds[1:]
    if t.get_distance(nn[-1],initial_seed)>=thresh:
        nn = [nbr for nbr in nn if t.get_distance(initial_seed,nbr)]
        ds = ds[:len(nn)]
        return nn,ds
    else:
        return reduce(lambda x,y: x+y,[snowball(seed=nbr,k=k,thresh=thresh,initial_seed=initial_seed) for nbr in nn])


def snowball(seed,k=10,sample_k=10,thresh=0.1):
    get = [seed]
    nodes_done = set()
    edges_done = set()
    data = []
    x = []
    y = []
    passes = 1
    while get:
        #print(len(get),passes)
        current = get.pop(0)
        nn,ds = t.get_nns_by_item(current,k+1,include_distances=True)
        #print(["{:.2f}".format(convert_distance(dis)) for dis in ds[1:]])
        #print(nn,ds)
        #random_indices = np.random.choice(range(1,k),sample_k,replace=False)
        #for n,d in zip(nn[1:sample_k+1],ds[1:sample_k+1]):
        #for n,d in zip(np.array(nn)[random_indices],np.array(ds)[random_indices]):
        for n,d in zip(nn[1:],ds[1:]):
            d = convert_distance(d)
            seed_distance = convert_distance(t.get_distance(n,seed))
            #print(seed_distance)
            if seed_distance>=thresh:
                continue
            if n in nodes_done:
                continue
            if d<thresh:
                if n>current:
                    n,current = current,n
                if (current,n) in edges_done:
                    continue
                data.append(d)
                x.append(current)
                y.append(n)
                get.append(n)
                edges_done.add(current)
        nodes_done.add(current)
        passes += 1
    #print(len(data))
    #return csr_matrix((data,(x,y)),shape=(total_docs,total_docs))
    return data,x,y


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
                    #h_a,sh_a,cats_a = cat_dict[a]
                    h_a,sh_a,cats_a = json.loads(r.get(a))
                    nca = len(cats_a)
                    #h_b,sh_b,cats_b = cat_dict[b]
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
                #first_cat_agreement.append(int(cats_a[0]==cats_b[0]))
                first_cat_agreement.append(int(intersection>0))
                cat_overlap.append( intersection / min(nca,ncb))
                #cat_overlap.append(len(set(cats_a).intersection(cats_b)) / min(nca,ncb))
                done += 1
    return pd.DataFrame({'a':a_,'b':b_,'dist':ds,'overlap':cat_overlap,'agree':first_cat_agreement,'cats_a':cats_a_,'cats_b':cats_b_})

def accuracy(tup):
    ds,comps = tup
    # heading, subheading, categories
    first_cat_agreement = []
    cat_overlap = []
    n = len(ds)
    for i,(d,(a,b)) in enumerate(zip(ds,comps)):
        if i%10000==0:
            print("{}/{} ({})".format(i,n,os.getpid()))
        try:
            #h_a,sh_a,cats_a = cat_dict[a]
            h_a,sh_a,cats_a = json.loads(r.get(a))
            nca = len(cats_a)
            #h_b,sh_b,cats_b = cat_dict[b]
            h_b,sh_b,cats_b = json.loads(r.get(b))
            ncb = len(cats_b)
            intersection = len(set(cats_a).intersection(cats_b))
            #first_cat_agreement.append(int(cats_a[0]==cats_b[0]))
            first_cat_agreement.append(int(intersection>0))
            cat_overlap.append( intersection / min(nca,ncb))
        except TypeError: # if no result in DB
            first_cat_agreement.append(np.nan)
            cat_overlap.append(np.nan)
            continue

    a,b = zip(*comps)

    result = pd.DataFrame({'a':a,'b':b,'dist':ds,'overlap':cat_overlap,'agree':first_cat_agreement})
    return result

def nn(seed,n=100):
    neighbors = t.get_nns_by_item(seed,100,include_distances=True)




if __name__ == '__main__':

    # BUILD ANNOY INDEX



    features = np.load('model_100-5-5.npy.docvecs.doctag_syn0.npy')
    total_docs,f = features.shape


    if False:
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
    else:
        t = AnnoyIndex(f)
        t.load('test.ann')

    del features
    #cat_dict = pickle.load(open('cat_dict.pkl','rb'))

    pool = mp.Pool(24)
    #result = pool.map(random_comps,itertools.repeat(100000,24))
    #acc = pool.map(accuracy,result)#,chunksize=2)
    %time acc =  pool.map(random_comps_acc,itertools.repeat(100000,24))
    pool.terminate()
    # for r in result:
    #     new = []
    #     for comp in r[1]:



    final = pd.concat(acc)

    def first_cat_tuple(row):
        a = row.cats_a[0]
        b = row.cats_b[0]
        if not pd.isnull(a) and not pd.isnull(b) and a>b:
            a,b = b,a
        return a,b
    #acc = pool.map(accuracy,result)
    #pool.terminate()


    # pool = mp.Pool(24)
    # result = pool.map(sample_graph,[10000000]*24)
    # for thresh in np.arange(0.1,0.91,0.1):
    #     combined = reduce(lambda x,y: (x>thresh)+(y>thresh),result)
    #     n_c,labels = csgraph.connected_components(combined)

    #     unique, counts = np.unique(labels, return_counts=True)



# ...

# u = AnnoyIndex(f)
# u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

