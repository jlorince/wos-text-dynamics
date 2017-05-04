from annoy import AnnoyIndex
import random,time,itertools,os,pickle,glob,argparse,datetime,warnings
import numpy as np
import pandas as pd
import multiprocess as mp
from tqdm import tqdm as tq
from collections import Counter

help_string="""

THIS CODE IS ONLY TESTED AGAINST PYTHON 3.6!!!

lde.py --> Local Density Estimation script

Generates estiamtes of local density in sematic space by computing the mean distance between documents and their nearest neighbors, leveraging the annoy approximate nearest neighbors library.

For each document, computes the mean distance to its top k nearest neighbors each year, as well as the temporal distribution of those top k neighbors.

We can generate (or load) three types of annoy knn indexes:
- global: Put all documents into a single global index
- global-norm: Still puts all documents in a single index, but randomly downsamples he number of documents from each year so that all years are equally represented.
- per_year: Generates an independent index for each year, stored in a separate file.


Annoy documentation: https://github.com/spotify/annoy
Doc2Vec documentation: https://radimrehurek.com/gensim/models/doc2vec.html
"""

##################################################
############ GLOBAL STATIC PARAMETERS ############
##################################################
year_range = range(1991,2016)
nyears = len(year_range)
year_dict = {y:i for i,y in enumerate(year_range)}


class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        print('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            print('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))

# annoy returns euclidean distance of normed vector
# (i.e. sqrt(2-2*cos(u, v))), so this just converts
# to standard cosine distance
def convert_distance(d):
   return (d**2) /2
vec_convert_distance = np.vectorize(convert_distance)



# computes LDE w.r.t. a reference year when index type is `per_year`
# NOT USABLE AS IS (needs new wrapper function, etc.) but leaving function in for reference
def mean_neighbor_dist_peryear(i,reference_year=None):

    year = index_years[i]
    idx = per_year_indices[i]
    
    if (reference_year is None) or (reference_year==year):
        neighbors,distances = indexes[year].get_nns_by_item(idx, args.knn+1, search_k=args.search_k, include_distances=True)
        return convert_distance(np.mean(distances[1:]))
    else:
        vec = indexes[year].get_item_vector(idx)
        neighbors,distances = indexes[reference_year].get_nns_by_vector(vec, n, search_k=search_k, include_distances=True)
        return convert_distance(np.mean(distances))

# computes LDE when index type is `global` or `global_norm`
# NOT USABLE AS IS (needs new wrapper function, etc.) but leaving function in for reference
def mean_neighbor_dist_global(i,reference_year=None):
    neighbors,distances = t.get_nns_by_item(idx, args.knn+1, search_k=args.search_k, include_distances=True)
    return convert_distance(np.mean(distances[1:]))



# computes LDE w.r.t to each year, as well as the neighbor distribution across years
# ONLY VALID when index type is `global` or `global_norm`
# query can be an integer (look up trained item) or  vector (look up held out item)
def lde (query):
    if type(query) in (np.int64,np.int32,int):
        neighbors,distances = t.get_nns_by_item(query, args.knn+1, search_k=args.search_k, include_distances=True)
    elif type(query) in (np.core.memmap,np.ndarray):
        neighbors,distances = t.get_nns_by_vector(query, args.knn, search_k=args.search_k, include_distances=True)
    else:
        raise Exception("Invalid input")
    neighbors = np.array(neighbors[1:])
    neighbor_years = index_years_sampled[neighbors]
    d_result = np.repeat(np.nan,nyears)
    n_result = np.zeros(nyears,dtype=int)
    for y in year_range:
        idx = np.where(neighbor_years==y)[0]
        year = year_dict[y]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            d_result[year] = convert_distance(np.mean([distances[i] for i in idx]))
        n_result[year] = len(idx)
    return d_result,n_result


def lde_wrapper(year):
    with open('{}results_{}'.format(result_path,year),'w') as out:
        current = np.where(index_years_sampled==year)[0]
        current_untrained = untrained[np.where(index_years[untrained]==2015)[0]]
        total = len(current)+len(current_untrained)
        done = 0
        for doc in current:
            if done%100==0:
                print("{}: {}/{} ({:.2f}%)".format(year,done,total,100*(done/total)))
                out.flush()
            d,n = lde(doc)
            out.write("{}\t{}\t{}\n".format(indices[doc],','.join(map(str,d)),','.join(map(str,n))))
            done +=1
        for doc in current_untrained:
            if done%100==0:
                print("{}: {}/{} ({:.2f}%)".format(year,done,total,100*(done/total)))
                out.flush()
            d,n = lde(features[doc])
            out.write("{}\t{}\t{}\n".format(doc,','.join(map(str,d)),','.join(map(str,n))))
            done +=1


if __name__ == '__main__':
    # chunksize=1000

    parser = argparse.ArgumentParser(help_string)
    # DEBUG LINE:
    parser.add_argument("--params", default='100-5-5',help="specify d2v model paramter in format 'size-window-min_count-sample', e.g. '100-5-5-0-None' (see gensim doc2vec documentation for details of these parameters)",type=str)
    #parser.add_argument("--params", required=True,help="specify d2v model paramter in format 'size-window-min_count-sample', e.g. '100-5-5-0-None' (see gensim doc2vec documentation for details of these parameters)",type=str)
    parser.add_argument("--index_type", help="Type of knn index to load/generate.",default='global-norm',choices=['global','global-norm','per_year'])
    parser.add_argument("--index_dir", help="Where annoy index files are located. Defaults to directory from which this script is run",default='./')
    parser.add_argument("--index_seed", help="Specify loading a random global-norm model with this seed. Only  useful if doing multiple runs with the `global-norm` option and we want to run against a particular randomly seeded model. IMPORTANT NOTE: if this arg is unspecified and running in `global-norm`, the first global-norm index found in `index_dir` will be loaded (if none exists, a new one is generated).",default=None)
    # parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=100)
    parser.add_argument("--d2vdir",help="path to doc2vec model directory",default='/backup/home/jared/storage/wos-text-dynamics-data/d2v-wos/',type=str)
    parser.add_argument("--procs",help="Specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("--knn",help="number of nearest neighbors to be used in density computations",default=1000,type=int)
    parser.add_argument("--trees",help="number of projection trees for knn index (see annoy documentation)",default=100,type=int)
    parser.add_argument("--search_k",help="search_k paramter for knn index, default = `trees`*`knn` (see annoy documentation)",default=None,type=int)
    parser.add_argument("--docs_per_year",help="number of papers to sample from each year when building `global-norm` annoy index. Deaults to number in year with least documents.",default=None,type=int)
    parser.add_argument("--result_dir",help="Output directory for results. A subfolder in this directory (named with relevant params) will be created here. Defaults to current dir.",default='./',type=str)
    args = parser.parse_args()

    if args.index_type != 'global_norm':
        raise Exception("LDE methods for {} index type are not implemented.")

    if args.search_k is None:
        args.search_k = args.trees * args.knn

    if args.index_type == 'per_year':
        mean_neighbor_dist = mean_neighbor_dist_peryear
    else:
        mean_neighbor_dist = mean_neighbor_dist_global

    result_path = args.result_dir+'_'.join([str(v) for v in [args.params,args.index_type,args.index_seed,args.knn,args.trees,args.search_k,args.docs_per_year]])+'/'
    if os.path.exists(result_path):
        raise Exception("Result directory already exists!!")
    os.mkdir(result_path)

    # index years contains the publication year for each document
    index_years = np.load(args.d2vdir+'index_years.npy')

    # If we split our index across multiple files, we need build a dict so we know
    # the year for each document (these must be zero-indexed within each year's annoy index)
    if args.index_type=='per_year':
        per_year_indices = {}
        for year in tq(year_range):
            for i,idx in tq(enumerate(np.where(index_years==year)[0])):
                per_year_indices[idx] = i

    # get dimensionality for index
    f = int(args.params.split('-')[0])

    # LOAD OR GENERATE KNN INDEX, AS APPROPRIATE
    if args.index_type=='global':

        t = AnnoyIndex(f,metric='angular')
          
        if os.path.exists(args.index_dir+'index_global.ann'):
            t.load(args.index_dir+'index.ann')
        
        else:
            if args.params.split('-')[-1] == 'None':
                features = np.load('{0}{1}/model_{1}.docvecs.doctag_syn0.npy'.format(args.d2vdir,args.params))
            else:
                features = np.load('{0}{1}/doc_features_expanded_{1}.npy'.format(args.d2vdir,args.params))
            for i,vec in tq(enumerate(features)):
                t.add_item(i, vec)
            t.build(args.trees) 
            t.save(args.index_dir+'index_global_{}.ann'.format(args.trees))
            del features
       

    elif args.index_type=='per_year':

        indexes = {}
        
        # we just check if the first year of the date range exists. If not, build all indexes
        if os.path.exists(args.index_dir+'index_{}_{}.ann'.format(year_range[0],trees)):
            for year in tq(year_range):
                t= AnnoyIndex(f,metric='angular')
                t.load('index_{}_{}.ann'.format(year,args.trees))
                indexes[year] = t

        else:
            if args.params.split('-')[-1] == 'None':
                features = np.load('{0}{1}/model_{1}.docvecs.doctag_syn0.npy'.format(args.d2vdir,args.params))
            else:
                features = np.load('{0}{1}/doc_features_expanded_{1}.npy'.format(args.d2vdir,args.params))
            for year in tq(year_range):
                current = features[np.where(index_years==year)[0]]
                t = AnnoyIndex(f,metric='angular')  
                for i,vec in tq(enumerate(current)):
                    t.add_item(i, vec)
                t.build(args.trees) 
                t.save(args.index_dir+'index_{}_{}.ann'.format(year,args.trees))
                indexes[year] = t
            del features


    elif args.index_type == 'global-norm':

        # we mmap these to facilitate parallel computations
        if args.params.split('-')[-1] == 'None':
            features = np.load('{0}{1}/model_{1}.docvecs.doctag_syn0.npy'.format(args.d2vdir,args.params),mmap_mode='r')
        else:
            features = np.load('{0}{1}/doc_features_expanded_{1}.npy'.format(args.d2vdir,args.params),mmap_mode='r')

        t = AnnoyIndex(f,metric='angular')
        existing = glob.glob('{}index_norm_{}_*.ann'.format(args.index_dir,args.trees))
        
        if len(existing)==0:
            
            indices = []
            idx = 0
            seed = np.random.randint(999999)
            print('----RANDOM SEED = {}----'.format(seed))
            np.random.seed(seed)

            if args.docs_per_year is None:
                unique_years,unique_year_counts = np.unique(index_years,return_counts=True)
                args.docs_per_year = np.argmin(np.argmin(unique_year_counts))
            
            for year in tq(year_range):
                idx_current = np.random.choice(np.where(index_years==year)[0],args.docs_per_year,replace=False)
                indices.append(idx_current)
                for vec in tq(features[idx_current]):
                    t.add_item(idx, vec)
                    idx+=1

            indices = np.concatenate(indices)
            np.save('{}index_norm_{}_{}.ann.indices'.format(args.index_dir,arg.trees,seed),indices)

            with timed('building index'):
                t.build(args.trees) 
            with timed('saving index'):
                t.save('{}index_norm_{}_{}.ann'.format(args.index_dir,arg.trees,seed))

        else:
            if args.index_seed is not None:
                try:
                    t.load('{}index_norm_{}_{}.ann'.format(args.index_dir,arg.trees,args.index_seed))
                    indices = np.load('{}index_norm_{}_{}.ann.indices.npy'.format(args.index_dir,arg.trees,args.index_seed))
                except FileNotFoundError:
                    raise Exception('You have specified an invalid seed (file does not exist)')
            else:
                t.load(existing[0])
                indices = np.load(existing[0]+'.indices.npy')

        #dict_indices = {raw_idx:i for i,raw_idx in enumerate(indices)}
        index_years_sampled = index_years[indices]
        untrained = np.delete(np.ogrid[:len(features)],indices)



    pool = mp.Pool(args.procs)
    pool.map(wrapper,year_range)
    pool.terminate()


   




    