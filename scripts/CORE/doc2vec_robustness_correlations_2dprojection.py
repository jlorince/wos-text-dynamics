'''Running and saving results explored in 
   doc2vec_robustness_correlations.ipynb
'''


import numpy as np
import os,sys
import pandas as pd
import glob
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist,cdist


## add src-path
# src_dir = os.path.abspath(os.path.join(os.pardir,os.pardir,'src'))
# sys.path[0] = src_dir

path_data = os.path.abspath('/scratch2/gerlach/Dropbox (Uzzi Lab)/WoS/wos-text-dynamics-data/d2v-wos')


## mapping of model and the corresponding filename

dict_model_fname = {
    '100-5-5-0': \
    ['doc_features_normed_100-5-5.npy.1M.lv_coords',\
    'doc_features_normed_100-5-5.npy.indices_1M',\
    'model_100-5-5.npy.docvecs.doctag_syn0.npy']
}

## parameters
N =  10**6 ## number of subsampled vectors (fixed to 1M at the moment)

N_pairs = 10**6 # howmany pairs to compare
# n_seed = 10
metric = 'euclidean' # which metric to use, default: cosine

## select 2 models

list_models = sorted(list(dict_model_fname.keys()))
print(list_models)

for i_m,model in enumerate(list_models):


    filename_save = 'doc2vec_2dproj_m%s_comparison_distances_%s_1M_N%s'\
                    %(model,metric,str(N_pairs))


    ## the 2D-projections - vectors
    ## these are 1M subsampled vectors
    path_read = os.path.join(path_data,model)
    fname_read = dict_model_fname[model][0]
    filename = os.path.join(path_read,fname_read)
    x= (np.loadtxt(filename))
    x_2D = x[1:,:]

    ## get corresponding indices in orginal dataset
    fname_read = dict_model_fname[model][1]
    filename = os.path.join(path_read,fname_read)
    with open(filename) as f:
        x=f.readlines()
    x_inds = [int(h) for h in x[0].split(',')]


    ## memory-map the original vectors
    fname_read = dict_model_fname[model][2] ## these are the normed vectors
    filename = os.path.join(path_read,fname_read)
    x1 = np.load(filename,mmap_mode='r')

    with open(filename_save,'w') as f:
        # np.random.seed(n_seed)
        i1,i2=0,0


        for i in range(N_pairs):
            i1,i2=0,0
            while i1 == i2:
                i1,i2 = np.random.randint(N,size=2)#np.random.choice(N,size=2,replace=False)
                
            ## distance in dataset 1
            vec1,vec2 = x_2D[i1],x_2D[i2]
            s1 = pdist([vec1,vec2],metric=metric)[0]
            
            ## distance in dataset 2
            vec1,vec2 = x1[x_inds[i1]],x1[x_inds[i2]]
            s2 = pdist([vec1,vec2],metric=metric)[0]

            # write to file
            f.write('%s \t %s \n'%(str(s1),str(s2)))