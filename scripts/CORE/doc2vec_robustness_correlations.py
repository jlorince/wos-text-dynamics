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
    '50-5-5-0': 'doc_features_normed_50-5-5-0.npy',\
    '50-5-5-9-0.00005' : 'doc_features_normed_50-5-5.npy',\
    '100-5-5-0': 'doc_features_normed.npy',\
    '200-5-5-0': 'doc_features_normed_200-5-5-0.npy',\
    '200-5-5-0.00005' : 'doc_features_normed_200-5-5.npy',\
    '300-5-5-0.00005': 'doc_features_normed_300-5-5.npy'
}
N =  22606989 # number of elements (fixed)

## parameters
N_pairs = 10**2 # howmany pairs to compare
# n_seed = 10
metric = 'cosine' # which metric to use, default: cosine

## select 2 models

list_models = sorted(list(dict_model_fname.keys()))
print(list_models)

for i_m1,model1 in enumerate(list_models):
    for i_m2,model2 in enumerate(list_models):
        if i_m1>i_m2:

            # model1 = '100-5-5-0'
            # model2 = '50-5-5-0'

            filename_save = 'doc2vec_m%s_m%s_comparison_distances_%s_N%s'\
                            %(model1,model2,metric,str(N_pairs))

            ## read vectors in mempory-mapped mode

            path_read = os.path.join(path_data,model1)
            fname_read = dict_model_fname[model1]
            filename = os.path.join(path_read,fname_read)
            x1 = np.load(filename,mmap_mode='r')

            path_read = os.path.join(path_data,model2)
            fname_read = dict_model_fname[model2]
            filename = os.path.join(path_read,fname_read)
            x2 = np.load(filename,mmap_mode='r')


            i1,i2=0,0


            with open(filename_save,'w') as f:
                for i in range(N_pairs):
                    i1,i2=0,0
                    while i1 == i2:
                        i1,i2 = np.random.randint(N,size=2)#np.random.choice(N,size=2,replace=False)
                        
                    ## distance in dataset 1
                    vec1,vec2 = x1[i1],x1[i2]
                    s1 = pdist([vec1,vec2],metric=metric)[0]

                    ## distance in dataset 2
                    vec1,vec2 = x2[i1],x2[i2]
                    s2 = pdist([vec1,vec2],metric=metric)[0]

                    f.write('%s \t %s \n'%(str(s1),str(s2)))