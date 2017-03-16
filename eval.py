
prefix,workers = {'p':('P:/Projects/',60),'e':("E:/Users/jjl2228/",24)}['p']


import graphlab as gl
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS',workers)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',workers)

import ctypes, inspect, os
from ctypes import wintypes
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
src_dir = os.path.split(inspect.getfile(gl))[0]
kernel32.SetDllDirectoryW(src_dir)



import numpy as np
import gzip
import pandas as pd

if False: # this only needs to be handled once
    category_data = gl.SFrame()
    for year in range(1991,2016):
        current = gl.SFrame.read_csv(prefix+"WoS/WoS/parsed/subjects/{}.txt.gz".format(year),header=None,delimiter='\t').dropna(columns=['X4'])
        current['year'] = year
        current['X4'] = current['X4'].apply(lambda x: x.split("|"))
        current['first_cat'] = current['X4'].apply(lambda x: x[0])
        category_data = category_data.append(current)

    category_data = category_data.rename({"X1":"uid","X2":'heading',"X3":"subheading","X4":'categories'})

    uid_indices = gl.SFrame(prefix+"WoS/WoS/parsed/abstracts/d2v/uid_indices/*.txt.gz")
    uid_indices = uid_indices.rename({'X1':'uid'}).add_row_number()

    joined = category_data.join(uid_indices,on='uid')

    oecd = pd.read_csv(prefix+'WoS/WoS/data/oecd.csv')
    oecd = oecd.groupby('WoS_Description').apply(lambda df: pd.Series({'level1':df.iloc[0].Description,'level2':df.iloc[1].Description})).reset_index()
    oecd['WoS_Description'] = oecd['WoS_Description'].str.title()
    oecd.columns = ['first_cat','oecd_1','oecd_2']

    joined.join(gl.SFrame(oecd),on='first_cat',how='left').save(prefix+"WoS/WoS/parsed/abstracts/d2v/categories.sframe")

    df = joined.to_dataframe()


else:
    #joined = gl.SFrame(prefix+"WoS/WoS/parsed/abstracts/d2v/categories.sframe")
    df = pd.read_pickle(prefix+"WoS/WoS/parsed/abstracts/d2v/categories.pkl")


model_params = '100-5-5'
features = np.load(prefix+"WoS/WoS/parsed/abstracts/d2v/{0}/model_{0}.npy.docvecs.doctag_syn0.npy".format(model_params))
#features = np.load(prefix+"WoS/WoS/parsed/abstracts/d2v/{0}/doc_features_normed_{0}.npy".format(model_params))

for idx in range(100):

    df_sampled = df.sample(n=1000000)
    df_sampled['features'] = df_sampled['id'].apply(lambda x: features[x])
    sf = gl.SFrame(df_sampled)

    nn = gl.nearest_neighbors.create(sf,features=['features'],distance='cosine',method='lsh')

    thresh = 0.9
    graph = nn.similarity_graph(k=None,radius=thresh)
    while True:
        cc = gl.connected_components.create(graph)

        thresh -= 0.1
        if thresh < 0.1:
            break
        graph = gl.SGraph(edges=graph.edges[graph.edges['distance']<thresh],vertices=graph.vertices)



# features = gl.SFrame(features[:100000])
# features = features.rename({'X1':'features'}).add_row_number()

final = test.join(features,on='id')
#final.save(prefix+"WoS/WoS/parsed/abstracts/d2v/{}/categories.sframe".format(model_params))
    nn = gl.nearest_neighbors.create(final,features=['features'],distance='cosine')

graph = nn.similarity_graph(k=None,radius=0.9)

