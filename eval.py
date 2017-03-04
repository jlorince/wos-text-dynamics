import graphlab as gl
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS',24)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',24)

import ctypes, inspect, os
from ctypes import wintypes
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
src_dir = os.path.split(inspect.getfile(gl))[0]
kernel32.SetDllDirectoryW(src_dir)

#prefix = "P:/Projects/"
prefix = "E:/Users/jjl2228/"

import numpy as np
import gzip

if True: # this only needs to be handled once
    category_data = gl.SFrame()
    for year in range(1991,2016):
        current = gl.SFrame.read_csv(prefix+"WoS/WoS/parsed/subjects/{}.txt.gz".format(year),header=None,delimiter='\t').dropna(columns=['X4'])
        current['year'] = year
        current['X4'] = current['X4'].apply(lambda x: x.split("|"))
        current['first_cat'] = current['X4'].apply(lambda x: x[0])
        category_data = category_data.append(current)

    category_data = category_data.rename({"X1":"uid","X2":'heading',"X3":"subheading","X4":'categories'})
    category_data.save(prefix+"WoS/WoS/parsed/abstracts/d2v/categories.sframe")

else:
    category

features = np.load(prefix+"WoS/WoS/parsed/abstracts/features_normed-w2v-200.npy")
features = gl.SFrame(features)
features.rename({'X1':'features'})
features['uid'] = [line.strip() for line in gzip.open(prefix+"WoS/WoS/parsed/abstracts/uid_indices.txt.gz").readlines()]

joined = features.join(metadata,on='uid')

viz_df = joined[#['uid','heading','subheading','categories','first_cat','year']].to_dataframe()