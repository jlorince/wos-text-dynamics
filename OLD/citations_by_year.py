from __future__ import print_function
import pandas as pd
import numpy as np
from tqdm import tqdm as tq
import multiprocessing as mp
from functools import partial
#gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 60)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

import time,datetime
class timed(object):
    def __init__(self,desc='command',logger=None,pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
        if logger is None:
            self.log = print
        else:
            self.log = logger.info
    def __enter__(self):
        self.start = time.time()
        self.log('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            self.log('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            self.log('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))


def read_refs(year):
    with timed('Reading refs for year {}'.format(year)):    
        df = pd.read_table('P:/Projects/WoS/WoS/parsed/references/{}.txt.gz'.format(year),header=None,names=['uid','n_refs','refs','missing'],usecols=['uid','refs']).dropna()
        df['year'] = year
        return df

def map_years(year_df_tuple,yeardict):
    year,df = year_df_tuple
    with timed("Adding pubyear for year {}".format(year)):
        df['pubyear'] = df.source.apply(lambda x: yeardict.get(x))
        df = df.dropna()
        df.pubyear = df.pubyear.astype(int).copy()
        return df

class unstacker(object):
    def __init__(self):
        self.uids = []
        self.years = []
        self.sources = []
    def go(self,df):
        for i,row in df.iterrows():
            refs = row.refs.split('|')
            n = len(refs)
            self.uids += [row.uid]*n
            self.years += [row.year]*n
            self.sources += refs

def unstackit(year_df_tuple):
    year,df = year_df_tuple
    with timed('Unstacking year {}'.format(year)):
        u = unstacker()
        u.go(df)
        return pd.DataFrame({'uid':u.uids,'year':u.years,'source':u.sources})

def split_by_year(year_df_tuple):
    year,df = year_df_tuple
    with timed('Splitting year {}'.format(year)):
        return dict([(y,grp) for y,grp in df.groupby('pubyear')])

def grouper(year_df_tuple):
    year,df = year_df_tuple
    with timed('Grouping year {}'.format(year)):
        result = df.groupby(['source','year']).apply(lambda x: '|'.join(x.uid))
        result.name = 'citing_papers'
        result.reset_index().to_csv('S:/UsersData_NoExpiration/jjl2228/citations_by_year/{}.txt.gz'.format(year),index=False,header=None,sep='\t',compression='gzip')



if __name__ == '__main__':

    all_years = range(1950,2016)
    pool = mp.Pool(mp.cpu_count())

    with timed('MAIN PROCESSING'):
        with timed('reading reference DFs'):
            concat = pool.map(read_refs,all_years)
        with timed('concatenating to get year-uid dict'):
            collapsed = pd.concat([df[['uid','year']] for df in concat])
            year_dict = dict(zip(collapsed.uid,collapsed.year))
        with timed('unstacking'):
            concat = pool.map(unstackit,zip(all_years,concat))
        with timed('adding publication years'):
            func_partial = partial(map_years,yeardict=year_dict)
            concat = pool.map(func_partial,zip(all_years,concat))
        with timed('splitting by pub year'):
            concat = pool.map(split_by_year,zip(all_years,concat))
        with timed('grouping by pub year'):
            final = []
            for year in tq(all_years):
                current_year_df = []
                for d in concat:
                    if year in d:
                        current_year_df.append(d[year])
                final.append(pd.concat(current_year_df))
        with timed('final stacking'):
            pool.map(grouper,zip(all_years,final))
    try:
        pool.close()
    except:
        pass






