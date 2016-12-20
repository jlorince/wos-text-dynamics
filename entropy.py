import pathos.multiprocessing as mp
import pandas as pd
import numpy as np
from glob import glob
import sys
sys.path.append('/backup/home/jared/thoth')
import thoth.thoth as thoth

vocab_thresh = 100
data_dir = '/backup/home/jared/storage/wos-text-dynamics-data/'


class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        print '{} started...'.format(self.desc)
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print '{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad)
        else:
            print '{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad)



if __name__ == '__main__':
    global_term_counts = pd.Series.from_csv(data_dir+'global_term_counts.csv',encoding='utf8')
    pruned = global_term_counts[global_term_counts>=vocab_thresh]
    vocab = pd.Index([term for term in pruned.index if term not in stop and term.isalpha()])
