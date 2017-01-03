import pathos.multiprocessing as mp
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter
from functools import partial
import sys,logging,time,datetime,math,argparse,os,codecs
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
sys.path.append('/backup/home/jared/thoth')
import thoth.thoth as thoth

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

def termcounts(abs_ser):
    tc = Counter(' '.join(abs_ser).split())
    arr = np.array([tc.get(k,0) for k in vocab])
    return arr


def gen_dists(fi):
    cat = fi[fi.rfind('/')+1:-4]
    dists = []
    with timed('dist generation for {}'.format(cat)):
        df = pd.read_pickle(fi)
        for year in xrange(1991,2016):
            with timed('dists {}-->{}'.format(cat,year)):
                year_df = df[df.year==year]
                current = termcounts(year_df.abstract)
                if current.sum()>0:
                    dists.append(current)
                else:
                    dists.append(np.nan)

    with open('/backup/home/jared/storage/wos-text-dynamics-data/token_counts/'+cat,'w') as fout:
        fout.write(cat+'\t'+','.join([str(int(d.sum())) if d is not np.nan else '0' for d in dists])+'\n')
    with open('/backup/home/jared/storage/wos-text-dynamics-data/vocab_sizes/'+cat,'w') as fout:
        fout.write(cat+'\t'+','.join([str(int((d>0).sum())) if d is not np.nan else '0' for d in dists])+'\n')


def new_terms_by_year(fi):
    cat = fi[fi.rfind('/')+1:-4]
    result = []
    cumulative = np.zeros(len(vocab))
    with timed('vocab change for {}'.format(cat)):
        df = pd.read_pickle(fi)
        last = None
        for year in xrange(1991,2016):
            with timed('dists {}-->{}'.format(cat,year)):
                year_df = df[df.year==year]
                current = termcounts(year_df.abstract)
                new_terms = ((current!=0)&(cumulative==0)).sum()
                cumulative += current
                if current.sum()>0:
                    result.append(new_terms)
                else:
                    result.append(0)    
    with open('/backup/home/jared/storage/wos-text-dynamics-data/new_terms/'+cat,'w') as fout:
        fout.write(cat+'\t'+','.join([str(int(d)) for d in result])+'\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='/backup/home/jared/storage/wos-text-dynamics-data/',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    args = parser.parse_args()



    with timed('vocabulary setup'):
        ### stopword setup
        stop = set(stopwords.words('english'))
        stemmer = EnglishStemmer()
        stop = stop.union([stemmer.stem(s) for s in stop])

        ### vocabulary setup
        vocab_path = args.datadir+'vocab_pruned_{}'.format(args.vocab_thresh)
        if os.path.exists(vocab_path):
            with timed('Loading existing vocab file'):
                vocab = [line.strip() for line in codecs.open(vocab_path,encoding='utf8')]
        else:
            with timed('generating new vocab file with thresh={}'.format(args.vocab_thresh)):
                global_term_counts = pd.Series.from_csv(args.datadir+'global_term_counts.csv',encoding='utf8')
                pruned = global_term_counts[global_term_counts>=args.vocab_thresh]
                vocab = sorted([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])
                with codecs.open(vocab_path,'w',encoding='utf8') as f:
                    f.write('\n'.join(vocab))

    with timed('pool setup'):
        ### file setup
        files = glob("{}by-cat/*".format(args.datadir))
        
        ### pool setup
        chunksize = int(math.ceil(len(files) / float(args.procs)))
        pool = mp.Pool(args.procs)


    #with timed('parallel dist generation'):
    #    pool.map(gen_dists,files) 
    with timed('new term calculations'):
        pool.map(new_terms,files)

