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
        rootLogger.info('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            rootLogger.info('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            rootLogger.info('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))
    
def termcounts(abs_ser):
    tc = Counter(' '.join(abs_ser).split())
    arr = np.array([tc.get(k,0) for k in vocab])
    return arr


def process_grp(fi):
    cat = fi[fi.rfind('/')+1:-4]
    with timed('base calculations for {}'.format(cat)):       
        if os.path.exists('{}{}/raw_results'.format(args.output,cat)):
            rootLogger.info('raw results for category "{}" already processed,skipping'.format(cat))
            return None
        df = pd.read_pickle(fi)
        # calculate raw entropy and divergence measures by year
        output = {}
        last = None
        cumulative = np.zeros(len(vocab))
        for year in xrange(1991,2016):
            with timed('raw {}-->{}'.format(cat,year)):
                year_df = df[df.year==year]
                output[year] = {'jsd':np.array([np.nan]*5),'jsd_c':np.array([np.nan]*5),'H':np.array([np.nan]*5),'H_c':np.array([np.nan]*5)}
                n = len(year_df)
                output[year]['n'] = n
                if n>0:
                    tc_current = termcounts(year_df.abstract)
                    current = np.random.multinomial(n=args.n_samples,pvals=tc_current/float(tc_current.sum()))

                    if last is not None:
                        output[year]['jsd'] = thoth.calc_jsd(current,last,0.5,args.thoth_mc_samples)
                        output[year]['jsd_c'] = thoth.calc_jsd(current,cumulative,0.5,args.thoth_mc_samples)
                    if current.sum()>0:
                        output[year]['H'] = thoth.calc_entropy(current,args.thoth_mc_samples)
                        cumulative += current
                        output[year]['H_c'] = thoth.calc_entropy(cumulative,args.thoth_mc_samples)
                        last = current
                    else:
                        last = None
                else:
                    last = None
    if not os.path.exists(args.output+cat):
        os.mkdir(args.output+cat)
    with open('{}{}/raw_results'.format(args.output,cat),'w') as fout:
        for year in output:
            fout.write('\t'.join([str(year),str(output[year]['n'])]+[','.join(output[year][measure].astype(str)) for measure in ('jsd','jsd_c','H','H_c')])+'\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-n", "--n_samples", help="number of tokens to sample each year",type=int,default=100000)
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='/backup/home/jared/storage/wos-text-dynamics-data/results_sampled/')
    parser.add_argument("-t", "--thoth_mc_samples", help="Number of monte carlo samples for thoth calculations",type=int,default=1000)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='/backup/home/jared/storage/wos-text-dynamics-data/',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    parser.add_argument("-x", "--debug", help="enable mode", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.thoth_mc_samples = 5

    ### LOGGING SETUP
    now = datetime.datetime.now()
    if args.logfile:
        args.logfiles += '_'
    log_filename = now.strftime('{}%Y%m%d_%H%M%S.log'.format(args.logfile))
    #log_filename = now.strftime('{}%Y%m%d_%H%M%S.log'.format(args.logfile+'_{}_{}'.format(args.window,args.side)))
    logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    rootLogger.info(str(args))

    with timed('full analysis'):

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

        with timed('category selection'):
            token_counts = pd.read_table(args.datadir+'token_counts/token_counts_concat',header=None,names=['cat','counts'])
            dr = pd.date_range(start='1991-01-01',end='2015-01-01',freq='AS')
            unpacked = token_counts['counts'].apply(lambda x: pd.Series(map(int,x.split(',')),index=dr)).T
            unpacked.columns = token_counts['cat'].values

            include = unpacked.columns[unpacked.apply(lambda x: np.all(x>=args.n_samples))]



        with timed('pool setup'):
            ### file setup
            files = glob("{}by-cat/*".format(args.datadir))

            files = [f for f in files if f[f.rfind('/')+1:-4] in include]
            if args.debug:
                files = files[:3]

            ### pool setup
            chunksize = int(math.ceil(len(files) / float(args.procs)))
            pool = mp.Pool(args.procs)

        with timed('parallel processing'):
            #results = pool.map(process_grp,files)
            pool.map(process_grp,files)

                                    