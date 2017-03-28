from __future__ import print_function
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys,glob,os,codecs,logging,argparse
from tqdm import tqdm as tq
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
#import pathos.multiprocessing as mp
import multiprocess as mp
from functools import partial


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
    
class process(object):

    def __init__(self,vocab,window,args,logger):
        self.vocab = vocab
        self.window = window
        self.logger = logger
        self.args = args
        if args.null_model_mode == 'fixed':
            self.vocabset = set(vocab)


    def entropy(self,arr,base=2):
        return scipy_entropy(arr,base=base)

    # Given a pandas.Series of procesed abstracts, return the word frequency distribution 
    # across all abstracts (limited to our chosen vocabulary)
    def termcounts(self,abs_ser,):
        if self.args.null_model_mode=='fixed':
            abstracts = []
            for abstract in abs_ser:
                if len(abstract)>self.sample_size_tokens:
                    abstract = list(np.random.choice(abstract,self.sample_size_tokens,replace=False))
                abstracts += abstract
            tc = Counter(abstracts)
        else:
            tc = Counter(' '.join(abs_ser).split())
        arr = np.array([tc.get(k,0) for k in self.vocab])
        return arr 

    # calcualte Jensen Shannon Divergence of two probabability distributions
    def jsd(self,p,q):
        return self.entropy((p+q)/2.,base=2) - 0.5*self.entropy(p,base=2) - 0.5*self.entropy(q,base=2)

    def calc_jsds(self,word_dists1,word_dists2):
        jsds = []
        mx = len(word_dists1)-(self.window)
        self.result = []
        for i in range(mx):
            #with timed('Calc for {}: {}/{} (window={})'.format(self.cat,i+1,mx+1,self.window),logger=self.logger):
            
            ## prob-dist a
            a = np.sum(word_dists1[i:i+self.window],axis=0)
            asm = float(a.sum())
            if asm ==0:
                aprop = 0.0       
            else:
                aprop = a/asm

            ## prob-dist b
            b = np.sum(word_dists2[i:i+self.window],axis=0)
            bsm = float(b.sum())
            if bsm == 0:
                bprop = 0.0
            else:
                bprop = b/bsm
            ## zero counts in any: jsd =nan
            if asm==0 or bsm==0:
                jsds.append(np.nan)
            else:
                jsds.append(self.jsd(aprop,bprop))
                
        return np.array(jsds)
            
       
    def parse_cat(self,fi):
        self.cat = fi[fi.rfind('/')+1:-4]
        with timed('Processing category "{}" (window={})'.format(self.cat,self.window),logger=self.logger):

            result_path = '{}results_{}_{}'.format(self.args.output,self.window,self.cat)
            if os.path.exists(result_path):
                self.logger.info('Category "{}" already done for window={}'.format(self.cat,self.window))
                return 0

            df = pd.read_pickle(fi).dropna(subset=['abstract_parsed'])
            if len(df)==0:
                self.logger.info('No data for category "{}"'.format(self.cat))
                return 0

            if self.args.null_model_mode == 'fixed':

                jsd_t_result = []
                jsd_tt_result = []
                with timed('Sampling measures for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                    sample_size = int(round(df.year.value_counts().min() * self.args.min_prop)) 
                    df['abstract_parsed'] = df.abstract_parsed.apply(lambda x: [word for word in x.split() if word in self.vocabset])
                    self.sample_size_tokens = int(round(df.groupby('year').apply(lambda grp: grp.abstract_parsed.apply(lambda x: len(x)).mean()).min()))
                    self.logger.info('Fixed sample size DOCUMENTS for category {} = {}'.format(self.cat,sample_size))
                    self.logger.info('Fixed sample size TOKENS for category {} = {}'.format(self.cat,self.sample_size_tokens))
                    

                    for i in range(self.args.null_bootstrap_samples):

                        ## generate sample 1
                        sampled1 = df.groupby('year').apply(lambda x: x.sample(n=sample_size))
                        word_dists1 = np.zeros((25,len(self.vocab)))
                        for year,grp in sampled1.groupby('year'):
                            word_dists1[year-1991] = self.termcounts(grp.abstract_parsed)
                        ## generate sample 2
                        sampled2 = df.groupby('year').apply(lambda x: x.sample(n=sample_size))
                        word_dists2 = np.zeros((25,len(self.vocab)))
                        for year,grp in sampled2.groupby('year'):
                            word_dists2[year-1991] = self.termcounts(grp.abstract_parsed)

                        jsds_t = self.calc_jsds(word_dists1,word_dists2)
                        jsds_tt = self.calc_jsds(word_dists1[:-1,:],word_dists2[1:,:])
                        jsd_t_result.append(jsds_t)
                        jsd_tt_result.append(jsds_tt)




            
            with timed('Writing results for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                with open(result_path,'w') as out:

                    if self.args.null_model_mode=='fixed':
                        jsd_t_result = np.vstack(jsd_t_result)
                        jsd_tt_result = np.vstack(jsd_tt_result)

                        for measure,data in zip(('jsd_t','jsd_tt'),(jsd_t_result,jsd_tt_result)):
                            m = data.mean(0)
                            ci = 1.96 * data.std(0) / np.sqrt(self.args.null_bootstrap_samples)
                            out.write("{}_m\t{}\n".format(measure,','.join(m.astype(str))))
                            out.write("{}_c\t{}\n".format(measure,','.join(ci.astype(str))))

        #rootLogger.info('Category "{}" processed successfully for window size={}'.format(cat_name,self.window))
        return 1
 

if __name__=='__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic measures of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-w", "--window", help="window size, enter a single value, range (x_y), or list (x,y,z)",type=str,default='1')
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='output/by-cat_wos/')
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=10)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='data/by-cat_wos_sample/',type=str)
    #parse.add_argument("-c", "--cats", help="path to pickled field-level dataframes", default='/backup/home/jared/storage/wos-text-dynamics-data/by-cat',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    # parser.add_argument("-n", "--null_model_mode",help='null model mode ("global" or "local")',default='fixed',type=str,choices=['global','local','fixed'])
    parser.add_argument("-n", "--null_model_mode",help='null model mode ("fixed")',default='fixed',type=str,choices=['fixed'])   
    parser.add_argument("-r", "--min_prop",help='pRoportion of year with least publications to establish fixed sample size ',default=0.5,type=float)


    args = parser.parse_args()

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

    ### Vocabulary setup
    vocab_path = args.datadir+'vocab_pruned_'+str(args.vocab_thresh)
    if os.path.exists(vocab_path):
        rootLogger.info('Loading existing vocab file')
        vocab = [line.strip() for line in codecs.open(vocab_path,encoding='utf8')]
        #vocab = [line.strip() for line in open(vocab_path)]
    else:
        rootLogger.info('Generating new vocab file')
        vocab_dict ={}
        for fpath in tq(glob.glob(args.datadir+'*.pkl')):
            df = pd.read_pickle(fpath)
            # for abstract in tq(df.abstract):
            ## change from jared's version
            for abstract in tq(df.abstract_parsed):
                for term in abstract.split():
                    vocab_dict[term] = vocab_dict.get(term,0)+1
        raw_term_counts = pd.Series(vocab_dict)  

        stemmer = EnglishStemmer()
        stop = set(stopwords.words('english'))
        stop = stop.union([stemmer.stem(s) for s in stop])
        pruned = raw_term_counts[raw_term_counts>=args.vocab_thresh]
        ## change from jared's version
        # vocab = sorted([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])
        vocab = sorted([term for term in pruned.index if term not in stop and type(term)==str and term.isalpha()])
        rootLogger.info("Total vocab size= {}".format(len(vocab)))

    pool = mp.Pool(args.procs)
    if '_' in args.window:
        start,end = map(int,args.window.split('_'))
        window_range = range(start,end+1) 
    elif ',' in args.window:
        window_range = map(int,args.window.split(','))
    else:
        window_range = [int(args.window)]

    files = glob.glob(args.datadir+'*.pkl')
    for w in window_range:
        # complete = 0
        # def gen_wrapper(vocab,w,args,rootLogger):
        #     def wrapper(fi):
        #         processor = process(vocab=vocab,window=w,args=args,logger=rootLogger)
        #         processor.parse_cat(fi)
        #     return wrapper
        # wrapper = gen_wrapper(vocab,w,args,rootLogger)
        processor = process(vocab=vocab,window=w,args=args,logger=rootLogger)
        pool.map(processor.parse_cat,files)#,chunksize=len(files)//args.procs)

