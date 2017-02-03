from __future__ import print_function
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys,glob,os,codecs,logging,argparse
from tqdm import tqdm as tq
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
import pathos.multiprocessing as mp
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


    def entropy(self,arr,base=2):
        return scipy_entropy(arr,base=base)

    # Given a pandas.Series of procesed abstracts, return the word frequency distribution 
    # across all abstracts (limited to our chose vocabulary)
    def termcounts(self,abs_ser):
        tc = Counter(' '.join(abs_ser).split())
        arr = np.array([tc.get(k,0) for k in self.vocab])
        return arr 

    # calcualte Jensen Shannon Divergence of two probabability distributions
    def jsd(self,p,q):
        return self.entropy((p+q)/2.,base=2) - 0.5*self.entropy(p,base=2) - 0.5*self.entropy(q,base=2)

    def calc_measures(self,word_dists):
        ents = []
        jsds = []
        ent_difs = []
        apnd = []
        mx = len(word_dists)-(2*self.window-1)
        self.result = []
        for i in range(mx):
            #with timed('Calc for {}: {}/{} (window={})'.format(self.cat,i+1,mx+1,self.window),logger=self.logger):
            a = np.sum(word_dists[i:i+self.window],axis=0)
            asm = float(a.sum())
            if asm ==0:
                enta = np.nan       
            else:
                aprop = a/asm
                enta = self.entropy(aprop)
            b = np.sum(word_dists[i+self.window:i+self.window*2],axis=0)
            bsm = float(b.sum())
            if bsm == 0:
                entb = np.nan
            else:
                bprop = b/bsm
                entb = self.entropy(bprop)

            ents.append(enta)
            if i+self.window>=mx:
                apnd.append(entb)
            
            if asm==0 or bsm==0:
                ent_difs.append(np.nan)
                jsds.append(np.nan)
                if self.args.null_model_mode == 'local':
                    x = [(np.nan,np.nan) for _ in xrange(self.args.null_bootstrap_samples)]
                    self.result.append([np.array(r) for r in zip(*x)])

            else:
                ent_difs.append(entb-enta)
                jsds.append(self.jsd(aprop,bprop))
                if self.args.null_model_mode == 'local':
                    with timed('Local null model for {}: {}/{} (window={})'.format(self.cat,i+1,mx+1,self.window),logger=self.logger):
                        combined_word_dist = (a+b).astype(int)
                        self.all_tokens = []
                        for term,cnt in enumerate(combined_word_dist):#,total=len(combined_word_dist):
                            self.all_tokens += [term]*cnt
                        self.all_tokens = np.array(self.all_tokens)
                        x = [self.local_shuffler(int(asm)) for _ in xrange(self.args.null_bootstrap_samples)]
                        self.result.append([np.array(r) for r in zip(*x)])
                
        return np.array(ents+apnd),np.array(ent_difs),np.array(jsds)
            

    def shuffler(self,idx):
        token_seq = self.all_tokens.copy()
        np.random.shuffle(token_seq)
        idx = 0
        shuffled_word_dists = np.zeros((25,len(self.vocab)))
        for i,toke in enumerate(self.token_counts):
            current = token_seq[idx:idx+toke]
            unique, counts = np.unique(current, return_counts=True)
            word_dist = np.zeros(len(self.vocab))
            word_dist[unique] = counts
            shuffled_word_dists[i] = word_dist
            idx+=toke
        return self.calc_measures(shuffled_word_dists)

    def local_shuffler(self,token_count_a):
        token_seq = self.all_tokens.copy()
        np.random.shuffle(token_seq)
        a,b = token_seq[:token_count_a],token_seq[token_count_a:]
        shuffled_dists = []
        for tokens in (a,b):
            unique, counts = np.unique(tokens, return_counts=True)
            word_dist = np.zeros(len(self.vocab))
            word_dist[unique] = counts
            shuffled_dists.append(word_dist)
        a,b = shuffled_dists
        enta = self.entropy(a)
        entb = self.entropy(b)
        return entb-enta,self.jsd(a,b)

        
    def parse_cat(self,fi):
        self.cat = fi[fi.rfind('/')+1:-4]
        with timed('Processing category "{}" (window={})'.format(self.cat,self.window),logger=self.logger):

            result_path = '{}results_{}_{}'.format(self.args.output,self.window,self.cat)
            if os.path.exists(result_path):
                self.logger.info('Category "{}" already done for window={}'.format(self.cat,self.window))
                return 0

            df = pd.read_pickle(fi)
            if len(df)==0:
                self.logger.info('No data for category "{}"'.format(self.cat))
                return 0

            if self.args.null_model_mode in ('global','local'):
                # generate word distributions 
                word_dists = np.zeros((25,len(self.vocab)))
                for year,grp in df.groupby('year'):
                    word_dists[year-1991] = self.termcounts(grp.abstract)

                if self.args.null_model_mode == 'global':

                    with timed('Running null model for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                        # total token count by year
                        self.token_counts = word_dists.sum(1,dtype=int)
                        # generate giant array of every token in data (for shuffling by null model)
                        combined_word_dist = word_dists.sum(0,dtype=int)
                        self.all_tokens = []
                        for term,cnt in enumerate(combined_word_dist):#,total=len(combined_word_dist):
                            self.all_tokens += [term]*cnt
                        self.all_tokens = np.array(self.all_tokens)

                        self.result = [self.shuffler(x) for x in range(self.args.null_bootstrap_samples)]

                # calculate raw measures
                with timed('Calculating raw measures for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                    ents,ent_difs,jsds = self.calc_measures(word_dists)
            
                dist_path = '{}termdist_{}.npy'.format(self.args.output,self.cat)
                if not os.path.exists(dist_path):
                    np.save(dist_path,word_dists)   

            elif self.args.null_model_mode == 'fixed':

                ent_result = []
                ent_dif_result = []
                jsd_result = []

                with timed('Sampling measures for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                    sample_size = int(round(df.year.value_counts().min() * self.args.min_prop)) 
                    self.logger.info('Fixed sample size for category {} = {} papers'.format(self.cat,sample_size))
                    for i in range(self.args.null_bootstrap_samples):
                        sampled = df.groupby('year').apply(lambda x: x.sample(n=sample_size))
                        # generate word distributions 
                        word_dists = np.zeros((25,len(self.vocab)))
                        for year,grp in sampled.groupby('year'):
                            word_dists[year-1991] = self.termcounts(grp.abstract)
                        ents,ent_difs,jsds = self.calc_measures(word_dists)
                        ent_result.append(ents)
                        ent_dif_result.append(ent_difs)
                        jsd_result.append(jsds)




            
            with timed('Writing results for {} (window={})'.format(self.cat,self.window),logger=self.logger):
                with open(result_path,'w') as out:

                    if self.args.null_model_mode=='fixed':
                        ent_result = np.vstack(ent_result)
                        ent_dif_result = np.vstack(ent_dif_result)
                        jsd_result = np.vstack(jsd_result)

                        for measure,data in zip(('ent','ent_dif','jsd'),(ent_result,ent_dif_result,jsd_result)):
                            m = data.mean(0)
                            ci = 1.96 * data.std(0) / np.sqrt(self.args.null_bootstrap_samples)
                            out.write("{}_m\t{}\n".format(measure,','.join(m.astype(str))))
                            out.write("{}_c\t{}\n".format(measure,','.join(ci.astype(str))))

                    else:
                        
                        for measure in ('ents','ent_difs','jsds'):
                            out.write("{}\t{}\n".format(measure,','.join(vars()[measure].astype(str))))
                        
                        if self.args.null_model_mode == 'global':
                            for i,measure in enumerate(['entropy-null','entdif-null','jsd-null']):
                                samples = np.array([r[i] for r in self.result])
                                m = samples.mean(0)
                                ci = 1.96 * samples.std(0) / np.sqrt(self.args.null_bootstrap_samples)
                                out.write('{}_m\t{}\n'.format(measure,','.join(m.astype(str))))
                                out.write('{}_c\t{}\n'.format(measure,','.join(ci.astype(str))))
                        
                        elif self.args.null_model_mode == 'local':
                            for i,measure in enumerate(['entdif-null','jsd-null']):
                                samples = np.vstack([r[i] for r in self.result])
                                m = samples.mean(1)
                                ci = 1.96 * samples.std(1) / np.sqrt(self.args.null_bootstrap_samples)
                                out.write('{}_m\t{}\n'.format(measure,','.join(m.astype(str))))
                                out.write('{}_c\t{}\n'.format(measure,','.join(ci.astype(str))))


        #rootLogger.info('Category "{}" processed successfully for window size={}'.format(cat_name,self.window))
        return 1
 

if __name__=='__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic measures of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-w", "--window", help="window size, enter a single value, range (x_y), or list (x,y,z)",type=str,default='1')
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='/backup/home/jared/storage/wos-text-dynamics-data/results/')
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=100)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='/backup/home/jared/storage/wos-text-dynamics-data/by-cat/',type=str)
    #parse.add_argument("-c", "--cats", help="path to pickled field-level dataframes", default='/backup/home/jared/storage/wos-text-dynamics-data/by-cat',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    parser.add_argument("-n", "--null_model_mode",help='null model mode ("global" or "local")',default='local',type=str,choices=['global','local','fixed'])
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
    else:
        rootLogger.info('Generating new vocab file')
        vocab_dict ={}
        for fpath in tq(glob.glob(args.datadir+'*.pkl')):
            df = pd.read_pickle(fpath)
            for abstract in tq(df.abstract):
                for term in abstract.split():
                    vocab_dict[term] = vocab_dict.get(term,0)+1
        raw_term_counts = pd.Series(vocab_dict)  

        stemmer = EnglishStemmer()
        stop = set(stopwords.words('english'))
        stop = stop.union([stemmer.stem(s) for s in stop])
        pruned = raw_term_counts[raw_term_counts>=args.vocab_thresh]
        vocab = sorted([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])
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
        pool.map(processor.parse_cat,files,chunksize=len(files)//args.procs)

