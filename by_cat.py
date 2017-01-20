import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys,glob,os,codecs,logging,argparse
from tqdm import tqdm as tq
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
from pathos import multiprocessing as mp
from functools import partial


import time,datetime
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
            rootLogger.info('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            rootLogger.info('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))
    
class process(object):

    def __init__(self,vocab,window,args,logger):
        self.vocab = vocab
        self.window = window
        self.logger = logger



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
        return entropy((p+q)/2.,base=2) - 0.5*entropy(p,base=2) - 0.5*entropy(q,base=2)

    def calc_measures(self,word_dists):
        ents = []
        jsds = []
        ent_difs = []
        apnd = []
        mx = len(word_dists)-(2*self.window-1)
        for i in range(mx):
            a = np.sum(word_dists[i:i+self.window],axis=0)
            asm = float(a.sum())
            if asm ==0:
                enta = np.nan       
            else:
                a = a/asm
                enta = entropy(a)
            b = np.sum(word_dists[i+self.window:i+self.window*2],axis=0)
            bsm = float(b.sum())
            if bsm == 0:
                entb = np.nan
            else:
                b = b/bsm
                entb = entropy(b)

            ents.append(enta)
            if i+self.window>=mx:
                apnd.append(entb)
            
            if asm==0 or bsm==0:
                ent_difs.append(np.nan)
                jsds.append(np.nan)
            else:
                ent_difs.append(entb-enta)
                jsds.append(jsd(a,b))
                    
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
        return calc_measures(shuffled_word_dists)
        
    def parse_cat(self,fi):
        df = pd.read_pickle(fi)
        if len(df==0):
            return 0
        # generate word distributions 

        cat_name = fi[fi.rfind('/')+1:-4]
        word_dists = np.zeros((25,len(self.vocab)))
        for year,grp in df.groupby('year'):
            word_dists[year-1991] = termcounts(grp.abstract)

        # total token count by year
        self.token_counts = word_dists.sum(1,dtype=int)
        # generate giant array of every token in data (for shuffling by null model)
        combined_word_dist = word_dists.sum(0,dtype=int)
        self.all_tokens = []
        for term,cnt in enumerate(combined_word_dist):#,total=len(combined_word_dist):
            self.all_tokens += [term]*cnt
        self.all_tokens = np.array(self.all_tokens)

        # calculate raw measures
        ents,ent_difs,jsds = calc_measures(word_dists)
        

        result = [shuffler(x) for x in range(self.args.null_bootstrap_samples)]
        
        dist_path = '{}results/termdist_{}.npy'.format(self.args.output,cat_name)
        if not os.path.exists(dist_path):
            np.save(dist_path,word_dists)   
        
        with open('{}results/results_{}_{}'.format(self.args.output,window,cat_name),'w') as out:
            
            for measure in ('ents','ent_difs','jsds'):
                out.write("{}\t{}\n".format(measure,','.join(vars()[measure].astype(str))))
            for i,measure in enumerate(['entropy-null','entdif-null','jsd-null']):
                samples = np.array([r[i] for r in result])
                m = samples.mean(0)
                ci = 1.96 * samples.std(0) / np.sqrt(self.args.null_bootstrap_samples)
                out.write('{}_m\t{}\n'.format(measure,','.join(m.astype(str))))
                out.write('{}_c\t{}\n'.format(measure,','.join(ci.astype(str))))
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
        complete = 0
        processor = process(vocab=vocab,window=w,args=args,logger=rootLogger)
        for fi,result  in tq(zip(files,pool.imap_unordered(processor.parse_cat,files)),total=len(files)):
            cat = fi[fi.rfind('/')+1:-4]
            if result == 0:
                rootLogger.info('No data for category "{}"'.format(cat))
            if result == 1:
                rootLogger.info('Category "{}" processed successfully for window size={}'.format(cat,window))
            complete+=result
        rootLogger.info('{} total categories processed for window size={}'.format(complete,window))
