from __future__ import print_function
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys,glob,os,codecs,logging,argparse,datetime,gzip
from tqdm import tqdm as tq
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
#import pathos.multiprocessing as mp
import multiprocess as mp
from functools import partial

def logger_setup():
    pid = os.getpid()
    now = datetime.datetime.now()
    log_filename = now.strftime('%Y%m%d_%H%M%S_{}.log.part'.format(pid))
    logFormatter = logging.Formatter("%(asctime)s\t[{}]\t[%(levelname)s]\t%(message)s".format(pid))
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    return rootLogger

logger = logger_setup()


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

    def __init__(self,vocab,window,args):
        self.vocab = vocab
        self.window = window
        self.args = args
        self.vocabset = set(vocab)


    def entropy(self,arr,base=2):
        return scipy_entropy(arr,base=base)

    def termcounts(self,abs_ser,):
        abstracts = []
        for abstract in abs_ser:
            #if len(abstract)>self.sample_size_tokens:
            abstract = list(np.random.choice(abstract,self.args.n_token_samples,replace=False))
            abstracts += abstract
        tc = Counter(abstracts)
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
            #with timed('Calc for {}: {}/{} (window={})'.format(self.cat,i+1,mx+1,self.window),logger=logger):
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

            else:
                ent_difs.append(entb-enta)
                jsds.append(self.jsd(aprop,bprop))
                
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
        self.cat = fi[fi.rfind('\\')+1:fi.find('.')]
        with timed('Processing category "{}" (window={})'.format(self.cat,self.window),logger=logger):

            result_path = '{}results_{}_{}'.format(self.args.output,self.window,self.cat)
            if os.path.exists(result_path):
                logger.info('Category "{}" already done for window={}'.format(self.cat,self.window))
                return 0

            df = pd.read_pickle(fi)
            if self.args.data_source == 'elsevier':
                df = df[(df.raw_token_count>0)&(df.formatted_text==1)&(df.year<2014)] # for now let's look at the overlap
            elif self.args.data_source == 'wos':
                df = df.dropna(subset=['abstract_parsed'])
            
            if len(df)==0:
                logger.info('No data after initial filtering for category "{}" (window={})'.format(self.cat,self.window))
                return 0

            ent_result = []
            ent_dif_result = []
            jsd_result = []

            with timed('Sampling measures for {} (window={})'.format(self.cat,self.window),logger=logger):
                
                if self.args.data_source == 'elsevier':

                    text_dict = {}
                    token_counts_by_uid = {}
                    for line in gzip.open('E:/Users/jjl2228/WoS/wos-text-dynamics-data/by-cat_elsevier/{}.txt.gz'.format(self.cat)):
                        line = line.decode('utf8').strip().split('\t')
                        if len(line)==8 and line[7]:
                            text = [word for word in line[7].split() if word in self.vocabset]
                            if text:
                                uid = line[0]
                                text_dict[uid] = text
                                token_counts_by_uid[uid] = len(text)
                    # join token counts
                    token_counts_by_uid = pd.Series(token_counts_by_uid,name='parsed_token_count')
                    df = df.join(token_counts_by_uid,on='uid',how='inner')

                elif self.args.data_source == 'wos':

                    df['abstract_parsed'] = df.abstract_parsed.apply(lambda x: [word for word in x.split() if word in self.vocabset])
                    df['parsed_token_count'] = df.abstract_parsed.apply(lambda x: len(x))

            df = df[df.parsed_token_count >= 2*self.args.n_token_samples]
            if len(df)==0:
                logger.info('No data after doc smaple filtering for category "{}" (window={})'.format(self.cat,self.window))
                return 0

            
            n_years = {'elsevier':64,'wos':25}[self.args.data_source]
            start_year = {'elsevier':1950,'wos':1991}[self.args.data_source]
            for i in range(self.args.null_bootstrap_samples):
                sampled = df.groupby('year').apply(lambda x: x.sample(n=self.args.n_doc_samples,replace=False) if len(x)>=2*self.args.n_doc_samples else None)
                if len(sampled)==0:
                    logger.info('No data after doc smaple filtering for category "{}" (window={})'.format(self.cat,self.window))
                    return 0

                # generate word distributions 
                word_dists = np.zeros((n_years,len(self.vocab)))
                for year,grp in sampled.groupby('year'):
                    if self.args.data_source == 'elsevier':
                        word_dists[year-start_year] = self.termcounts([text_dict[uid] for uid in grp.uid])
                    elif self.args.data_source == 'wos':
                        word_dists[year-start_year] = self.termcounts(grp.abstract_parsed)
                ents,ent_difs,jsds = self.calc_measures(word_dists)
                ent_result.append(ents)
                ent_dif_result.append(ent_difs)
                jsd_result.append(jsds)                        

            
            with timed('Writing results for {} (window={})'.format(self.cat,self.window),logger=logger):
                with open(result_path,'w') as out:
                    ent_result = np.vstack(ent_result)
                    ent_dif_result = np.vstack(ent_dif_result)
                    jsd_result = np.vstack(jsd_result)

                    for measure,data in zip(('ent','ent_dif','jsd'),(ent_result,ent_dif_result,jsd_result)):
                        m = data.mean(0)
                        ci = 1.96 * data.std(0) / np.sqrt(self.args.null_bootstrap_samples)
                        out.write("{}_m\t{}\n".format(measure,','.join(m.astype(str))))
                        out.write("{}_c\t{}\n".format(measure,','.join(ci.astype(str))))


if __name__=='__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic measures of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-w", "--window", help="window size, enter a single value, range (x_y), or list (x,y,z)",type=str,default='1')
    parser.add_argument("-o", "--output", help="output path for results",default=None)
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=100)
    parser.add_argument("-d", "--datadir",help="root input data directory",default=None,type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    parser.add_argument("-n", "--n_doc_samples",help='Number of documents to sample each year',default=100,type=int)
    parser.add_argument("-t", "--n_token_samples",help='Number of tokens to sample from each document',default=100,type=int)
    parser.add_argument("-s","--data_source",help="Source of data (pickeld WoS abstracts, or Elsevier data in gzipped text files",default='wos',choices=['wos','elsevier'])

    args = parser.parse_args()
    # set conditional argument defaults
    if args.datadir is None:
        if args.data_source == 'wos':
            args.datadir = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/by-cat_wos/'
        elif args.data_source == 'elsevier':
            args.datadir = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/metadata/'
    # if args.min_prop>1:
    #     args.min_prop = int(args.min_prop)
    if args.output is None:
        args.output = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/results/by-cat-results-{}/'.format(args.data_source)


    ### LOGGING SETUP
    
    logger.info(str(args))

    ### Vocabulary setup
    vocab_path = args.datadir+'vocab_pruned_'+str(args.vocab_thresh)
    if os.path.exists(vocab_path):
        logger.info('Loading existing vocab file')
        vocab = [line.strip() for line in open(vocab_path,encoding='utf8')]
        #vocab = [line.strip() for line in open(vocab_path)]
    else:
        logger.info('Generating new vocab file')
        vocab_dict ={}
        # NOTE THIS LOOKS AT JUST FORMATTED TEXT
        if args.data_source == 'elsevier':
            d = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/termcounts_elsevier/global_term_counts_formatted_*'
        elif args.data_source == 'wos':
            d = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/termcounts_wos/global_term_counts_*'
        for f in tq(glob.glob(d)):
            for line in open(f,encoding='utf8'):
                term,cnt = line.strip().split(',')
                cnt = int(cnt)
                vocab_dict[term] = vocab_dict.get(term,0)+cnt

        vocab = sorted([k for k,v in vocab_dict.items() if v>=args.vocab_thresh])
        logger.info("Total vocab size= {}".format(len(vocab)))
        with open(vocab_path,'w',encoding='utf8') as out:
            for term in vocab:
                out.write(term+'\n')

    pool = mp.Pool(args.procs)
    if '_' in args.window:
        start,end = map(int,args.window.split('_'))
        window_range = range(start,end+1) 
    elif ',' in args.window:
        window_range = map(int,args.window.split(','))
    else:
        window_range = [int(args.window)]

    if arg.data_source == 'wos':
        files = glob.glob(args.datadir+'*.trimmed.pkl')
    elif args.data_source == 'elsevier':
        files = glob.glob(args.datadir+'*.pkl')

    
    for w in window_range:

        processor = process(vocab=vocab,window=w,args=args)
        pool.map(processor.parse_cat,files,chunksize=len(files)//args.procs)

    try:
        pool.close()
        pool.terminate()
    except:
        pass

## This happens out of the main block to close all handlers
for handler in logger.handlers:
    handler.close()
    logger.removeHandler(handler)

## now back in the main process we consolidate the logs
if __name__=='__main__':
    files = glob.glob('*.log.part')
    log_df = pd.concat([pd.read_table(f,header=None,parse_dates=[0]) for f in files])
    for f in files: os.remove(f)
    new_log_filename= datetime.datetime.now().strftime('%Y%m%d_%H%M%S.log')
    log_df.sort_values(0).to_csv(new_log_filename,sep='\t',index=False)


