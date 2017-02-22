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
        if args.null_model_mode == 'fixed':
            self.vocabset = set(vocab)


    def entropy(self,arr,base=2):
        return scipy_entropy(arr,base=base)


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
                if self.args.null_model_mode == 'local':
                    x = [(np.nan,np.nan) for _ in xrange(self.args.null_bootstrap_samples)]
                    self.result.append([np.array(r) for r in zip(*x)])

            else:
                ent_difs.append(entb-enta)
                jsds.append(self.jsd(aprop,bprop))
                if self.args.null_model_mode == 'local':
                    with timed('Local null model for {}: {}/{} (window={})'.format(self.cat,i+1,mx+1,self.window),logger=logger):
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

    def parse_cat_elsevier(self,fi):
        self.cat = fi[fi.rfind('\\')+1:-4]
        with timed('Processing category "{}" (window={})'.format(self.cat,self.window),logger=logger):

            result_path = '{}results_{}_{}'.format(self.args.output,self.window,self.cat)
            if os.path.exists(result_path):
                logger.info('Category "{}" already done for window={}'.format(self.cat,self.window))
                return 0

            if self.args.null_model_mode in ('global','local'):
                raise Exception('NULL MODEL MODE "{}" NOT IMPLEMENTED'.format(self.args.null_model_mode))

            elif self.args.null_model_mode == 'fixed':

                #df = pd.read_table(fi,compression='gzip',header=None,names=['uid','el_id','year','abstract_text','formatted_text','raw_text','abs_only','text'],dtype={'uid':str,'el_id':int,'year':int,'abstract_text':int,'formatted_text':int,'raw_text':int,'abs_only':int,'text':str}).dropna()
                #df = df[df.formatted_text==1]
                df = pd.read_pickle(fi)
                df = df[(df.raw_token_count>0)&(df.formatted_text==1)&(df.year<2014)] # for now let's look at the overlap
                if len(df)==0:
                    logger.info('No data after filtering for category "{}" (window={})'.format(self.cat,self.window))
                    return 0


                ent_result = []
                ent_dif_result = []
                jsd_result = []

                with timed('Sampling measures for {} (window={})'.format(self.cat,self.window),logger=logger):
                    
                    ### HERE IS WHERE WE NEED REFINEMENTS

                    if type(self.args.min_prop)==float:
                        sample_size = int(round(df.year.value_counts().min() * self.args.min_prop)) 
                        df['text'] = df.text.apply(lambda x: [word for word in x.split() if word in self.vocabset])
                        self.sample_size_tokens = int(round(df.groupby('year').apply(lambda grp: grp.text.apply(lambda x: len(x)).mean()).min()))

                    elif type(self.args.min_prop)==int:
                        counts_by_year = df.year.value_counts()
                        downsample = counts_by_year[counts_by_year >= (2*self.args.min_prop)].index
                        text_dict = {}
                        token_counts_by_uid = {}
                        for line in gzip.open('E:/Users/jjl2228/WoS/wos-text-dynamics-data/by-cat_elsevier/{}.txt.gz'.format(self.cat)):
                            line = line.decode('utf8').strip().split('\t')
                            if line[7]:
                                text = [word for word in line[7].split() if word in self.vocabset]
                                if text:
                                    uid = line[0]
                                    text_dict[uid] = text
                                    token_counts_by_uid[uid] = len(text)
                        # join token counts
                        token_counts_by_uid = pd.Series(token_counts_by_uid,name='parsed_token_count')
                        df = df.join(token_counts_by_uid,on='uid')
                        sample_size = max(self.args.min_prop, int(round(df.year.value_counts().min() * 0.5)))
                        self.sample_size_tokens = int(round(df.groupby('year').parsed_token_count.mean().min()))
                    

                    logger.info('Fixed sample size DOCUMENTS for category {} = {}'.format(self.cat,sample_size))
                    logger.info('Fixed sample size TOKENS for category {} = {}'.format(self.cat,self.sample_size_tokens))
                    for i in range(self.args.null_bootstrap_samples):
                        sampled = df.groupby('year').apply(lambda x: x.sample(n=sample_size,replace=False) if len(x)>=2*sample_size else None)
                        if len(sampled)==0:
                            logger.info('No data after filtering for category "{}" (window={})'.format(self.cat,self.window))
                            return 0

                        # generate word distributions 
                        word_dists = np.zeros((64,len(self.vocab)))
                        for year,grp in sampled.groupby('year'):
                            word_dists[year-1950] = self.termcounts([text_dict[uid] for uid in grp.uid])
                        ents,ent_difs,jsds = self.calc_measures(word_dists)
                        ent_result.append(ents)
                        ent_dif_result.append(ent_difs)
                        jsd_result.append(jsds)

            
            with timed('Writing results for {} (window={})'.format(self.cat,self.window),logger=logger):
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


    def parse_cat_wos(self,fi):
        self.cat = fi[fi.rfind('\\')+1:-4]
        with timed('Processing category "{}" (window={})'.format(self.cat,self.window),logger=logger):

            result_path = '{}results_{}_{}'.format(self.args.output,self.window,self.cat)
            if os.path.exists(result_path):
                logger.info('Category "{}" already done for window={}'.format(self.cat,self.window))
                return 0

            df = pd.read_pickle(fi).dropna(subset=['abstract_parsed'])
            if len(df)==0:
                logger.info('No data for category "{}"'.format(self.cat))
                return 0

            if self.args.null_model_mode in ('global','local'):
                # generate word distributions 
                word_dists = np.zeros((25,len(self.vocab)))
                for year,grp in df.groupby('year'):
                    word_dists[year-1991] = self.termcounts(grp.abstract_parsed)

                if self.args.null_model_mode == 'global':

                    with timed('Running null model for {} (window={})'.format(self.cat,self.window),logger=logger):
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
                with timed('Calculating raw measures for {} (window={})'.format(self.cat,self.window),logger=logger):
                    ents,ent_difs,jsds = self.calc_measures(word_dists)
            
                dist_path = '{}termdist_{}.npy'.format(self.args.output,self.cat)
                if not os.path.exists(dist_path):
                    np.save(dist_path,word_dists)   

            elif self.args.null_model_mode == 'fixed':

                ent_result = []
                ent_dif_result = []
                jsd_result = []

                with timed('Sampling measures for {} (window={})'.format(self.cat,self.window),logger=logger):
                    sample_size = int(round(df.year.value_counts().min() * self.args.min_prop)) 
                    df['abstract_parsed'] = df.abstract_parsed.apply(lambda x: [word for word in x.split() if word in self.vocabset])
                    self.sample_size_tokens = int(round(df.groupby('year').apply(lambda grp: grp.abstract_parsed.apply(lambda x: len(x)).mean()).min()))
                    logger.info('Fixed sample size DOCUMENTS for category {} = {}'.format(self.cat,sample_size))
                    logger.info('Fixed sample size TOKENS for category {} = {}'.format(self.cat,self.sample_size_tokens))
                    for i in range(self.args.null_bootstrap_samples):
                        sampled = df.groupby('year').apply(lambda x: x.sample(n=sample_size))
                        # generate word distributions 
                        word_dists = np.zeros((25,len(self.vocab)))
                        for year,grp in sampled.groupby('year'):
                            word_dists[year-1991] = self.termcounts(grp.abstract_parsed)
                        ents,ent_difs,jsds = self.calc_measures(word_dists)
                        ent_result.append(ents)
                        ent_dif_result.append(ent_difs)
                        jsd_result.append(jsds)

            
            with timed('Writing results for {} (window={})'.format(self.cat,self.window),logger=logger):
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
    parser.add_argument("-o", "--output", help="output path for results",default=None)
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=100)
    parser.add_argument("-d", "--datadir",help="root input data directory",default=None,type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    parser.add_argument("-n", "--null_model_mode",help='null model mode ("global" or "local")',default='fixed',type=str,choices=['global','local','fixed'])
    parser.add_argument("-r", "--min_prop",help='pRoportion of year with least publications to establish fixed sample size. If we pass a value greater than 1, convert to int and treat as discrete number of documents to smaple.',default=0.5,type=float)
    parser.add_argument("-s","--data_source",help="Source of data (pickeld WoS abstracts, or Elsevier data in gzipped text files",default='wos',choices=['wos','elsevier'])



    args = parser.parse_args()
    # set conditional argument defaults
    if args.datadir is None:
        if args.data_source == 'wos':
            args.datadir = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/by-cat_wos/'
        elif args.data_source == 'elsevier':
            args.datadir = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/metadata/'
    if args.min_prop>1:
        args.min_prop = int(args.min_prop)
    if args.output is None:
        args.output = 'E:/Users/jjl2228/WoS/wos-text-dynamics/results/by-cat-results-{}/'.format(args.data_source)

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
        for f in tq(glob.glob('E:/Users/jjl2228/WoS/wos-text-dynamics-data/termcounts_{}/global_term_counts_formatted_*'.format(args.data_source))):
            for line in open(f,encoding='utf8'):
                term,cnt = line.strip().split(',')
                cnt = int(cnt)
                vocab_dict[term] = vocab_dict.get(term,0)+cnt

        # for fpath in tq(glob.glob(args.datadir+'*.pkl')):
        #     df = pd.read_pickle(fpath)
        #     for abstract in tq(df.abstract):
        #         for term in abstract.split():
        #             vocab_dict[term] = vocab_dict.get(term,0)+1
        #raw_term_counts = pd.Series(vocab_dict)  

        # this is now handled in data-prep.py
        #stemmer = EnglishStemmer()
        #stop = set(stopwords.words('english'))
        #stop = stop.union([stemmer.stem(s) for s in stop])
        #pruned = raw_term_counts[raw_term_counts>=args.vocab_thresh]
        #vocab = sorted([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])

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

    files = glob.glob(args.datadir+'*.pkl')

    
    for w in window_range:

        processor = process(vocab=vocab,window=w,args=args)
        if args.data_source == 'wos':
            pool.map(processor.parse_cat_wos,files,chunksize=len(files)//args.procs)
        elif args.data_source == 'elsevier':
            pool.map(processor.parse_cat_elsevier,files,chunksize=len(files)//args.procs)

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


