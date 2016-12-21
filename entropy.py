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

def shuffler(n1,combined,n2=None):
    #combined = np.sum(dists,0)
    probs = combined/float(combined.sum())
    div_result = []
    ent_result = []
    for nr_idx in xrange(args.null_bootstrap_samples):
        new1 = np.random.multinomial(n1,probs)
        if n2 is not None:
            new2 = np.random.multinomial(n2,probs)
            div_result.append(thoth.calc_jsd(new1,new2, 0.5, args.thoth_mc_samples)[0])
        ent_result.append(thoth.calc_entropy(new1, args.thoth_mc_samples)[0])
    if n2 is None:
        return np.percentile(np.array(ent_result),[5,50,95])
    else:
        return np.percentile(np.array(ent_result),[5,50,95]),np.percentile(np.array(div_result),[5,50,95])

# window: integer, number of years before and/or after current year
# side: string, 'before','after', or 'both'
def windowed_null_measures(dist_tuple,window=1,side='before'):
    cat,dists = dist_tuple
    last = None
    output ={}

    for i,(year,d) in enumerate(zip(xrange(1991,2016),dists)):
        with timed('null {}-->{}'.format(cat,year)):
            output[year] = {'jsd':np.array([np.nan]*3),'H':np.array([np.nan]*3)}
            if type(d) is not float:
                combine = d.copy()

                if side in ('before','both'):
                    for idx in xrange(1,window+1):
                        rel = i-idx
                        if rel>=0:
                            if dists[rel] is not np.nan:
                                combine += dists[rel]
                        else:
                            break
                if side in ('after','both'):
                    for idx in xrange(1,window+1):
                        rel = i+idx
                        if rel<len(dists):
                            if dists[rel] is not np.nan:
                                combine += dists[rel]
                        else:
                            break
                
                if (combine.sum()>0) and (d.sum()>0):
                    if last is not None:
                        ent,div = shuffler(n1=d.sum(),n2=last.sum(),combined=combine)
                        output[year]['jsd'] = div
                        output[year]['H'] = ent
                    else:
                        ent = shuffler(n1=d.sum(),n2=None,combined=combine)
                        output[year]['H'] = ent
                last = d
            else:
                last = None

    return cat,output



def process_grp(fi):
    cat = fi[fi.rfind('/')+1:-4]
    with timed('base calculations for {}'.format(cat)):
        df = pd.read_pickle(fi)
        # calculate raw entropy and divergence measures by year
        output = {}
        dists = []
        last = None
        cumulative = np.zeros(len(vocab))
        for year in xrange(1991,2016):
            with timed('raw {}-->{}'.format(cat,year)):
                year_df = df[df.year==year]
                output[year] = {'jsd':np.array([np.nan]*5),'jsd_c':np.array([np.nan]*5),'H':np.array([np.nan]*5),'H_c':np.array([np.nan]*5)}
                n = len(df)
                output[year]['n'] = n
                if n>0:
                    current = termcounts(year_df.abstract)
                    if last is not None:
                        output[year]['jsd'] = thoth.calc_jsd(current,last,0.5,args.thoth_mc_samples)
                        output[year]['jsd_c'] = thoth.calc_jsd(current,cumulative,0.5,args.thoth_mc_samples)
                    if current.sum()>0:
                        output[year]['H'] = thoth.calc_entropy(current,args.thoth_mc_samples)
                        cumulative += current
                        output[year]['H_c'] = thoth.calc_entropy(cumulative,args.thoth_mc_samples)
                        last = current
                        dists.append(current)
                    else:
                        last = None
                        dists.append(np.nan)
                else:
                    last = None
                    dists.append(np.nan)

        return cat,output,dists

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
    return cat,dists



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-r", "--raw", help="calculate raw entropy/divergence measures",action="store_true")
    parser.add_argument("-n", "--null", help="calculate null model entropy/divergence measures",action="store_true")
    parser.add_argument("-w", "--window", help="window size for null model calculations",type=int,default=1)
    parser.add_argument("-s", "--side", help="which sides of current year in which to extend window for null model calculations",type=str,default='before',choices=['before','after','both'])
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='/backup/home/jared/storage/wos-text-dynamics-data/results/')
    parser.add_argument("-t", "--thoth_mc_samples", help="Number of monte carlo samples for thoth calculations",type=int,default=10000)
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=10000)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='/backup/home/jared/storage/wos-text-dynamics-data/',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    parser.add_argument("-x", "--debug", help="enable mode", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.null_bootstrap_samples = 5
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

        with timed('pool setup'):
            ### file setup
            files = glob("{}by-cat/*".format(args.datadir))
            if args.debug:
                files = files[:3]

            ### pool setup
            chunksize = int(math.ceil(len(files) / float(args.procs)))
            pool = mp.Pool(args.procs)

        if args.raw:
            with timed('parallel processing, raw measures'):
                results = pool.map(process_grp,files)
            with timed('writing results, raw measures'):
                all_dists = []
                with open(args.output+'raw_results','w') as fout:
                    for cat,output,dists in results:
                        all_dists.append((cat,dists))
                        for year in output:
                            fout.write('\t'.join([cat,str(year),str(output[year]['n'])]+[','.join(output[year][measure].astype(str)) for measure in ('jsd','jsd_c','H','H_c')])+'\n')
        else:
            ### just generate dists, assuming we've already done our other computations
            with timed('parallel dist generation'):
                all_dists = pool.map(gen_dists,files) 
                                    

        if args.null:
            for window in (1,2,3,4,5,25):
                with timed('parallel processing, null model (window={}, side={})'.format(window,'both')):
                    #func_partial = partial(windowed_null_measures,window=args.window,side=args.side)
                    func_partial = partial(windowed_null_measures,window=window,side='both')
                    null_results = pool.map(func_partial,all_dists)
                    with timed('writing results, null models'):
                        with open(args.output+'null_results_{}_{}'.format(args.window,args.side),'w') as fout:
                            for cat,output in null_results:
                                for year in output:
                                    fout.write('\t'.join([cat,str(year)]+[','.join(output[year][measure].astype(str)) for measure in ('jsd','H')])+'\n')
