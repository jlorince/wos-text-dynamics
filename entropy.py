import pathos.multiprocessing as mp
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter
from functools import partial
import sys,logging,time,datetime,math,argparse
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
sys.path.append('/backup/home/jared/thoth')
#import thoth.thoth as thoth

vocab_thresh = 100
thoth_mc_samples = 1000
data_dir = '/backup/home/jared/storage/wos-text-dynamics-data/'


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

def shuffler(n1,combined,nr=1000,n2=None):
    #combined = np.sum(dists,0)
    probs = combined/float(combined.sum())
    div_result = []
    ent_result = []
    for nr_idx in xrange(nr):
        new1 = np.random.multinomial(n1,probs)
        if n2 is not None:
            new2 = np.random.multinomial(n2,probs)
            div_result.append(thoth.calc_jsd(new1,new2, 0.5, thoth_mc_samples)[0])
        ent_result.append(thoth.calc_entropy(new1, thoth_mc_samples)[0])
    if n2 is None:
        np.percentile(np.array(ent_result),[5,50,95])
    else:
        return np.percentile(np.array(ent_result),[5,50,95]),np.percentile(np.array(div_result),[5,50,95])

# window: integer, number of years before and/or after current year
# side: string, 'before','after', or 'both'
def windowed_null_measures(dists,window=1,side='before'):
    last = None
    output ={}

    for i,(year,d) in enumerate(zip(xrange(1991,2016),dists)):
        combine = d.copy()
        if side in ('before','both'):
            for idx in xrange(1,window+1):
                rel = i-idx
                if rel>=0:
                    if dists[rel] is not np.nan:
                        combine += dist[rel]
                else:
                    break
        if side in ('after','both'):
            for idx in xrange(1,window+1):
                rel = i+idx
                if rel<len(dists):
                    if dists[rel] is not np.nan:
                        combine += dist[rel]
                else:
                    break
        output[year] = {}
        if (combine.sum()>0) and (d.sum()>0):
            if last is not None:
                div,ent = shuffler(n1=d.sum(),n2=last.sum(),combined=combine)
                output[year]['jsd'] = div
                output[year]['H'] = ent
            else:
                ent = shuffler(n1=d.sum(),n2=None,combined=combine)
                output[year]['jsd'] = np.nan
                output[year]['H'] = ent
        else:
            output[year]['jsd'] = np.nan
            output[year]['H'] = np.nan

    return output



def process_grp(fi):
    cat = fi[fi.rfind('/')+1:]
    with timed('base calculations for {}'.format(cat)):
        df = pd.read_pickle(fi)
        # calculate raw entropy and divergence measures by year
        output = {}
        dists = []
        last = None
        cumulative = np.zeros(len(vocab))
        #for year,year_df in df.groupby('year'):
        for year in xrange(1991,2016):
            year_df = df[df.year==year]
            output[year] = {'jsd':np.nan,'jsd_c':np.nan,'H':np.nan,'H__c':np.nan}
            if len(year_df)>0:
                current = termcounts(year_df.abstract)
                if last is not None:
                    output[year]['jsd'] = thoth.calc_jsd(current,last,0.5,thoth_mc_samples)
                    output[year]['jsd_c'] = thoth.calc_jsd(current,cumulative,0.5,thoth_mc_samples)
                if current.sum()>0:
                    output[year]['H'] = thoth.calc_entropy(current,thoth_mc_samples)
                    cumulative += current
                    output[year]['H_c'] = thoth.calc_entropy(cumulative,thoth_mc_samples)
                    last = current
                    dists.append(current)
                else:
                    last = None
                    dists.append(np.nan)
            else:
                last = None

        return cat,output,dists




if __name__ == '__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=str)
    parser.add_argument("-r", "--raw", help="calculate raw entropy/divergence measures",action="store_true")
    parser.add_argument("-n", "--null", help="calculate null model entropy/divergence measures",action="store_true")
    parser.add_argument("-w", "--window", help="window size for null model calculations",type=int,default=1)
    parser.add_argument("-s", "--side", help="which sides of current year in which to extend window for null model calculations",type=str,default='before',choices=['before','after','both'])
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='/backup/home/jared/storage/wos-text-dynamics-data/results/')
    parser.add_argument("-t", "--thoth_mc_samples", help="Number of monte carlo samples for thoth calculations",type=int,default=10000)
    parser.add_argument("-n", "--null_model_samples", help="Number of monte carlo samples for null model calculations",type=int,default=10000)
    args = parser.parse_args()


    ### LOGGING SETUP
    now = datetime.datetime.now()
    if args.logfile:
        args.logfiles += '_'
    log_filename = now.strftime('{}%Y%m%d_%H%M%S.log'.format(args.logfile))
    logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging_level)

    ### stopword setup
    stop = set(stopwords.words('english'))
    stemmer = EnglishStemmer()
    stop = stop.union([stemmer.stem(s) for s in stop])

    ### vocabulary setup
    global_term_counts = pd.Series.from_csv(data_dir+'global_term_counts.csv',encoding='utf8')
    pruned = global_term_counts[global_term_counts>=vocab_thresh]
    vocab = set([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])

    ### file setup
    files = glob("{}by-cat/*".format(data_dir))

    ### pool setup
    chunksize = int(math.ceil(len(files) / float(procs)))
    pool = mp.Pool(procs)

    if args.raw:
        with timed('parallel processing, raw measures'):
            results = pool.map(process_grp,files)
        with timed('writing results, raw measures')
            all_dists = []
            with open(args.output,'w') as fout:
                for cat,output,dists in results:
                    all_dists.append(dists)
                    for year in output:

    if args.null:
        with timed('parallel processing, null model'):
            func_partial = partial(windowed_null_measures,window=args.window,side=args.side)
            null_results = pool.map(func_partial,files)
