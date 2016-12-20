import pathos.multiprocessing as mp
import pandas as pd
import numpy as np
from glob import glob
from collections import Counter
from functools import parial
import sys,logging,time,datetime,math,argparse
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
sys.path.append('/backup/home/jared/thoth')
import thoth.thoth as thoth

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
    result ={}

    for i,(year,d) in enumerate(zip(dists,xrange(1991,2016))):
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
        result[year] = {}
        if (combine.sum()>0) and (d.sum()>0):
            if last is not None:
                div,ent = shuffler(n1=d.sum(),n2=last.sum(),combined=combine)
                result[year]['jsd'] = div
                result[year]['H'] = ent
            else:
                ent = shuffler(n1=d.sum(),n2=None,combined=combine)
                result[year]['jsd'] = np.nan
                result[year]['H'] = ent
        else:
            result[year]['jsd'] = np.nan
            result[year]['H'] = np.nan



def process_grp(fi):
    with timed('base calculations for {}'.format(fi[fi.rfind('/')+1:])):
        df = pd.read_pickle(fi)
        # calculate raw entropy and divergence measures by year
        output = {}
        all_dists = []
        last = None
        cumulative = np.zeros(len(vocab))
        for year,year_df in df.groupby('year'):
            output[year] = {}
            current = termcounts(year_df.abstract)
            if last is not None:
                output[year]['jsd'] = thoth.calc_jsd(current,last,0.5,thoth_mc_samples)
                output[year]['jsd_c'] = thoth.calc_jsd(current,cumulative,0.5,thoth_mc_samples)
                break
            else:
                output[year]['jsd'] = np.nan
                output[year]['jsd_c'] = np.nan
            if current.sum()>0:
                output[year]['H'] = thoth.calc_entropy(current,thoth_mc_samples)
                cumulative += current
                last = current
                all_dists.append(current)
            else:
                last = None
                output[year]['H'] = np.nan
                all_dists.append(np.nan)




if __name__ == '__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations",default=mp.cpu_count(),type=str)
    parser.add_argument("-r", "--raw", help="calculate raw entropy/divergence measures",action="store_true")
    parser.add_argument("-p", "--preprocess", help="perform preprocessing of listening histories",action="store_true")
    parser.add_argument("-r", "--rawtext",help="Load scrobbles from raw text files. If not specififed, assumes files are already pickled and saved in `pickledir`",action="store_true")
    parser.add_argument("-s","--save",help='save newly generated DFs',action="store_true")
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("--datadir", help="specify base directory containing input files",default='/home/jlorince/scrobbles/')
    parser.add_argument("--suppdir", help="specify supplementary data location",default='/home/jlorince/support/')
    parser.add_argument("--resultdir", help="specify results location",default='/home/jlorince/results/')
    parser.add_argument("--session_thresh", help="session segmentation threshold. Use 0 for no time-based segmentation.",type=int,default=None) # 1800
    parser.add_argument("--min_patch_length", help="minimum patch length",type=int,default=None) # 5
    parser.add_argument("--dist_thresh", help="distance threshold defining patch neigborhood",type=float,default=None) # 0.2
    parser.add_argument("-n", help="number of processes in processor pool",type=int,default=1)
    parser.add_argument("--feature_path", help="path to artist feature matrix",default=None) # '/home/jlorince/lda_tests_artists/features_190.numpy'
    parser.add_argument("--distance_metric", help="distance metric",type=str,default='cosine')
    parser.add_argument("--patch_basis", help="If specified, perform patch summaries with the given basis",type=str,choices=['block','patch_idx_shuffle','patch_index_simple'])
    parser.add_argument("--skip_complete", help="If specified, check for existing files and skip if they exist",action='store_true')
    parser.add_argument("--prefix_input", help="inpout file prefix",type=str,default='')
    parser.add_argument("--prefix_output", help="output file prefix",type=str,default='')
    #parser.add_argument("--patch_len_dist", help="compute distribution of patch lengths",default=None,type=str,choices=['shuffle','simple','block','both'])
    parser.add_argument("--patch_len_dist", help="compute distribution of patch lengths",action='store_true')
    parser.add_argument("--blockdists", help="",action='store_true')
    parser.add_argument("--blockgaps", help="",action='store_true')
    parser.add_argument("--scrobblegaps", help="",action='store_true')
    parser.add_argument("--ee_artists",help="",action='store_true')
    parser.add_argument("--ee_artists_2",help="",action='store_true')
    parser.add_argument("--ee_artists_dists",help="",action='store_true')
    parser.add_argument("--block_len_dists",help="",action='store_true')

    args = parser.parse_args()


    ### LOGGING SETUP
    now = datetime.datetime.now()
    log_filename = now.strftime('setup_%Y%m%d_%H%M%S.log')
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

    
    results = pool.map(process_grp,files)
    func_partial = partial(windowed_null_measures,window=)
    null_results = pool.map()
