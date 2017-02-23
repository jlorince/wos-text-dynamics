import numpy as np
import gzip,time,datetime,string,signal,sys,pickle,codecs,csv,glob,unicodedata
import pandas as pd
import multiprocess as mp
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm as tq
import redis

def logger_setup():
    pid = os.getpid()
    now = datetime.datetime.now()
    log_filename = now.strftime('data_prep_%Y%m%d_%H%M%S_{}.log.part'.format(pid))
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


r = redis.StrictRedis(host='localhost', port=6379, db=0)

translator = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
# punct_string = ''
# for i in range(sys.maxunicode):
#     if unicodedata.category(chr(i)).startswith('P'):
#         punct_string += chr(i)

stemmer = EnglishStemmer()
stop = set(stopwords.words('english'))
stop = stop.union([stemmer.stem(s) for s in stop])


class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        logger.info('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            logger.info('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            logger.info('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.items()]),self.pad))


def parse_text(line):
    line = line.strip().split('\t')
    if len(line)==1:
        return 0
    elif len(line)==2:
        uid,rawtext = line
        words  = np.array(word_tokenize(rawtext))
        indices = np.where((np.char.endswith(words[:-1],'-'))&(words[:-1]!='-'))[0]
        dehyphenated = [a[:-1]+b for a,b in zip(words[indices],words[indices+1])]
        words[indices] = dehyphenated
        words = np.delete(words,indices+1)

        # lowercase and remove punctuation
        #translator = str.maketrans('', '', string.punctuation)
        words = np.char.lower(np.char.translate(words,translator))


        # remove all words that are purely alpha[are purely numeric]
        #words = words[~np.char.isnumeric(words)]
        words = words[(np.char.isalpha(words))&(np.char.str_len(words)>=3)]

        # apply stemming
        result = []
        for w in words:
            w = stemmer.stem(w)
            if w not in stop:
                result.append(w)
        if len(words)>0:
            r.set(uid,' '.join(result))
            return 1
        return 0

    else:
        raise Exception('Too many fields?')

def wrapper(f):
    with timed('Processing file {}'.format(f)):
        for i,line in enumerate(gzip.open(f),1):
            parse_text(line.decode('utf8'))
            if i%1000==0:
                logger.info("{}: {} lines processed (overall: {})".format(f,i,r.dbsize()))

        
if __name__=='__main__':

    #  df_metadata = pd.concat([pd.read_table(f,header=None,compression='gzip') for f in ['metadata_{}'.format(i) for i in range(24)]]).dropna()
    #  df_metadata.columns = ['el_id','wos_id','abstract_text','formatted_text','raw_text']

    # for year in tq(range(1950,2016)):

    files = glob.glob('E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/raw/matched/text_*')
    pool = mp.Pool(len(files))

    with timed('Parallel processing'):
        pool.map(wrapper,files)
    try:
        pool.terminate()
        pool.close()
    except:
        pass
