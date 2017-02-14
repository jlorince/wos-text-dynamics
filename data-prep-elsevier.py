import numpy as np
import gzip,time,datetime,string,signal,sys,pickle,codecs,csv,glob,unicodedata
import pandas as pd
import multiprocessing as mp
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm as tq
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)

translator = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
punct_string = ''
for i in range(sys.maxunicode):
    if unicodedata.category(chr(i)).startswith('P'):
        punct_string += chr(i)

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
        print('{} started...'.format(self.desc))
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            print('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.items()]),self.pad))


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
        words = np.char.lower(np.char.translate(words,translator,string.punctuation))


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
                print("{}: {} lines processed (overall: {})".format(f,i,r.dbsize()))

        
if __name__=='__main__':

    files = glob.glob('E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/raw/matched/text_*')
    pool = mp.Pool(len(files))

    with timed('Parallel processing'):
        pool.map(wrapper,files)
    try:
        pool.close()
    except:
        pass