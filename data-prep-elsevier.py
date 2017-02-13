import numpy as np
import gzip,time,datetime,string,signal,sys,pickle,codecs,csv,glob
import pandas as pd
import multiprocessing as mp
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
stemmer = EnglishStemmer()
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop = stop.union([stemmer.stem(s) for s in stop])
from nltk.tokenize import word_tokenize
from tqdm import tqdm as tq
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)


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
    line = line.decode('utf8').strip().split('\t')
    if len(line)==1:
        return None
    elif len(line)==2:
        uid,rawtext = line
        words  = np.array(word_tokenize(rawtext))
        indices = np.where((np.char.endswith(words[:-1],'-'))&(words[:-1]!='-'))[0]
        dehyphenated = [a[:-1]+b for a,b in zip(words[indices],words[indices+1])]
        words[indices] = dehyphenated
        words = np.delete(words,indices+1)

        # lowercase and remove punctuation
        translator = str.maketrans('', '', string.punctuation)
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
    else:
        raise Excetion('Too many fields?')

def wrapper(f):
    with timed('Processing file {}'.format(f)):
        for i,line in enumerate(gzip.open(f),1):
            parse_text(line)
            if i%1000==0:
                print("{}: {} lines processed (overall: {})".format(f,i,    r.dbsize()))

        
if __name__=='__main__':

    files = glob.glob('E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/raw/matched/text_*')
    pool = mp.Pool(len(files))

    with timed('Parallel processing'):
        pool.map(wrapper,files)
    try:
        pool.close()
    except:
        pass
