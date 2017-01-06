import numpy as np
import gzip,time,datetime,string,signal,sys,cPickle,codecs,csv
import pandas as pd
import multiprocessing as mp
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
stemmer = EnglishStemmer()

tmpdir = 'S:/UsersData_NoExpiration/jjl2228/keywords/temp/'

debug = None # set this to some small-ish number to only read part of raw files, for debugging

limit_journals = ['PHYSICAL REVIEW A','PHYSICAL REVIEW B','PHYSICAL REVIEW C','PHYSICAL REVIEW D','PHYSICAL REVIEW E','PHYSICAL REVIEW LETTERS']
limit_journal_key = 'physics'

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
            print '{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad)
        else:
            print '{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad)


def parse_abs(rawtext_arr):
    result = []
    for i,rawtext in enumerate(rawtext_arr):
        if pd.isnull(rawtext):
            result.append('')
        else:
            rawtext = rawtext.translate(None,string.punctuation).decode('utf8').split()
            if len(rawtext)>0:
                cleaned = [stemmer.stem(w) for w in rawtext]
                result.append(' '.join(cleaned))
            else:
                result.append('')
    return result


def process(year):
    with timed(desc=year,pad='----'):
        with timed('metadata loading',year=year):
            md_current = pd.read_table('P:/Projects/WoS/WoS/parsed/metadata/{}.txt.gz'.format(year),header=None, nrows=debug,
                                   names=["uid","date","pubtype","volume","issue","pages","paper_title","source_title","doctype"],
                                  usecols=["uid","pubtype","paper_title","source_title","doctype"])
            md_current = md_current[md_current.source_title.isin(limit_journals)]
        with timed('abstract loading',year=year):
            abs_current = pd.read_table('P:/Projects/WoS/WoS/parsed/abstracts/{}.txt.gz'.format(year),header=None,names=['uid','abstract'], nrows=debug).dropna()
        with timed('abstract parsing',year=year):
            abs_current['abstract'] = parse_abs(abs_current['abstract'].values)
        with timed('keyword loading',year=year):
            #kw_current = pd.read_table('S:/UsersData_NoExpiration/jjl2228/keywords/pubs_by_year/{}.txt.gz'.format(year),header=None,names=['keyword','uid'],nrows=debug)
            kw_current = pd.read_table('P:/Projects/WoS/WoS/parsed/keywords/{}.txt.gz'.format(year),header=None,names=['uid','nk','keywords'],usecols=['uid','keywords'],quoting=csv.QUOTE_NONE,nrows=debug)
        with timed('category loading',year=year):
            cats_current = pd.read_table('P:/Projects/WoS/WoS/parsed/subjects/{}.txt.gz'.format(year),header=None,names=['uid','heading','subheading','categories'], nrows=debug)#.dropna()
        with timed('category formatting',year=year):
            #cats_current = pd.concat([cats_current[['uid','heading','subheading']],cats_current['categories'].apply(gen_series)],axis=1)
            cats_current['categories'] = cats_current['categories'].apply(lambda x: x if pd.isnull(x) else x.split('|'))
        with timed('reference loading',year=year):
            refs_current = pd.read_table('P:/Projects/WoS/WoS/parsed/references/{}.txt.gz'.format(year),header=None,names=['uid','n_refs','refs','missing'],usecols=['uid','refs'], nrows=debug)
        with timed('data merging',year=year):
            current = abs_current.merge(md_current,on='uid',how='inner').merge(cats_current,on='uid',how='inner').merge(refs_current,on='uid',how='left').merge(kw_current,on='uid',how='left')
            current['year'] = year
        print 'final datasize: {} ({})'.format(current.shape,year)
    return current
        
if __name__=='__main__':


    with timed('main data processing',pad=' ######## '):
        with timed('parallel processing'):
            pool = mp.Pool(25)
            result = pool.map(process,xrange(1991,2016))
            print '----result collected----'
            with timed('pool shutdown'):
                try:
                    pool.terminate()
                    pool.close()
                except:
                   print "exception in pool shutdown, but let's keep going..."


    with timed('dataframe concatenation'):
        df = pd.concat(result)
    with timed('dataframe saving'):
        df.to_pickle('S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/{}/all_pubs.pkl')

    with timed('word freq distribution'):
        termdict = {}
        total = len(df)

        for i,row in enumerate(df.abstract.dropna(),1):
            for term in row.split():
                termdict[term] = termdict.get(term,0)+1
            if i%100000==0: 
                print "{}/{} ({}%)".format(i,total,100*(i/float(total)))

        global_term_counts = pd.Series(termdict)
        global_term_counts.to_csv('S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/{}/global_term_counts.csv'.format(limit_journal_key),encoding='utf8')






