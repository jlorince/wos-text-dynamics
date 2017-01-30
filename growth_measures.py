import pandas as pd
import multiprocessing as mp
from tqdm import tqdm as tq

all_years = range(1991,2016)
debug = None

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
            print '{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad)
        else:
            print '{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad)

def process_year(year):
    with timed('Processing {}'.format(year)):
        with timed('Loading data (year={})'.format(year)):
            cats_current = pd.read_table('P:/Projects/WoS/WoS/parsed/subjects/{}.txt.gz'.format(year),header=None,names=['uid','heading','subheading','categories'],usecols=['uid','categories'],nrows=debug)#.dropna() 
            authors_current = pd.read_table('P:/Projects/WoS/WoS/parsed/authors/{}.txt.gz'.format(year),header=None,names=['uid','author_ids','author_names','affil','affil_idx'],usecols=['uid','author_ids'],nrows=debug)
            citations_current = pd.read_table('P:/Projects/WoS/WoS/parsed/citations/{}.txt.gz'.format(year),header=None,names=['uid','year','citing_papers'],nrows=debug)

        total_pubs = len(cats_current)

        d_pubs = {}
        d_citations = {year:{} for year in all_years}
        d_authors = {}

        with timed('Getting pub counts (year={})'.format(year)):
            for cats in cats_current.categories.dropna():
                for cat in cats.split('|'):
                    d_pubs[cat] = d_pubs.get(cat,0)+1
        
        with timed('Getting citation counts (year={})'.format(year)):
            citations_current['citing_papers'] = citations_current['citing_papers'].apply(lambda x: len(x.split('|')))
            total_citations = citations_current.citing_papers.sum()
            
            merged = cats_current.merge(citations_current)

            for i, row in merged.dropna(subset=['categories']).iterrows():
                for cat in row.categories.split('|'):
                    try:
                        d_citations[row.year][cat] = d_citations[row.year].get(cat,0)+1
                    except KeyError:
                        continue
            
        with timed('Getting author counts (year={})'.format(year)):
            merged = cats_current.merge(authors_current)

            for i, row in merged.dropna(subset=['categories']).iterrows():
                for cat in row.categories.split('|'):
                    if cat in d_authors:
                        [d_authors[cat].add(a) for a in row.author_ids.split('|')]
                    else:
                        d_authors[cat] = set(row.author_ids.split('|'))

            total_authors = len(set.union(*d_authors.values()))
            d_authors = {v:sum(k) for v,k in d_authors.iteritems()}

        return d_pubs,d_citations,d_authors,total_pubs,total_citations,total_authors


if __name__=='__main__':
    pool = mp.Pool(25)
    with timed('Parallel processing'):
        d_pubs,d_citations,d_authors,total_pubs,total_citations,total_authors = zip(*pool.map(process_year,all_years)) #zip(*[process_year(y) for y in (1991,1992)])#
    
    with timed('Buidling final dictionaries'):
        d_pubs_final = dict(zip(all_years,d_pubs))
        d_authors_final = dict(zip(all_years,d_authors))
        d_citations_final = {year:{} for year in all_years}
        for d in d_citations:
            for year in d:
                for cat in d[year]:
                    d_citations_final[year][cat] = d_citations_final[year].get(cat,0)+d[year][cat]

    ddir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/'
    with timed('finalizing dataframes'):
        cat_pubs_by_year = pd.DataFrame(d_pubs_final).T        
        cat_pubs_by_year['ALL'] = total_pubs
        cat_pubs_by_year.to_pickle(ddir+'cat_pubs_by_year.pkl')

        cat_citations_by_year = pd.DataFrame(d_citations_final).T        
        cat_citations_by_year['ALL'] = total_citations
        cat_citations_by_year.to_pickle(ddir+'cat_citations_by_year.pkl')

        cat_authors_by_year = pd.DataFrame(d_authors_final).T        
        cat_authors_by_year['ALL'] = total_authors
        cat_authors_by_year.to_pickle(ddir+'cat_authors_by_year.pkl')

