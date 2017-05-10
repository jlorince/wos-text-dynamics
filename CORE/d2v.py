from gensim.models.doc2vec import Doc2Vec,TaggedLineDocument,TaggedDocument
from gensim import utils
import gzip,os,glob,time,datetime,sys,argparse,platform
import numpy as np
from tqdm import tqdm as tq
from math import floor
from nltk.tokenize import word_tokenize
import multiprocess as mp

help_string="""
THIS CODE IS ONLY TESTED AGAINST PYTHON 3.6!!!

This script generates Doc2Vec models for a corpus of documents. Some, but not all, d2v options are available.

All model results are saved in a folder with name generated by the following (see arguments below):

"{}-{}-{}-{}-{}".format(size,window,min_count,sample,seed)

Doc2Vec documentation: https://radimrehurek.com/gensim/models/doc2vec.html

Note that inference on unseen documents (when using year_sample) currently runs pretty slow. We might want to think of a workaround.
"""


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
            print('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))

### Modified TaggedLineDocument Class
# rather than iterating over all documents in the docs file, only yields those indexed by `include_set`
# (used if we do by-year sampling)
class sampling_tld(TaggedLineDocument):
    def __init__(self,source,include_set):
        self.source = source
        self.include_set = include_set
        self.n_docs = None
    def __iter__(self):
        idx = 0
        with utils.smart_open(self.source) as fin:
            for i,line in enumerate(fin):
                if i in self.include_set:
                    yield TaggedDocument(utils.to_unicode(line).split(), [idx])
                    idx+=1
        if self.n_docs is None:
            self.n_docs = i+1
    def iter_skipped(self):
        with utils.smart_open(self.source) as fin:
            for i,line in enumerate(fin):
                if i not in self.include_set:
                    yield i,utils.to_unicode(line).split()


def normalize_text(text):
    return ' '.join(word_tokenize(text.strip().lower().replace('|', ' ')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(help_string)
    parser.add_argument("--size", help="Dimensionality of D2V vectors (see Doc2Vec documentation).",type=int,default=100)
    parser.add_argument("--window", help="Doc2Vec window size (see Doc2Vec documentation).",type=int,default=5)
    parser.add_argument("--min_count", help="Mininum number of times a word must occur to be included in model (see Doc2Vec documentation).",type=int,default=5)
    parser.add_argument("--sample", help="Threshold for configuring which higher-frequency words are randomly downsampled (see Doc2Vec documentation).",type=float,default=0)
    parser.add_argument("--workers", help="Number of workers to use in parallel computations. Defaults to output of mp.cpu_count()",type=int,default=mp.cpu_count())
    parser.add_argument("--preprocess", action='store_true',help="Perform initial preprocessing of raw data files.")
    parser.add_argument("--year_sample", action='store_true',help="If provided, randomly sample an equal number of documents from each year.")
    parser.add_argument("--norm", action='store_true',help="If provided, also generate l2-normed word and document vector arrays.")
    parser.add_argument("--raw_text_dir", help="Location of the raw text files. Assumes there is one file per year, and eah file has one line per document in the format 'ID <tab> text...'. Argument ignored if `preprocess` not set to true.",type=str,default='P:/Projects/WoS/parsed/abstracts/')
    parser.add_argument("--d2v_dir", help="Output directory. A new subfolder in this directory will be generated for each new model run",type=str,default='P:/Projects/WoS/wos-text-dynamics-data/d2v-wos/')
    parser.add_argument("--infer",help="If year_sample is true, specify this option infer document vectors for documents that were not included in model training.",action='store_true')


    args = parser.parse_args()

    # in case user forgets trailing slash
    if not args.d2v_dir.endswith('/'):
        args.d2v_dir = args.d2v_dir + '/'


    if args.preprocess:

        pool = mp.Pool(min(25,mp.cpu_count()))

        def process_year(year):
            import gzip
            from nltk.tokenize import word_tokenize
            result = []
            with gzip.open('{}uid_indices/{}.txt.gz'.format(args.d2v_dir,year),'wb') as idx_out:
                for line in gzip.open('{}{}.txt.gz'.format(args.raw_text_dir,year)):
                    uid,text = line.decode('utf8').split('\t')
                    normed = normalize_text(text)
                    if normed == "":
                        continue
                    idx_out.write((uid+'\n').encode('utf8'))
                    result.append(normed)
            return '\n'.join(result)


        with gzip.open(args.d2v_dir+'docs.txt.gz','wb') as docs:
            for r in tq(pool.imap(process_year,range(1991,2016))):
                docs.write((r+'\n').encode('utf8'))

        pool.terminate()
        pool.close()


    if args.year_sample:
        #### Generate Randomized year-matched samples
        index_years = np.load(args.d2v_dir+'index_years.npy')
        seed = np.random.randint(999999)
        print('----RANDOM SEED = {}----'.format(seed))
        np.random.seed(seed)

        years,counts = np.unique(index_years,return_counts=True)
        indices_to_write = []
        for year in tq(years):
            indices_to_write.append(np.random.choice(np.where(index_years==year)[0],counts.min(),replace=False))
        indices_to_write = np.concatenate(indices_to_write)
        index_set = set(indices_to_write)

        # with gzip.open(args.d2v_dir+'docs.txt.gz') as fin,\
        #      gzip.open(args.d2v_dir+'docs_sampled_{}.txt.gz'.format(seed)) as fout,\
        #      gzip.open(args.d2v_dir+'docs_heldout_{}.txt.gz'.format(seed)) as holdut:
        #     for i,line in enumerate(fin):
        #         if i in indices_to_write:
        #             fout.write(line)
        #         else:
        #             holdout.write(line)

        documents = sampling_tld(args.d2v_dir+'docs.txt.gz',index_set)

    else:

        seed = None
        documents = TaggedLineDocument(args.d2v_dir+'docs.txt.gz')

    pathname = "{}-{}-{}-{}-{}".format(args.size,args.window,args.min_count,args.sample,seed)
    if os.path.exists(args.d2v_dir+pathname):
        raise Exception("It appears this model has already been run.")
    else:
        os.mkdir(args.d2v_dir+pathname)
    if args.year_sample:
        np.save('{}{}/doc_indices_sampled_{}.npy'.format(args.d2v_dir,pathname,seed),indices_to_write)
           


    with timed('Running Doc2Vec'):
        model = Doc2Vec(documents, dm=1, sample=args.sample, size=args.size, window=args.window, min_count=args.min_count,workers=args.workers)


    if args.year_sample and args.infer:
        with timed('Inferring vectors for unseen documents'):
            expanded_docvecs = np.empty((documents.n_docs,args.size))
            expanded_docvecs[indices_to_write] = model.docvecs.doctag_syn0
        
            #--------------------
            # Parallel version has been problematic and is disabled for now
            #--------------------
            
            # if platform.system()!='Windows':
            #     def wrapper(tup):
            #         i,doc = tup
            #         return i,model.infer_vector(doc)
            #     pool = mp.Pool(args.workers)
            #     total=documents.n_docs-len(indices_to_write)
            #     start = time.time()
            #     chunksize = floor(total/args.workers)
            #     for i,(idx,docvec) in enumerate(pool.imap_unordered(wrapper,documents.iter_skipped(),chunksize=chunksize),1):
            #         if i%chunksize==0:
            #             eta = ((time.time()-start)/i) * (total-i)
            #             print("{}/{} documents processed ({:.2f}%) - time elapsed: {} - eta: {}".format(i,total,100*(i/total),str(datetime.timedelta(seconds=time.time()-start)),str(datetime.timedelta(seconds=eta))))
            #         expanded_docvecs[idx] = docvec
            #     pool.terminate()
            # else:
            for i,doc in tq(documents.iter_skipped(),total=documents.n_docs-len(indices_to_write)):
                expanded_docvecs[i] = model.infer_vector(doc)

    if args.norm:
        with timed('Norming vectors'):
            from sklearn.preprocessing import Normalizer
            nrm = Normalizer('l2')
            normed = nrm.fit_transform(model.docvecs.doctag_syn0)
            if args.year_sample:
                normed_expanded = nrm.fit_transform(expanded_docvecs)
            words_normed = nrm.fit_transform(model.wv.syn0)


    with timed('Saving data'):        
        if args.norm:
            np.save('{0}{1}/doc_features_normed_{1}.npy'.format(args.d2v_dir,pathname),normed)
            np.save('{0}{1}/word_features_normed_{1}.npy'.format(args.d2v_dir,pathname),words_normed)
        model.save('{0}{1}/model_{1}'.format(args.d2v_dir,pathname))
        if args.year_sample:
            np.save('{0}{1}/doc_features_expanded_{1}.npy'.format(args.d2v_dir,pathname),expanded_docvecs)
            if args.norm:
                np.save('{0}{1}/doc_features_expanded_normed_{1}.npy'.format(args.d2v_dir,pathname),normed_expanded)
