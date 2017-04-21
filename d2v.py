from gensim.models.doc2vec import Doc2Vec,TaggedLineDocument,TaggedDocument
from gensim import utils
import gzip,os,glob,time,datetime,sys#,redis
import numpy as np
from tqdm import tqdm as tq
from nltk.tokenize import word_tokenize
import multiprocess as mp


#base_dir = 'E:/Users/jjl2228/WoS/wos-text-dynamics-data/elsevier/author2vec/'
base_dir = 'P:/Projects/WoS/wos-text-dynamics-data/'
#base_dir = '/backup/home/jared/storage/wos-text-dynamics-data/author2vec/'
#base_dir = '/backup/home/jared/storage/wos-text-dynamics-data/'
text_dir = base_dir+'docs_all/' 
d2v_dir = base_dir+'d2v-wos/'

#r = redis.StrictRedis(host='localhost', port=9999, db=0)




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

# custom class to parse documents 
class custom_TLD(TaggedLineDocument):
    def __init__(self, source):

        if source.endswith('/'):
            self.files = sorted(glob.glob(source+'*'))
        else:
            self.files = [source]
    def __iter__(self):
        item_no = -1
        for fi in self.files:            
            #with utils.smart_open(fi) as fin:
            with gzip.open(fi) as fin:
                for line in fin:
                    item_no += 1
                    #if item_no == 100000:
                    #     break
                    yield TaggedDocument(line.decode('utf8').lower().split(), [item_no])
            #break

class list_TLD(TaggedLineDocument):
    def __init__(self, source):
        self.source = source
    def __iter__(self):
        for item_no,doc in enumerate(self.source):
            yield TaggedDocument(doc.split(), [item_no])



def normalize_text(text):
    return word_tokenize(text.strip().lower().replace('|', ' '))


if __name__ == '__main__':
    if len(sys.argv)>1:
        args = sys.argv[1:]
        size,window,min_count,workers,preprocess = map(int,args)

    else:
        size= 200
        window = 5
        min_count = 5
        workers = 24
        preprocess = False
        sample=0



if preprocess:

    pool = mp.Pool(25)

    def process_year(year):
        import gzip
        from nltk.tokenize import word_tokenize
        abs_dir = 'P:/Projects/WoS/WoS/parsed/abstracts/'
        d2v_dir = 'P:/Projects/WoS/WoS/parsed/abstracts/d2v/'
        result = []
        with gzip.open('{}uid_indices/{}.txt.gz'.format(d2v_dir,year),'wb') as idx_out:
            for line in gzip.open('{}{}.txt.gz'.format(abs_dir,year)):
                uid,text = line.decode('utf8').split('\t')
                normed = ' '.join(word_tokenize(text.strip().lower().replace('|', ' ')))
                if normed == "":
                    continue
                #docs.write((normed+'\n').encode('utf8'))
                idx_out.write((uid+'\n').encode('utf8'))
                result.append(normed)
        return '\n'.join(result)


    with gzip.open(d2v_dir+'docs.txt.gz','wb') as docs:
        #for year in tq(range(1991,2016)):
        for r in tq(pool.imap(process_year,range(1991,2016))):
            docs.write((r+'\n').encode('utf8'))

    pool.terminate()
    pool.close()

    ### populate redis db



#documents = custom_TLD(d2v_dir+'docs.txt.gz')
#%time documents = [doc for doc in tq(custom_TLD(text_dir))]
#documents = custom_TLD(text_dir)
documents = TaggedLineDocument(d2v_dir+'docs.txt.gz')


# docs = []
# for fi in tq(sorted(glob.glob(text_dir+'*'))):
#     #docs += gzip.open(fi).readlines()
#     docs += [line.decode('utf8').strip().lower() for line in tq(gzip.open(fi))]

# documents = list_TLD(docs)

            


with timed('Running Doc2Vec'):
    model = Doc2Vec(documents, dm=1, sample=sample, size=size, window=window, min_count=min_count,workers=workers)

with timed('Norming vectors'):
    from sklearn.preprocessing import Normalizer
    nrm = Normalizer('l2')
    normed = nrm.fit_transform(model.docvecs.doctag_syn0)
    words_normed = nrm.fit_transform(model.wv.syn0)

with timed('Saving data'):
    pathname = "{}-{}-{}-{}".format(size,window,min_count,sample)
    if not os.path.exists(d2v_dir+pathname):
        os.mkdir(d2v_dir+pathname)
    np.save('{0}{1}/doc_features_normed_{1}.npy'.format(d2v_dir,pathname),normed)
    np.save('{0}{1}/word_features_normed_{1}.npy'.format(d2v_dir,pathname),words_normed)
    model.save('{0}{1}/model_{1}'.format(d2v_dir,pathname))
