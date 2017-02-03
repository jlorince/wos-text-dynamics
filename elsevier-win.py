#import pymssql
import gzip,codecs,time,datetime,glob,logging,string
from tqdm import tqdm as tq
from lxml import etree
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
import multiprocessing as mp
stemmer = EnglishStemmer()


ddir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/xml/matched/'

mode = 'stem' # 'raw'

if mode =='stem':
    outdir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/stemmed/matched/'
elif mode =='raw':
    outdir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/raw/matched/'


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


def parse_xml(filename):
    tree = etree.parse(filename)

    all_text = [] 
    found_abstract = False
    found_formatted_text = False
    found_rawtext = False
    l=0

    ## Check for raw text
    for rawtext in tree.findall('.//{*}raw-text'):
        if rawtext.text:
            t = rawtext.text.strip()
            if t:
                all_text.append(t)
    # did we find an formatted text?
    if len(all_text)>l:
        found_rawtext = True
    l = len(all_text)

    ## Get abstract, ONLY IF WE DID NOT GET RAW TEXT (which should include abstract already)
    if not found_rawtext:
        for abstract in tree.findall('.//{*}abstract'):
            for node in abstract.iter('*'):
                if node.text:
                    t = node.text.strip()
                    if t:
                        all_text.append(t)
        # did we find an abstract?
        if len(all_text)>l:
            found_abstract = True
        l = len(all_text)

    ## Check for formatted text
    for body in tree.findall('.//{*}body'):
        for node in body.iter('*'):
            if node.text:
                t = node.text.strip()
                if t:
                    all_text.append(t)
    # did we find an formatted text?
    if len(all_text)>l:
        found_formatted_text = True
    l = len(all_text)

    #rawtext = np.array(' '.join(all_text).split())
    rawtext = np.array(word_tokenize(' '.join(all_text)))
    return rawtext,found_abstract,found_formatted_text,found_rawtext

def parse_rawtext(words):
    # handle hyphenated terms (ignoring last term)
    indices = np.where((np.char.endswith(words[:-1],'-'))&(words[:-1]!='-'))[0]
    dehyphenated = [a[:-1]+b for a,b in zip(words[indices],words[indices+1])]
    words[indices] = dehyphenated
    words = np.delete(words,indices+1)

    # lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    words = np.char.lower(np.char.translate(words,translator,string.punctuation))

    # remove all words that are not pure letters
    #words = words[~np.char.isnumeric(words)]
    words = words[np.char.isalpha(words)]

    # apply stemming
    if mode == 'stem':
        words = [stemmer.stem(w) for w in words]

    return words

def wrapper(filename):
    paper_id = filename[filename.rfind('\\')+1:]
    rawtext,found_abstract,found_formatted_text,found_rawtext = parse_xml(filename)
    rawtext_length = len(rawtext)
    if rawtext_length>0:
        parsed = parse_rawtext(rawtext)
        parsed_length = len(parsed)
        with open(outdir+paper_id,'w',encoding='utf8') as fout:
            fout.write(' '.join(parsed)+'\n')
    else:
        parsed_length = 0

    return paper_id,int(found_abstract),int(found_formatted_text),int(found_rawtext),rawtext_length,parsed_length




if __name__=='__main__':


    files = glob.glob(ddir+'*')

    procs = mp.cpu_count()
    pool = mp.Pool(procs)
    chunksize = len(files)//procs

    #pool.map(wrapper,files)
    with open('S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/parse_log','w') as out:
        for result in tq(pool.imap_unordered(wrapper,files),total=len(files)):
            out.write('\t'.join(map(str,result))+'\n')
        


