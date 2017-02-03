#import pymssql
import gzip,codecs,time,datetime,glob,logging,string
from tqdm import tqdm as tq
from lxml import etree
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
import pathos.multiprocessing as mp
stemmer = EnglishStemmer()


# id_dict = {}
# with gzip.open('data/SD_WoS_id_match.txt.gz') as id_file:
#     for line in tq(id_file,total=total_lines):
#         k,v = line.strip().split()
#         id_dict[int(k)] = v

#namespaces = {"dp":"http://www.elsevier.com/xml/common/doc-properties/schema",'ja':'http://www.elsevier.com/xml/ja/schema','ce':'http://www.elsevier.com/xml/common/schema'} 


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

class xml_parser(object):
    def __init__(self,filename,logger,outdir):
        self.filename = filename
        self.paper_id = filename[filename.rfind('/')+1:]
        self.logger = logger
        self.outdir = outdir

    def parse_xml(self):
        tree = etree.parse(self.filename)

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

        #self.words  = np.array(' '.join(all_text).split())
        self.words  = np.array(word_tokenize(' '.join(all_text)))

        self.logger.info("{}: Raw XML processed -- abstract={:d},formatted_text={:d},rawtext={:d} -- wordcount={}".format(
                        self.paper_id,found_abstract,found_formatted_text,found_rawtext,len(self.words)))


    def parse_rawtext(self):

        # handle hyphenated terms (ignoring last term)
        indices = np.where((np.char.endswith(self.words[:-1],'-'))&(self.words[:-1]!='-'))[0]
        dehyphenated = [a[:-1]+b for a,b in zip(self.words[indices],self.words[indices+1])]
        self.words[indices] = dehyphenated
        self.words = np.delete(self.words,indices+1)

        # lowercase and remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        self.words = np.char.lower(np.char.translate(self.words,translator,string.punctuation))

        # remove all words that are purely alpha[are purely numeric]
        #self.words = self.words[~np.char.isnumeric(self.words)]
        self.words = self.words[np.char.isaplha(self.words)]

        # apply stemming
        self.words = [stemmer.stem(w) for w in self.words]

        self.logger.info("{}: Text cleaning complete -- wordcount={}".format(self.paper_id,len(self.words)))

    def write_output(self):
        with open(self.outdir+self.paper_id,'w' ) as fout:
            fout.write(' '.join(self.words)+'\n')

    def process_file(self):
        self.parse_xml()
        if len(self.words)>0:
            self.parse_rawtext()
        if len(self.words)>0:
            self.write_output()






if __name__=='__main__':
    # LOGGING SETUP
    now = datetime.datetime.now()
    log_filename = now.strftime('%Y%m%d_%H%M%S.log')
    logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    #consoleHandler = logging.StreamHandler()
    #consoleHandler.setFormatter(logFormatter)
    #rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    ddir = '/backup/home/jared/storage/elsevier/raw/matched/'
    outdir = '/backup/home/jared/storage/elsevier/parsed/matched/'

    files = glob.glob(ddir+'*')

    def wrapper(fi):
        processor = xml_parser(filename=fi,logger=rootLogger,outdir=outdir)
        processor.process_file()

    procs = mp.cpu_count()
    pool = mp.Pool(procs)
    chunksize = len(files)//procs

    #pool.map(wrapper,files)
    for _ in tq(pool.imap_unordered(wrapper,files),total=len(files)):
        pass


"""
## just the dump code I wrote to (slooooooowly) get elsevier data off the DB
ddir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/raw/'
total_lines=7775842
total_el_recs=12101870

### Server setup
server,user,password = [line.strip() for line in open('server_credentials.txt')]
conn = pymssql.connect(server, user, password, "tempdb")
cursor = conn.cursor()

# with gzip.open('data/SD_WoS_id_match.txt.gz') as id_file:
#     for line in tq(id_file,total=total_lines):
#         el_id,wos_id = line.strip().split()
#         cursor.execute("SELECT PaperContent  FROM [Papers].[dbo].[Papers] where FileID={}".format(el_id))
#         result = cursor.fetchone()[0]
#         with codecs.open(ddir+wos_id,'w',encoding='utf8') as out:
#             out.write(result)

id_dict = {}
with gzip.open('data/SD_WoS_id_match.txt.gz') as id_file:
    for line in tq(id_file,total=total_lines):
        k,v = line.strip().split()
        id_dict[int(k)] = v

cursor.execute("SELECT FileID, PaperContent FROM [Papers].[dbo].[Papers]")

for el_id,text in tq(cursor,total=total_el_recs):
    wos_id = id_dict.get(el_id,None)
    if wos_id is not None:
        with codecs.open(ddir+'matched/'+wos_id,'w',encoding='utf8') as out:
            out.write(text+'\n')
    else:
        with codecs.open(ddir+'unmatched/'+str(el_id),'w',encoding='utf8') as out:
            out.write(text+'\n')
cursor.close()
conn.close()
"""

# FIZING DUPLICATE MATCHES
import pymssql,os,gzip,codecs
from tqdm import tqdm as tq
ddir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/xml/'
total_lines=7775842
total_el_recs=12101870

### Server setup
server,user,password = [line.strip() for line in open('server_credentials.txt')]
conn = pymssql.connect(server, user, password, "tempdb")
cursor = conn.cursor()

id_dict2 = {}
with gzip.open('data/SD_WoS_id_match.txt.gz') as id_file:
    for line in tq(id_file,total=total_lines):
        k,v = line.strip().split()
        id_dict2[int(k)] = id_dict2.get(int(k),[])+[v]
all_dubz = []
all_dub_k = []
for k,v in tq(id_dict2.iteritems()):
    if len(v)>1:
        all_dubz += v
        all_dub_k.append(k)
for uid in tq(all_dubz):
    try:
        os.remove(ddir+'matched/{}'.format(uid))
    except:
        continue

for k in tq(all_dub_k):
    cursor.execute("SELECT FileID, PaperContent FROM [Papers].[dbo].[Papers] where FileID={}".format(k))
    el_id,text = cursor.fetchone()
    with codecs.open(ddir+'ambig/'+str(el_id),'w',encoding='utf8') as out:
        out.write(text+'\n')
cursor.close()
conn.close()













