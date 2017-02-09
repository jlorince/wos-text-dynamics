import gzip,codecs,time,datetime,glob,logging,string,os,pymssql,sys
from tqdm import tqdm as tq
from lxml import etree
from nltk.tokenize import word_tokenize
import multiprocessing as mp
import numpy as np
import pandas as pd

server,user,password = [line.strip() for line in open('server_credentials.txt')]
ddir = "P:/Projects/WoS/wos-text-dynamics-data/elsevier/raw/"

id_dict = {}
for line in gzip.open('data/SD_WoS_id_match_DEDUPED.txt.gz'):
    el,wos = line.strip().split()
    id_dict[int(el)] = wos.decode('utf8')

import time,datetime
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


def process(input_tuple):
    idx,min_id,max_id,total = input_tuple

    # read data
    conn = pymssql.connect(server, user, password, "tempdb")
    cursor = conn.cursor()
    cursor.execute("SELECT FileID, PaperContent FROM [Papers].[dbo].[Papers] where FileID>{} and FileID<={}".format(min_id,max_id))

    with timed('Processing idx {}'.format(idx)):
        with gzip.open("{}metadata_{}".format(ddir,idx),'w') as meta_out,\
             gzip.open("{}matched/text_{}".format(ddir,idx),'w') as matched_out,\
             gzip.open("{}unmatched/text_{}".format(ddir,idx),'w') as unmatched_out,\
             open("{}log_{}".format(ddir,idx),'w') as log:
            for i,(FileID,PaperContent) in enumerate(cursor,76764):
                try:
                    rawtext,found_abstract,found_formatted_text,found_rawtext = parse_xml(PaperContent)
                except etree.XMLSyntaxError:
                    log.write("ERROR FOR FILEID {}\n".format(FileID))
                    continue
                wos_id = id_dict.get(FileID,'')
                if wos_id is not '':
                    matched_out.write("{}\t{}\n".format(wos_id,' '.join(rawtext)).encode('utf8'))
                else:
                    unmatched_out.write("{}\t{}\n".format(FileID,' '.join(rawtext)).encode('utf8'))
                meta_out.write("{}\t{}\t{}\t{}\t{}\n".format(FileID,wos_id,int(found_abstract),int(found_formatted_text),int(found_rawtext)).encode('utf8'))
                
                if i%1000==0:
                    s = "Idx {}: {}/{} ({:2f}%) records processed".format(idx,i,total,100*(i/total))
                    print(s)
                    log.write(s+'\n')
                    log.flush()
                print("{}/{} ({:2f}%) ".format(i,total,100*(i/total))) # DEBUG
            s = "Idx {} DONE: {}/{} ({:2f}%) records processed\n".format(idx,i,total,100*(i/total))
            print(s)
            log.write(s+'\n')
            log.flush()
        cursor.close()
        conn.close()


def parse_xml(text):
    tree = etree.fromstring(text)

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
    rawtext = word_tokenize(' '.join(all_text))
    return rawtext,found_abstract,found_formatted_text,found_rawtext


if __name__=='__main__':

    if len(sys.argv)>1:
        procs = int(sys.argv[1])
    else:
        procs = mp.cpu_count()
    pool = mp.Pool(procs)

    with timed('Building chunks'):
        conn = pymssql.connect(server, user, password, "tempdb")
        cursor = conn.cursor()
        cursor.execute("SELECT FileID from [Papers].[dbo].[Papers]")
        all_ids = sorted([fid[0] for fid in cursor.fetchall()])
        cursor.close()
        conn.close()

        chunks = np.percentile(all_ids,np.linspace(0,100,procs)).astype(int)
        chunks[0]=-1
        #np.array_split(np.array(sorted([k for k in id_dict.keys()])),60)
        per_chunk_counts = pd.Series(np.digitize(all_ids,chunks,right=True)-1).value_counts().sort_index().values

    with timed('Running main processing'):
        pool.map(process,zip(range(procs),chunks,chunks[1:],per_chunk_counts))
