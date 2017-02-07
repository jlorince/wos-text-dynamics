import gzip,codecs,time,datetime,glob,logging,string,os,pymssql
from import tqdm as tq
from lxml import etree
from nltk.tokenize import word_tokenize
import multiprocessing as mp

server,user,password = [line.strip() for line in open('server_credentials.txt')]
ddir = "P:/Projects/WoS/wos-text-dynamics-data/raw/"

id_dict = {}
for line in gzip.open('data/SD_WoS_id_match_DEDUPED.txt.gz'):
    el,wos = line.strip().split()
    id_dict[int(el)] = wos


def process(input_tuple):
    idx,min_id,max_id = input_tuple

    # read data
    conn = pymssql.connect(server, user, password, "tempdb")
    cursor = conn.cursor()
    cursor.execute("SELECT FileID, PaperContent FROM [Papers].[dbo].[Papers] where FileID>{} and FileID<={}".format(min_id,max_id))

    with gzip.open("{}metadata_{}".format(ddir,idx),'w') as meta_out,gzip.open("{}text_{}".format(ddir,idx),'w') as text_out:
        for FileID,PaperContent in cursor:
            rawtext,found_abstract,found_formatted_text,found_rawtext = parse_xml(PaperContent)
            wos_id = id_dict[FileID]
            meta_out.write("{}\t{}\t{}\t{}\t{}\n".format(FileID,wos_id,found_abstract,found_formatted_text,found_rawtext))
            text_out.write("{}\t{}\n".format(wos_id,' '.join(rawtext)))
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

    procs = np.cpu_count()

    chunks = np.percentile(id_dict.keys(),np.linspace(0,100,procs))
    chunks[0]=-1

    pool.map(process,zip(range(procs),chunks,chunks[1:]))
