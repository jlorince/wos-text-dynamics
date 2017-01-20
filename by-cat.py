import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys,glob,os,codecs,logging,argparse
from tqdm import tqdm as tq
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
from pathos import multiprocessing as mp
from functools import partial


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
            rootLogger.info('{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad))
        else:
            rootLogger.info('{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad))
    


all_cats = ['Environmental Sciences', 'Chemistry, Analytical', 'Oncology','Psychology, Developmental','Hospitality, Leisure, Sport & Tourism','Chemistry, Multidisciplinary', 'Astronomy & Astrophysics','Biotechnology & Applied Microbiology', 'Chemistry, Organic','Pharmacology & Pharmacy', 'Virology','Cardiac & Cardiovascular Systems', 'Ophthalmology','Marine & Freshwater Biology', 'Chemistry, Physical','Biochemistry & Molecular Biology', 'Clinical Neurology','Dermatology', 'Surgery', 'Physics, Multidisciplinary','Medical Laboratory Technology', 'Multidisciplinary Sciences','Nuclear Science & Technology', 'Medicine, General & Internal','Materials Science, Multidisciplinary', 'Engineering, Chemical','Computer Science, Information Systems', 'Mathematics','Medicine, Research & Experimental', 'Biodiversity Conservation','Urology & Nephrology', 'Dentistry, Oral Surgery & Medicine','Fisheries', 'Geochemistry & Geophysics', 'Physics, Mathematical','Veterinary Sciences', 'Physics, Atomic, Molecular & Chemical','Engineering, Environmental', 'Polymer Science', 'Plant Sciences','Chemistry, Inorganic & Nuclear', 'Engineering, Biomedical','Meteorology & Atmospheric Sciences', 'Endocrinology & Metabolism','Psychology, Multidisciplinary', 'Mathematics, Applied','Physics, Condensed Matter', 'Immunology', 'Cell Biology','Microbiology', 'Electrochemistry', 'Neurosciences', 'Acoustics','Public, Environmental & Occupational Health','Computer Science, Theory & Methods', 'Otorhinolaryngology','Genetics & Heredity', 'Physiology', 'Paleontology', 'Optics','Zoology', 'Agronomy', 'Psychology, Clinical', 'Mineralogy','Rehabilitation', 'Engineering, Mechanical','Statistics & Probability', 'Critical Care Medicine','Nutrition & Dietetics', 'Pathology', 'Ecology', 'Allergy','Biochemical Research Methods', 'Rheumatology','Metallurgy & Metallurgical Engineering', 'Physics, Applied','Economics', 'Crystallography', 'Materials Science, Ceramics','Psychology, Experimental', 'Pediatrics','Agricultural Engineering','Radiology, Nuclear Medicine & Medical Imaging','Gastroenterology & Hepatology', 'Education & Educational Research','Engineering, Aerospace', 'Physics, Particles & Fields','Behavioral Sciences', 'Agriculture, Multidisciplinary','Engineering, Electrical & Electronic','Peripheral Vascular Disease', 'Food Science & Technology','Mechanics', 'Anesthesiology', 'Engineering, Multidisciplinary','Business, Finance', 'Hematology', 'Computer Science, Cybernetics','Business', 'Computer Science, Interdisciplinary Applications','Infectious Diseases', 'Geography', 'Andrology','Instruments & Instrumentation', 'Psychology, Biological','Gerontology', 'Thermodynamics','Education, Scientific Disciplines', 'Entomology', 'Family Studies','Agriculture, Dairy & Animal Science', 'Developmental Biology','Psychology, Educational', 'Oceanography', 'Parasitology','Spectroscopy', 'Biophysics', 'Anatomy & Morphology','Chemistry, Applied', 'Computer Science, Software Engineering','Obstetrics & Gynecology', 'Nursing', 'Substance Abuse','Psychology', 'Mathematical & Computational Biology','Audiology & Speech-Language Pathology', 'Psychology, Social','Medicine, Legal', 'Geosciences, Multidisciplinary', 'Orthopedics','Political Science', 'Psychology, Applied', 'Biology', 'Psychiatry','Chemistry, Medicinal', 'Emergency Medicine', 'Horticulture','Forestry', 'Education, Special', 'Management','Reproductive Biology', 'Physics, Fluids & Plasmas','Physics, Nuclear', 'Ornithology', 'History', "Women's Studies",'Art', 'Computer Science, Hardware & Architecture','Social Sciences, Interdisciplinary','Agricultural Economics & Poli cy', 'Mycology','Operations Research & Management Science', 'Respiratory System','Soil Science', 'Remote Sensing', 'Ethics', 'Geology', 'Microscopy','Law', 'Automation & Control Systems', 'Psychology, Mathematical','Transplantation', 'Sociology', 'Geriatrics & Gerontology','Literature', 'Toxicology', 'Materials Science, Paper & Wood','Materials Science, Coatings & Films', 'Engineering, Civil','Health Care Sciences & Services','Information Science & Library Science','Materials Science, Composites', 'Sport Sciences','Construction & Building Technology', 'Linguistics', 'Anthropology','Tropical Medicine', 'Computer Science, Artificial Intelligence','Criminology & Penology', 'Demography', 'Geography, Physical','Nanoscience & Nanotechnology', 'Primary Health Care','Social Work', 'Engineering, Geological','Engineering, Manufacturing', 'Environmental Studies', 'Limnology','Communication', 'Health Policy & Services','Language & Linguistics', 'Planning & Development', 'Social Issues','Public Administration','Mathematics, Interdisciplinary Applications','History & Philosophy Of Science', 'Psychology, Psychoanalysis','Water Resources', 'Engineering, Industrial','Energy & Fuels','Materials Science, Characterization & Testing','Engineering, Ocean', 'Materials Science, Textiles', 'Archaeology','Imaging Science & Photographic Technology','Industrial Relations & Labor', 'Classics', 'Philosophy','Social Sciences, Mathematical Methods', 'Robotics', 'Music','Area Studies', 'Telecommunications', 'Engineering, Marine','Literature, American', 'Literature, German, Dutch, Scandinavian','Evolutionary Biology', 'International Relations', 'Ergonomics','Folklore', 'Humanities, Multidisciplinary','Literature, British Isles', 'Religion', 'Film, Radio, Television','Transportation', 'Integrative & Complementary Medicine','Medieval & Renaissance Studies', 'History Of Social Sciences','Literature, Slavic', 'Literature, Romance', 'Ethnic Studies','Architecture', 'Cultural Studies', 'Literary Theory & Criticism','Urban Studies', 'Materials Science, Biomaterials', 'Poetry','Literary Reviews', 'Asian Studies','Literature, African, Australian, Canadian', 'Theater', 'Logic','Mining & Mineral Processing', 'Cell & Tissue Engineering','Transportation Science & Technology', 'Engineering, Petroleum','Dance', 'Social Sciences, Biomedical', 'Medical Informatics','Medical Ethics', 'Neuroimaging']

def entropy(arr,base=2):
    return scipy_entropy(arr,base=base)

# Given a pandas.Series of procesed abstracts, return the word frequency distribution 
# across all abstracts (limited to our chose vocabulary)
def termcounts(abs_ser):
    tc = Counter(' '.join(abs_ser).split())
    arr = np.array([tc.get(k,0) for k in vocab])
    return arr 

# calcualte Jensen Shannon Divergence of two probabability distributions
def jsd(p,q):
    return entropy((p+q)/2.,base=2) - 0.5*entropy(p,base=2) - 0.5*entropy(q,base=2)

def calc_measures(word_dists,window_size=1):
    ents = []
    jsds = []
    ent_difs = []
    apnd = []
    mx = len(word_dists)-(2*window_size-1)
    for i in range(mx):
        a = np.sum(word_dists[i:i+window_size],axis=0)
        asm = float(a.sum())
        if asm ==0:
            enta = np.nan       
        else:
            a = a/asm
            enta = entropy(a)
        b = np.sum(word_dists[i+window_size:i+window_size*2],axis=0)
        bsm = float(b.sum())
        if bsm == 0:
            entb = np.nan
        else:
            b = b/bsm
            entb = entropy(b)

        ents.append(enta)
        if i+window_size>=mx:
            apnd.append(entb)
        
        if asm==0 or bsm==0:
            ent_difs.append(np.nan)
            jsds.append(np.nan)
        else:
            ent_difs.append(entb-enta)
            jsds.append(jsd(a,b))
                
    return np.array(ents+apnd),np.array(ent_difs),np.array(jsds)
        

def shuffler(idx,all_tokens,token_counts,window=1):
    token_seq = all_tokens.copy()
    np.random.shuffle(token_seq)
    idx = 0
    shuffled_word_dists = np.zeros((25,len(vocab)))
    for i,toke in enumerate(token_counts):
        current = token_seq[idx:idx+toke]
        unique, counts = np.unique(current, return_counts=True)
        word_dist = np.zeros(len(vocab))
        word_dist[unique] = counts
        shuffled_word_dists[i] = word_dist
        idx+=toke
    return calc_measures(shuffled_word_dists,window)
    
def parse_cat(cat_name,window=1):
    df = pd.read_pickle('{}by-cat/{}.pkl'.format(datadir,cat_name))
    if len(df==0):
        return 0
    # generate word distributions 
    word_dists = np.zeros((25,len(vocab)))
    for year,grp in df.groupby('year'):
        word_dists[year-1991] = termcounts(grp.abstract)

    # total token count by year
    token_counts = word_dists.sum(1,dtype=int)
    # generate giant array of every token in data (for shuffling by null model)
    combined_word_dist = word_dists.sum(0,dtype=int)
    all_tokens = []
    for term,cnt in enumerate(combined_word_dist):#,total=len(combined_word_dist):
        all_tokens += [term]*cnt
    all_tokens = np.array(all_tokens)

    # calculate raw measures
    ents,ent_difs,jsds = calc_measures(word_dists,window_size=window)
    
    #calculate null measures
    # try:
    #     pool.close()
    # except:
    #     pass
    # pool = mp.Pool(procs)
    #result = [r for r in tq(pool.imap_unordered(lambda x: shuffler(x,all_tokens,token_counts),range(bootstraps),chunksize=bootstraps/procs),total=bootstraps)]
    result = [shuffler(x,all_tokens,token_counts,window_size=window) for x in range(bootstraps)]
    
    dist_path = '{}results/termdist_{}.npy'.format(datadir,cat_name)
    if not os.path.exists(dist_path):
        np.save(dist_path,word_dists)   
    
    with open('{}results/results_{}_{}'.format(datadir,window,cat_name),'w') as out:
        
        for measure in ('ents','ent_difs','jsds'):
            out.write("{}\t{}\n".format(measure,','.join(vars()[measure].astype(str))))
        for i,measure in enumerate(['entropy-null','entdif-null','jsd-null']):
            samples = np.array([r[i] for r in result])
            m = samples.mean(0)
            ci = 1.96 * samples.std(0) / np.sqrt(bootstraps)
            out.write('{}_m\t{}\n'.format(measure,','.join(m.astype(str))))
            out.write('{}_c\t{}\n'.format(measure,','.join(ci.astype(str))))
    return 1

def go(window_size,files):
    complete = 0
    for fi,result  in tq(zip(all_cats,pool.imap_unordered(lambda f: parse_cat(f,window=window),files)),total=len(files)):
        cat = fi[fi.rfind('/')+1:-4]
        if result == 0:
            rootLogger.info('No data for category "{}"'.format(cat))
        if result == 1:
            rootLogger.info('Category "{}" processed successfully for window size={}'.format(cat,window_size))
        complete+=result
    rootLogger.info('{} total categories processed for window size={}'.format(complete,window))
        
# for cat in tq(all_cats):
#     with timed("Processing category - {}".format(cat)):
#         parse_cat(cat,window=window)    

if __name__=='__main__':

    parser = argparse.ArgumentParser("Script for calculating information theoretic measures of text evolution among WoS abstracts")
    parser.add_argument("-p", "--procs",help="specify number of processes for parallel computations (defaults to output of mp.cpu_count())",default=mp.cpu_count(),type=int)
    parser.add_argument("-w", "--window", help="window size, enter a single value, range (x_y), or list (x,y,z)",type=str,default='1')
    parser.add_argument("-l", "--logfile", help="prefix for logfile",type=str,default='')
    parser.add_argument("-o", "--output", help="output path for results",type=str,default='/backup/home/jared/storage/wos-text-dynamics-data/results/')
    parser.add_argument("-b", "--null_bootstrap_samples", help="Number of monte carlo samples for bootstrap null model calculations",type=int,default=100)
    parser.add_argument("-d", "--datadir",help="root input data directory",default='/backup/home/jared/storage/wos-text-dynamics-data/by-cat',type=str)
    #parse.add_argument("-c", "--cats", help="path to pickled field-level dataframes", default='/backup/home/jared/storage/wos-text-dynamics-data/by-cat',type=str)
    parser.add_argument("-v", "--vocab_thresh",help="vocabulary trimming threshold",default=100,type=int)
    args = parser.parse_args()

    ### LOGGING SETUP
    now = datetime.datetime.now()
    if args.logfile:
        args.logfiles += '_'
    log_filename = now.strftime('{}%Y%m%d_%H%M%S.log'.format(args.logfile))
    #log_filename = now.strftime('{}%Y%m%d_%H%M%S.log'.format(args.logfile+'_{}_{}'.format(args.window,args.side)))
    logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    ### Vocabulary setup
    vocab_path = args.datadir+'vocab_pruned_'+str(args.vocab_thresh)
    if os.path.exists(vocab_path):
        vocab = [line.strip() for line in codecs.open(vocab_path,encoding='utf8')]
    else:
        vocab_dict ={}
        for fpath in tq(glob.glob(args.datadir+'*.pkl')):
            df = pd.read_pickle(fpath)
            for abstract in tq(df.abstract):
                for term in abstract.split():
                    vocab_dict[term] = vocab_dict.get(term,0)+1
        raw_term_counts = pd.Series(vocab_dict)  

        stemmer = EnglishStemmer()
        stop = set(stopwords.words('english'))
        stop = stop.union([stemmer.stem(s) for s in stop])
        pruned = raw_term_counts[raw_term_counts>=VOCAB_THRESH]
        vocab = sorted([term for term in pruned.index if term not in stop and type(term)==unicode and term.isalpha()])
        rootLogger.info("Total vocab size= {}".format(len(vocab)))

    pool = mp.Pool(procs)
    if '_' in args.window:
        start,end = map(int,args.window.split('_'))
        window_range = range(start,end+1) 
    if ',' in args.window:
        window_range = map(int,args.window.split(','))
    else:
        window_range = [int(args.window)]

    files = glob.glob(args.datadir+'*.pkl')
    for w in window_range:
        go(w,files)
