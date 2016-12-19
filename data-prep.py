import numpy as np
import gzip,time,datetime,string,signal,sys,cPickle,codecs
import pandas as pd
import multiprocessing as mp
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
stemmer = EnglishStemmer()

kwdir = 'S:/UsersData_NoExpiration/jjl2228/keywords/parsed/'
tmpdir = 'S:/UsersData_NoExpiration/jjl2228/keywords/temp/'

debug = None
#temp_data_generated = False

all_cats = ['Environmental Sciences', 'Chemistry, Analytical', 'Oncology','Psychology, Developmental','Hospitality, Leisure, Sport & Tourism','Chemistry, Multidisciplinary', 'Astronomy & Astrophysics','Biotechnology & Applied Microbiology', 'Chemistry, Organic','Pharmacology & Pharmacy', 'Virology','Cardiac & Cardiovascular Systems', 'Ophthalmology','Marine & Freshwater Biology', 'Chemistry, Physical','Biochemistry & Molecular Biology', 'Clinical Neurology','Dermatology', 'Surgery', 'Physics, Multidisciplinary','Medical Laboratory Technology', 'Multidisciplinary Sciences','Nuclear Science & Technology', 'Medicine, General & Internal','Materials Science, Multidisciplinary', 'Engineering, Chemical','Computer Science, Information Systems', 'Mathematics','Medicine, Research & Experimental', 'Biodiversity Conservation','Urology & Nephrology', 'Dentistry, Oral Surgery & Medicine','Fisheries', 'Geochemistry & Geophysics', 'Physics, Mathematical','Veterinary Sciences', 'Physics, Atomic, Molecular & Chemical','Engineering, Environmental', 'Polymer Science', 'Plant Sciences','Chemistry, Inorganic & Nuclear', 'Engineering, Biomedical','Meteorology & Atmospheric Sciences', 'Endocrinology & Metabolism','Psychology, Multidisciplinary', 'Mathematics, Applied','Physics, Condensed Matter', 'Immunology', 'Cell Biology','Microbiology', 'Electrochemistry', 'Neurosciences', 'Acoustics','Public, Environmental & Occupational Health','Computer Science, Theory & Methods', 'Otorhinolaryngology','Genetics & Heredity', 'Physiology', 'Paleontology', 'Optics','Zoology', 'Agronomy', 'Psychology, Clinical', 'Mineralogy','Rehabilitation', 'Engineering, Mechanical','Statistics & Probability', 'Critical Care Medicine','Nutrition & Dietetics', 'Pathology', 'Ecology', 'Allergy','Biochemical Research Methods', 'Rheumatology','Metallurgy & Metallurgical Engineering', 'Physics, Applied','Economics', 'Crystallography', 'Materials Science, Ceramics','Psychology, Experimental', 'Pediatrics','Agricultural Engineering','Radiology, Nuclear Medicine & Medical Imaging','Gastroenterology & Hepatology', 'Education & Educational Research','Engineering, Aerospace', 'Physics, Particles & Fields','Behavioral Sciences', 'Agriculture, Multidisciplinary','Engineering, Electrical & Electronic','Peripheral Vascular Disease', 'Food Science & Technology','Mechanics', 'Anesthesiology', 'Engineering, Multidisciplinary','Business, Finance', 'Hematology', 'Computer Science, Cybernetics','Business', 'Computer Science, Interdisciplinary Applications','Infectious Diseases', 'Geography', 'Andrology','Instruments & Instrumentation', 'Psychology, Biological','Gerontology', 'Thermodynamics','Education, Scientific Disciplines', 'Entomology', 'Family Studies','Agriculture, Dairy & Animal Science', 'Developmental Biology','Psychology, Educational', 'Oceanography', 'Parasitology','Spectroscopy', 'Biophysics', 'Anatomy & Morphology','Chemistry, Applied', 'Computer Science, Software Engineering','Obstetrics & Gynecology', 'Nursing', 'Substance Abuse','Psychology', 'Mathematical & Computational Biology','Audiology & Speech-Language Pathology', 'Psychology, Social','Medicine, Legal', 'Geosciences, Multidisciplinary', 'Orthopedics','Political Science', 'Psychology, Applied', 'Biology', 'Psychiatry','Chemistry, Medicinal', 'Emergency Medicine', 'Horticulture','Forestry', 'Education, Special', 'Management','Reproductive Biology', 'Physics, Fluids & Plasmas','Physics, Nuclear', 'Ornithology', 'History', "Women's Studies",'Art', 'Computer Science, Hardware & Architecture','Social Sciences, Interdisciplinary','Agricultural Economics & Policy', 'Mycology','Operations Research & Management Science', 'Respiratory System','Soil Science', 'Remote Sensing', 'Ethics', 'Geology', 'Microscopy','Law', 'Automation & Control Systems', 'Psychology, Mathematical','Transplantation', 'Sociology', 'Geriatrics & Gerontology','Literature', 'Toxicology', 'Materials Science, Paper & Wood','Materials Science, Coatings & Films', 'Engineering, Civil','Health Care Sciences & Services','Information Science & Library Science','Materials Science, Composites', 'Sport Sciences','Construction & Building Technology', 'Linguistics', 'Anthropology','Tropical Medicine', 'Computer Science, Artificial Intelligence','Criminology & Penology', 'Demography', 'Geography, Physical','Nanoscience & Nanotechnology', 'Primary Health Care','Social Work', 'Engineering, Geological','Engineering, Manufacturing', 'Environmental Studies', 'Limnology','Communication', 'Health Policy & Services','Language & Linguistics', 'Planning & Development', 'Social Issues','Public Administration','Mathematics, Interdisciplinary Applications','History & Philosophy Of Science', 'Psychology, Psychoanalysis','Water Resources', 'Engineering, Industrial','Energy & Fuels','Materials Science, Characterization & Testing','Engineering, Ocean', 'Materials Science, Textiles', 'Archaeology','Imaging Science & Photographic Technology','Industrial Relations & Labor', 'Classics', 'Philosophy','Social Sciences, Mathematical Methods', 'Robotics', 'Music','Area Studies', 'Telecommunications', 'Engineering, Marine','Literature, American', 'Literature, German, Dutch, Scandinavian','Evolutionary Biology', 'International Relations', 'Ergonomics','Folklore', 'Humanities, Multidisciplinary','Literature, British Isles', 'Religion', 'Film, Radio, Television','Transportation', 'Integrative & Complementary Medicine','Medieval & Renaissance Studies', 'History Of Social Sciences','Literature, Slavic', 'Literature, Romance', 'Ethnic Studies','Architecture', 'Cultural Studies', 'Literary Theory & Criticism','Urban Studies', 'Materials Science, Biomaterials', 'Poetry','Literary Reviews', 'Asian Studies','Literature, African, Australian, Canadian', 'Theater', 'Logic','Mining & Mineral Processing', 'Cell & Tissue Engineering','Transportation Science & Technology', 'Engineering, Petroleum','Dance', 'Social Sciences, Biomedical', 'Medical Informatics','Medical Ethics', 'Neuroimaging']

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

def gen_series(s):
    ser = pd.Series(index=['cat{}'.format(i) for i in xrange(8)])
    if pd.isnull(s):
        return ser
    else:
        li = s.split('|')
    ser[:len(li)] = li
    return ser

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
        with timed('abstract loading',year=year):
            abs_current = pd.read_table('P:/Projects/WoS/WoS/parsed/abstracts/{}.txt.gz'.format(year),header=None,names=['uid','abstract'], nrows=debug).dropna()
        with timed('abstract parsing',year=year):
            abs_current['abstract'] = parse_abs(abs_current['abstract'].values)
        with timed('keyword loading',year=year):
            #kw_current = pd.read_table('S:/UsersData_NoExpiration/jjl2228/keywords/pubs_by_year/{}.txt.gz'.format(year),header=None,names=['keyword','uid'],nrows=debug)
            kw_current = pd.read_table(':/Projects/WoS/WoS/parsed/keywords/{}.txt.gz'.format(year),header=None,names=['uid','nk','keywords'],usecols=['uid','keywords'],nrows=debug)
        with timed('metadata loading',year=year):
            md_current = pd.read_table('P:/Projects/WoS/WoS/parsed/metadata/{}.txt.gz'.format(year),header=None, nrows=debug,
                                   names=["uid","date","pubtype","volume","issue","pages","paper_title","source_title","doctype"],
                                  usecols=["uid","pubtype","paper_title","source_title","doctype"])
        with timed('category loading',year=year):
            cats_current = pd.read_table('P:/Projects/WoS/WoS/parsed/subjects/{}.txt.gz'.format(year),header=None,names=['uid','heading','subheading','categories'], nrows=debug)
        with timed('category formatting',year=year):
            #cats_current = pd.concat([cats_current[['uid','heading','subheading']],cats_current['categories'].apply(gen_series)],axis=1)
            cats_current['categories'] = cats_current['categories'].split('|')
        with timed('reference loading',year=year):
            refs_current = pd.read_table('P:/Projects/WoS/WoS/parsed/references/{}.txt.gz'.format(year),header=None,names=['uid','n_refs','refs','missing'],usecols=['uid','refs'], nrows=debug)
        with timed('data merging',year=year):
            current = abs_current.merge(md_current,on='uid',how='inner').merge(cats_current,on='uid',how='inner').merge(refs_current,on='uid',how='left').merge(kw_current,on='uid',how='left')
            current['year'] = year
        with timed('saving data'):
            current.to_pickle('{}{}.pkl'.format(tmpdir,year))
        print 'final datasize: {} ({})'.format(current.shape,year)
    return current
        
if __name__=='__main__':

    procs = 25

    temp_data_generated = int(sys.argv[1])

    if not temp_data_generated:
        with timed('keyword processing',pad=' ######## '):
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

    else:
        with timed('keyword processing',pad=' ######## '):        
            with timed('dataframe concatenation'):
                result = []
                for year in xrange(1991,2016):
                    result.append(pd.read_pickle('{}{}.pkl'.format(tmpdir,year)))
                    print year,

    with timed('dataframe concatenation'):
        df = pd.concat(result)

    with timed('dataframe partitioning'):
        for cat in all_cats:
            with timed(cat):
                cat_df = df[df.categories.apply(lambda x: cat in x)]
                cat_df.to_pickle('S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/by-cat/{}'.format(cat))



