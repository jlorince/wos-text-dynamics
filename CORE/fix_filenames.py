"""

This is just a simple little script to fix filenames for old d2v model runs that don't meet the latest naming conventions.

This can be modified pretty simply if we change naming conventions in the future.

CURRENT CORRECT FORMAT:

"{}-{}-{}-{}-{}".format(size,window,min_count,sample,seed)

ASSUMES SEED IS ALWAYS THE LAST PARAMETER


"""
import glob,os
from gensim.models import Doc2Vec

d2v_dir = 'P:/Projects/WoS/wos-text-dynamics-data/d2v-wos/'
os.chdir(d2v_dir)

folders = [f for f in os.listdir('.') if os.path.isdir(f)]

for f in folders:
    print(f)
    if len(f.split('-')) == 5:
        continue

    files = os.listdir(f)
    try:
        model_file = sorted([fi for fi in files if fi.startswith('model')],key=lambda x: len(x))[0]
        model = Doc2Vec.load('{}/{}'.format(f,model_file))
    except:
        continue

    old_params = model_file.split('_')[1]

    size = model.vector_size
    window = model.window
    min_count = model.min_count
    sample = '{:f}'.format(model.sample).rstrip('0').rstrip('.') # ensure consistent handling of decimals
    seed = None

    new_params = "{}-{}-{}-{}-{}".format(size,window,min_count,sample,seed)

    for fi in files:
        os.rename(f+'/'+fi,f+'/'+fi.replace(old_params,new_params))
    os.rename(f,new_params)



