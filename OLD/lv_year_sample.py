import numpy as np
from tqdm import tqdm as tq
import LargeVis,sys,pickle

n = 200000

if __name__ == '__main__':

    infile,workers = sys.argv[1:]

    dict_year = pickle.load(open('/backup/home/jared/eval_wos/dict_year.pkl.py2','rb'))

    sample = np.concatenate([np.random.choice(dict_year[y],n,replace=False) for y in range(1991,2016)])
    with open(infile+'.indices_year_sample','w') as out:
        out.write(','.join(sample.astype(str)))


    print "loading numpy file..."
    features = np.load(infile)[sample]
    print features.shape


    # now we write the data to file in the required LargeVis format (which requires a header with the number of items and the dimensionality of the feature vectors)
    #with open('/backup/home/jared/lv_format.txt','w') as out:
    with open('/tmp/lv_format.txt','w') as out:
        out.write("{}\t{}\n".format(*features.shape))
        for row in tq(features):
            out.write('\t'.join(row.astype(str))+'\n')
    del features

    # now run Large Vis!

    # LargeVis doesn't take named arguments, so we have to pass all these explicitly, using -1 to indicate defaults 
    # (this is handled more smoothly by argparse when calling from command line)

    outdim = 2
    threads = int(workers)
    samples = -1
    prop = -1
    alpha = -1
    trees = -1
    neg = -1
    neigh = -1
    gamma = -1
    perp = -1

    LargeVis.loadfile("/tmp/lv_format.txt")

    Y = LargeVis.run(outdim, threads, samples, prop, alpha, trees, neg, neigh, gamma, perp)


    LargeVis.save(infile+".year_sample.lv_coords")

