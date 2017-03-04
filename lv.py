import numpy as np
from tqdm import tqdm as tq
import LargeVis

if __name__ == '__main__':

    infile,sample_size,workers = sys.argv[1:]
    if ',' in sample_size:
        sample_sizes = [int(s) for s in sample_size.split(',')]
    else:
        sample_sizes = [int(sample_size)]

    for s in sample_sizes:

        print "loading numpy file..."
        features = np.load(infile)

        if s > 0:
            print "generating random sample..."
            random_indices = np.random.choice(xrange(len(features)),s*1000000,replace=False)
            with open(infile+'.indices_{}M'.format(s),'w') as out:
                out.write(','.join(random_indices.astype(str)))
            features = features[random_indices]


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
        threads = workers
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


        LargeVis.save(infile+".{}M.lv_coords".format(s))

