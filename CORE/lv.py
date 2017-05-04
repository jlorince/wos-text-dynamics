import numpy as np
from tqdm import tqdm as tq
import LargeVis,sys,argparse
import multiprocessing as mp

help_string="""

THIS CODE MUST BE RUN IN A PYTHON 2.x ENVIRONMENT!!!
(largevis is not python 3 compatible)

Script generates a 2D embedding of a high dimensional data set stored a .npy file, where each row is a high dimensional feature vector. Depending on the size of the dataset, generate an embedding for all datapoints might not be possible (due to RAM constraints), and as such this script supports randomly sampling points for visualization.

Takes as input a numpy array, stored as FILENAME.npy, and generates two new files:

1 - the 2d coordinates of each point, saved as raw text file.
2 - The indices of the randomly sampled data points, saved as numpy array (only generated if downsampling is used).

All output files are saved to the same directory as the input file, and prepended with the name of that file.

Specific options described below.

See the largevis documentation for details on parameters and the algorithm: https://github.com/lferry007/LargeVis.
"""


if __name__ == '__main__':


    parser = argparse.ArgumentParser(help_string)
    parser.add_argument("--threads", help="Numer of parallel threads for largevis algorithm to use. Defaults to output of mp.cpu_count()",type=int,default=mp.cpu_count())
    parser.add_argument("--sampling", help="Method for randomly downsampling points. If `random` is used, `sample_size` total points are sampled. if `by_year`, randomly sample `sample_size` points *from each year* are sampled. Defaults to using all points in the input array.",type=str,default=None,choices=['random','by_year'])
    parser.add_argument("--sample_size", help="Number of points to randomly sample. See `sampling`. Ignored if `sampling` is not specified.",type=int,default=None)
    parser.add_argument('--input', help = 'input numpy file (assumed to be n_items x n_features numpy array)')
    parser.add_argument('--years', help = 'Path to numpy file with year for each document. Only used if `sampling`="by_year"', default='/backup/home/jared/storage/wos-text-dynamics-data/d2v-wos/index_years.npy',dtype=str)
    
    parser.add_argument('--prop', default = -1, type = int, help = 'number of propagations (see largevis documentation). Default 3.')
    parser.add_argument('--alpha', default = -1, type = float, help = 'learning rate (see largevis documentation). Default 1.0')
    parser.add_argument('--trees', default = -1, type = int, help = 'number of rp-trees (see largevis documentation). Default is set according to the data size.')
    parser.add_argument('--neg', default = -1, type = int, help = 'number of negative samples (see largevis documentation). Default 5.')
    parser.add_argument('--neigh', default = -1, type = int, help = 'number of neighbors in the NN-graph (see largevis documentation). Default 150.')
    parser.add_argument('--perp', default = -1, type = float, help = 'perplexity for the NN-graph (see largevis documentation). Default 50.')
    parser.add_argument('--temp', default = '/tmp/', type = str, help = 'Directory for writing temporary largevis-formatted data file.')
    args = parser.parse_args()


    print "loading numpy file..."
    features = np.load(args.infile)
    print features.shape

    if args.sampling == 'random':
        print "generating random sample..."
        random_indices = np.random.choice(xrange(len(features)),s,replace=False)
        if args.sample_size%1000000==0:
            filename = '.indices_{}M'.format(int(args.sample_size/1000000))
        else:
            filename = '.indices_{}'.format(args.sample_size)
        np.save(args.input+filename,random_indices)
        features = features[random_indices]

    elif args.sampling == 'by_year':

        index_years = np.load(args.years)
        random_indices = []
        for year in tq(years):
            random_indices.append(np.random.choice(np.where(index_years==year)[0],args.sample_size,replace=False))
        random_indices = np.concatenate(random_indices)
        filename = '.indices_year_{}'.format(args.sample_size)
        np.save(args.input+filename,random_indices)
        features = features[random_indices]


    # now we write the data to file in the required LargeVis format (which requires a header 
    # with the number of items and the dimensionality of the feature vectors)
    with open(args.temp+'lv_format.txt','w') as out:
        out.write("{}\t{}\n".format(*features.shape))
        for row in tq(features):
            out.write('\t'.join(row.astype(str))+'\n')
    del features

    # now run Large Vis! (in 2D mode)

    LargeVis.loadfile(args.temp+"lv_format.txt")

    Y = LargeVis.run(2, args.threads, args.samples, args.prop, args.alpha, args.trees, args.neg, args.neigh, args.gamma, args.perp)
    if arg.sampling == 'by_year':
        if args.sample_size is not None and args.sample_size%1000000==0:
            filename = '.{}M.year_lv_coords'.format(int(args.sample_size/1000000))
        else:
            filename = '.{}.year.lv_coords'.format(args.sample_size)
    else:
        if args.sample_size is not None and args.sample_size%1000000==0:
            filename = '.{}M.lv_coords'.format(int(args.sample_size/1000000))
        else:
            filename = '.{}.lv_coords'.format(args.sample_size)
    LargeVis.save(args.infile+filename)

