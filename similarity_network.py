from annoy import AnnoyIndex
import numpy as np
import multiprocess as mp
import pickle,os,gzip,glob
from tqdm import tqdm as tq
from functools import partial

procs = 24

def convert_distance(d):
    return (d**2) /2

#def get_neighbors(i,nns=1000):
#    return [(i,n,convert_distance(d)) for n,d in zip(*t.get_nns_by_item(i,n=nns+1,search_k=10000,include_distances=True)) if i!=n]

def get_neighbors(arr,nns=1000):
    pid = os.getpid()
    #done = set()
    total = len(arr)
    with gzip.open('/backup/home/jared/storage/similarity_network/simnet_{}_pid{}'.format(year,pid),'wb') as out:
        for idx,doc in enumerate(arr):
            if idx%1000==0:
                print("{} --> {}/{} ({:.2f}%)".format(pid,idx,total,100*(idx/total)))
            for n,d in zip(*t.get_nns_by_item(doc,n=nns+1,search_k=10000,include_distances=True)):
                if doc==n:
                    continue
                if doc>n:
                    tup = (n,doc)
                else:
                    tup = (doc,n)
                #if tup in done:
                #    continue
                #else:
                out.write("{} {} {}\n".format(tup[0],tup[1],convert_distance(d)).encode('utf8'))
                    #done.add(tup)


def gen_truncated_network(year,n=100,d='/backup/home/jared/storage/similarity_network/'):
    files = glob.glob("{}simnet_{}*".format(d,year))
    
    # pool  = mp.Pool(procs)
    # def process(fi):
    #     idx=0
    #     for line in gzip.open(fi):
    #         if idx<n:
    #             line = line.decode('utf8')
    #             #out.write(line)
    #             yield line
    #         if idx==999:
    #             idx=-1
    #         idx += 1
    
    with open('network_{}_{}'.format(year,n),'w') as out:
        for f in tq(files):
            idx=0
            for line in tq(gzip.open(f)):
                if idx<n:
                    line = line.decode('utf8')
                    line = line.strip().split()
                    d = line[2]
                    line[2] = str(1-float(d))
                    out.write(' '.join(line)+'\n')
                if idx==999:
                    idx=-1
                idx += 1


# def get_neighbors(year,nns=1000):
#     done = set()
#     total = len(dict_year[year])
#     idx=0
#     with open('similarity_network_100-5-5_{}'.format(year),'w') as out:
#         for idx,doc in enumerate(dict_year[year]):
#             if idx%10000==0:
#                 print("{} --> {}/{} ({:.2f}%)".format(year,idx,total,100*(idx/total)))
#             for n,d in zip(*t.get_nns_by_item(doc,n=nns+1,search_k=10000,include_distances=True)):
#                 if doc==n:
#                     continue
#                 if doc>n:
#                     doc,n = n,doc
#                 if (doc,n) in done:
#                     continue
#                 else:
#                     out.write("{} {} {}\n".format(doc,n,d))
#                     done.add((doc,n))
    


if __name__ == '__main__':



    dict_year = pickle.load(open('dict_year.pkl','rb'))
    # year_dict = pickle.load(open('year_dict.pkl','rb'))
    # dict_year = {year:[] for year in range(1991,2016)}
    # for k,v in tq(year_dict.items(),total=len(year_dict)):
    #     dict_year[v].append(k)
    # del year_dict
    # for y in tq(range(1991,2016)):
    #     dict_year[y] = np.array(dict_year[y])


    for year in tq(range(1991,2016)):

        if not os.path.exists('annoy_index_{}.ann'.format(year)):
            dict_year = pickle.load(open('dict_year.pkl','rb'))
            features = np.load('model_100-5-5.npy.docvecs.doctag_syn0.npy')
            total_docs,f = features.shape
            t = AnnoyIndex(f,metric='angular')  # Length of item vector that will be indexed
            #add = sum([len(dict_year[year]) for year in range(1991,year)])
            for i,vec in tq(enumerate(features[dict_year[year]]),total=len(dict_year[year])):
                t.add_item(i, vec)

            del features
            %time t.build(50) # 10 trees
            %time t.save('annoy_index_{}.ann'.format(year))
            del t
            del dict_year
        t = AnnoyIndex(100,metric='angular')
        t.load('annoy_index_{}.ann'.format(year))
            

        pool = mp.Pool(procs)
        #pool.map(get_neighbors,range(1991,2016))

        #with open('similarity_network_100-5-5_{}'.format(year),'w') as out:
        #    done = set()
        arrs = np.array_split(range(len(pickle.load(open('dict_year.pkl','rb'))[year])),procs)
        pool.map(get_neighbors,arrs)


            # for result in tq(pool.imap_unordered(get_neighbors,dict_year[year],chunksize=1000),total=len(dict_year[year])):
            #     for i,n,d in result:
            #         if i>n:
            #             i,n = n,i
            #         if (i,n) in done:
            #             continue
            #         else:
            #             out.write("{} {} {}\n".format(i,n,d))
            #             done.add((i,n))
        pool.terminate()




            

