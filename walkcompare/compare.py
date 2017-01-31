import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib_venn import venn2, venn3
import mca

import matplotlib.pyplot as plt
from collections import defaultdict
import json
from adjustText import adjust_text as AT


class compare:
    
    """ 
        
    Compare -- plot collections for comparison purposes.
    
    Description:
        Compare is a set of tools intended for comparing collections of Web Archives based 
        on derivative sets created by Warcbase (https://github.com/lintool/warcbase).
    
    Args:
        @collectionset (list):  A list of lists or a dict() of size 2 or greater for comparison 
        purposes.
        @names (list):  An optional list of names for the collections.  Must be equal in size to 
        collections. If collections is a dict, this parameter will be overwritten.
        @exclude (list):  A list of collection names or index keys to exclude from the analysis.
        @REMOVE_SINGLES (Bool): (Default:True) For 4 collections or more, remove from the analysis 
        any data points that are
            members of only one collection. This reduces the chance that a disproportionately 
            large collection
            will be seen as an outlier merely because it is disproportionately large.
            
    Example:
        $ data = [["happy", "excited", "content", "nostalgic"],
                    ["sad", "unhappy", "nostalgic", "melancholic"],
                    ["reminiscing", "remembering", "nostalgic"], 
                    ["happy", "get lucky", "money maker"], 
                    ["get lucky", "money maker"],
                    ["excited", "love actually", "get lucky"]]
        $ names = ["happy words", "sad words", "memory words", "pharrel williams songs", "dancehall slang", "rom coms"]
        $ comp = compare.Compare(data, names)
        $ comp.plot_ca()
        
            
    """
    
    def __init__ (self, collectionset, names=[], exclude=[], REMOVE_SINGLES=True):
        self.collection_names = names
        self.exclude = exclude
        self.collectionset = collectionset
        self.REMOVE_SINGLES = REMOVE_SINGLES
        self.DIMS = 2
        self.LABEL_BOTH_FACTORS = False
        self.adjust = False
        self.dimensions = None
        self.counts = None
        self.result = {}
        self.clabels = []
        self.rlabels = []
        self.plabels = []
        if isinstance(self.collectionset, dict):
            self.collection_names = [x.strip() for x in self.collectionset.keys()]
            self.collectionset = [x.strip() for x in self.collectionset.values()]

        if type([y[0] for y in self.collectionset][0]) is tuple: #will need to include checks for size of sample
            print ("yay mca")
            self.collection_names = list(set([x[0] for y in self.collectionset for x in y]))
            if self.index:
                self.collectionset = self.sublist(self.collectionset, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
            self.mca(self.collectionset, self.collection_names)
        else:            
            #self.collectionset = dict([(x[0], x[1]) for y in self.collectionset for x in y])
            if not self.collection_names:
                self.collection_names = range(1, len(self.collectionset)+1)
            # if index var is provided, use index to filter collection list
            if self.exclude:
                self.collectionset = self.sublist(self.collectionset, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
        #two sample venn
            if len(self.collectionset) == 2:
                self.response = self.two_venn(self.collectionset)
        #three sample venn
            elif len(self.collectionset) == 3:
                self.response = self.three_venn(self.collectionset)
        #use mca for greater than three
            elif len(self.collectionset) >3:
                self.ca = self.ca(self.collectionset, self.collection_names)
            else:
                self.no_compare()
                    
    def excluded (self):
        if all(isinstance(item, int) for item in self.exclude):
            self.collectionset = self.sublist(self.collectionset, self.exclude)
            self.collection_names = self.sublist(self.collection_names, self.exclude)
        else:
            self.collection, self.collection_names = PC.process_collection(self.collection, self.collection_names, self.exclude)

    def handle_json (self, input):
        pass
    
    def examine_input (self, input):
        if isinstance(self.collectionset, dict): #passed a plain dictionary
            self.collection_names = [x for x in self.collection.keys()]
            self.collectionset=[x for x in self.collection.values()]
                    
    def recur_len(self, L):
        return sum(L + recur_len(item) if isinstance(item, list) else L for item in L)

    def no_compare(self):
        return ("Need at least two collectionset to compare results.")

    #get a sublist from a list of indices
    def sublist (self, list1, list2):
        return([list1[x] for x in list2]) 
    

		
    def two_venn (self, collectionset):
        """ Return a two-way venn diagram of two sets """
        self.V2_AB = set(collectionset[0]).intersection(set(collectionset[1]))
        return  (venn2([set(x) for x in collectionset], set_labels=self.collection_names))

    def three_venn (self, collectionset):
        """ Return a three-way venn diagram of three sets """
        self.V3_ABC = set(collectionset[0]) & set(collectionset[1]) & set(collectionset[2]) 
        self.V3_AB = set(collectionset[0]) & set(collectionset[1]) - self.V3_ABC
        self.V3_BC = set(collectionset[1]) & set(collectionset[2]) - self.V3_ABC
        self.V3_AC = set(collectionset[0]) & set(collectionset[2]) - self.V3_ABC
        self.V3_A = set(collectionset[0]) - (self.V3_ABC | self.V3_AB | self.V3_AC )
        self.V3_B = set(collectionset[1]) - (self.V3_ABC | self.V3_AB | self.V3_BC )
        self.V3_C = set(collectionset[2]) - (self.V3_ABC | self.V3_BC | self.V3_AC )
        return  (venn3([set(x) for x in collectionset], set_labels=self.collection_names))


    #get set of all items (unduplicated)
    def unionize (self, sets_list):
        """ Take a list of sets and return a set with all duplicates removed """
        return (set().union(*sets_list))
    
    def create_matrix (self, dd, collectionset):
        d = []
        for y in collectionset:
            d.append({x:x in y for x in dd})
        return (pd.DataFrame(d, index=self.collection_names))

    def remove_singles (self, df):
        return (df.loc[:, df.sum(0) >1].fillna(False))

    def fill_vars (self, df):
        self.response = df
        self.counts = mca.mca(df)
        if len(self.counts.L >1):
            self.dimensions = self.counts.L
        else:
            self.dimensions = np.append(self.counts.L, 0.0)
        self.result["rows"] = self.counts.fs_r(N=self.DIMS)
        self.result["columns"] = self.counts.fs_c(N=self.DIMS)
        self.rlabels = df.columns.values
        self.clabels = self.collection_names

    def plot3d (self):
        if self.DIMS != 3:
            print ("There was a problem, Trying to do a 3D plot for a non-3D data.")
        clabels = self.collection_names
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

    def ca(self, collectionset, names):
        # use dd to create a list of all websites in the collectionset
        dd = self.unionize(collectionset)
        #populate table with matches for actors (weblists)
        df = self.create_matrix(dd, collectionset)
        if self.REMOVE_SINGLES:
            df = self.remove_singles(df)
        self.fill_vars(df)
        return(True)
    
    def mca(self, collectionset, names):
        #print ([x[2] for y in collectionset for x in y][0:3])
        default = defaultdict(list)
        coll = defaultdict(list)
        src_index, var_index, d = [], [], []
        for x in collectionset:
            for y,k,v in x:
                default[y+'%'+k].append(v)
        #print(list(default)[0:3])
        dd = self.unionize([j for y, j in default.items()])
        #print (dd)
        for key, val in default.items():
            #print (key)
            keypair = key.split("%")
            collect, year = keypair[0], keypair[1]
            coll[collect].append(year)
            d.append({url: url in val for url in dd})
        for happy, sad in coll.items():
            src_index = (src_index + [happy] * len(sad))
        #src_index = (happy * len(sad) for happy, sad in coll.items())
            var_index = (var_index + sad)
        col_index = pd.MultiIndex.from_arrays([src_index, var_index], names=["Collection", "Date"])
        #X = {x for x in (self.unionize(collectionset))}
        table1 = pd.DataFrame(data=d, index=col_index, columns=dd)
        if self.REMOVE_SINGLES:
            table1 = table1.loc[:, table1.sum(0) >1 ]
        table2 = mca.mca(table1)
        #print (table2.index)
        self.response = table1
        self.dimensions = table2.L 
        #print(table2.inertia)
        fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
        data = pd.DataFrame(columns=table1.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, self.DIMS+1)]))
        #print(data)
        noise = 0.07 * (np.random.rand(*data.T[fs].shape) - 0.5)
        if self.DIMS > 2:
            data.loc[fs, :] = table2.fs_r(N=self.DIMS).T
            self.result["rows"] = table2.fs_r(N=self.DIMS).T
            self.result["columns"] = table2.fs_c(N=self.DIMS).T
            self.result["df"] = data.T[fs].add(noise).groupby(level=['Collection'])
            
        data.loc[fs,    :] = table2.fs_r(N=self.DIMS).T
 #       print(data.loc[fs, :])

        #print(points)
        urls = table2.fs_c(N=self.DIMS).T
        self.plabels = var_index        

        fs_by_source = data.T[fs].add(noise).groupby(level=['Collection'])

        fs_by_date = data.T[fs]
        self.dpoints = data.loc[fs].values
        print(self.dpoints[1:3])
        fig, ax = plt.subplots(figsize=(10,10))
        plt.margins(0.1)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)')
        plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
        ax.margins(0.1)
        markers = '^', 's', 'o', 'o', 'v', "<", ">", "p", "8", "h"
        colors = 'r', 'g', 'b', 'y', 'orange', 'peachpuff', 'm', 'c', 'k', 'navy'
        for fscore, marker, color in zip(fs_by_source, markers, colors):
            #print(type(fscore))
            label, points = fscore
            ax.plot(*points.T.values[0:1], marker=marker, color=color, label=label, linestyle='', alpha=.5, mew=0, ms=12)
            for plabel, x, y in zip(self.plabels, *self.dpoints[1:3]):
                plt.annotate(plabel, xy=(x, y), xytext=(x + .15, y + .15))
        ax.legend(numpoints=1, loc=4)
        plt.show()

    def duplicates (self):
        return(set([x for x in l if l.count(x) > 1]) == set())
    

    def  plot_ca (self, asfile=""):
        texts = []
        ctexts = []
        plt.figure(figsize=(10,10))
        plt.margins(0.1)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)')
        plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
        plt.scatter(*self.result['columns'].T,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
        plt.scatter(*self.result['rows'].T,  s=120, marker='s', c='blue', alpha=.5, linewidths=0)
        for clabel, x, y in zip(self.rlabels, *self.result['columns'].T):
            ctexts.append(plt.text(x, y, clabel))
        if self.LABEL_BOTH_FACTORS:
            for label, x, y in zip(self.clabels, *self.result['rows'].T):
                texts.append(plt.text(x, y, label))
            if self.adjust:
                    AT(texts,arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
                    AT(ctexts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
        if asfile:
            plt.savefig(asfile, bbox_inches='tight')
        plt.show()

    def plot_ca_3d(self):
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        plt.margins(0.1)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        ax.set_xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)')
        ax.set_ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
        ax.set_zlabel('Factor 3 (' + str(round(float(self.dimensions[2]), 3)*100) + '%)')
        ax.scatter(*self.result['columns'],  s=120, marker='o', c='r', alpha=.5, linewidths=0)
        ax.scatter(*self.result['rows'], s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
        for clabel, x, y, z in zip(self.clabels, *self.result['rows']):
            ax.text(x,y,z,  '%s' % (clabel), size=20, zorder=1, color='k')



if __name__ == "__main__":
    with open('./data/parliamentary_committees.json') as f:
        data = json.load(f)
    values = [y['membership'] for  y in data.values()]
    names = [q for q in data.keys()]
    print(names)
    compare = compare(values, names)
    compare.LABEL_BOTH_FACTORS = True
    compare.adjust = True
    compare.plot_ca()





