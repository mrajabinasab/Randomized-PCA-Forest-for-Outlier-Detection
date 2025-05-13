import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.decomposition import PCA
from progress.bar import Bar
import pickle
import random
import string
import math
import threading


warnings.filterwarnings('ignore', category=DeprecationWarning)


class IdGen:
    def __init__(self, size):
        self.ids = self.generate_ids(size)
    
    def generate_ids(self, size):
        ids = set()
        while len(ids) < size:
            random_string = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=math.ceil(math.log(size) / math.log(62)))) #automatically determines lowest possible number of characters
            ids.add(random_string) 
        return list(ids)
    
    def draw(self):
        id = random.choice(self.ids)
        self.ids.remove(id)
        return id
    

class RandomizedPCATree:

    max_depth = {}

    def __init__(self, data, dataidx, npc, ec, depth, id):

        self.left = None
        self.leftidx = None
        self.right = None
        self.rightidx = None
        self.data = data
        self.dataidx = dataidx
        self.size = data.shape[0]
        self.npc = npc
        self.rpca = None
        self.split = None
        self.leftchild = None
        self.rightchild = None
        self.endcriteria = ec
        self.depth = depth
        self.id = id


    def fit(self):
        self.rpca = PCA(n_components = self.npc, svd_solver = 'randomized')
        self.rpca.fit(self.data)
        projdata = self.rpca.transform(self.data)
        self.split = np.empty(self.npc)
        for u in range(self.npc):
            tempmean = np.mean(projdata[:, u])
            tempdisp = np.std(projdata[:, u])
            self.split[u] = np.random.laplace(tempmean, tempdisp, 1)
        lhc = 0
        rhc = 0
        for m in range(self.data.shape[0]):
            lc = rc = 0
            for n in range(self.npc):
                if projdata[m, n] < self.split[n]:
                    lc += 1
                else:
                    rc += 1
            if rc < lc:
                lhc += 1
            else:
                rhc += 1
        self.left = np.empty([lhc, self.data.shape[1]])
        self.leftidx = np.empty([lhc])
        self.right = np.empty([rhc, self.data.shape[1]])
        self.rightidx = np.empty([rhc])
        lhc = 0
        rhc = 0
        for z in range(self.data.shape[0]):
            lc = rc = 0
            for n in range(self.npc):
                if projdata[z, n] < self.split[n]:
                    lc += 1
                else:
                    rc += 1
            if rc < lc:
                self.left[lhc, :] = self.data[z, :]
                self.leftidx[lhc] = self.dataidx[z]
                lhc += 1
            else:
                self.right[rhc, :] = self.data[z, :]
                self.rightidx[rhc] = self.dataidx[z]
                rhc += 1
        self.leftchild = RandomizedPCATree(self.left, self.leftidx, self.npc, self.endcriteria, self.depth + 1, self.id)
        self.rightchild = RandomizedPCATree(self.right, self.rightidx, self.npc, self.endcriteria, self.depth + 1, self.id)
        if self.id not in RandomizedPCATree.max_depth:
            RandomizedPCATree.max_depth[self.id] = self.depth + 1
        else:
            if self.depth + 1 > RandomizedPCATree.max_depth[self.id]:
               RandomizedPCATree.max_depth[self.id] = self.depth + 1
        self.data = None
        self.dataidx = None
        if self.left.shape[0] > self.endcriteria:
            self.leftchild.fit()
        self.left = self.leftidx = None
        if self.right.shape[0] > self.endcriteria:
            self.rightchild.fit()
        self.right = self.rightidx = None
        return


def _predict_proba(instance, node):
    if node.size <= node.endcriteria:
        return (node.depth / node.max_depth[node.id]), 1 - (node.depth / node.max_depth[node.id])
    rpcad = node.rpca.transform(instance.reshape(1, -1))
    lc = rc = 0
    for j in range(node.npc):
        if rpcad[:, j] < node.split[j]:
            lc += 1
        else:
            rc += 1
    if rc < lc:
        return _predict_proba(instance, node.leftchild)
    else:
        return _predict_proba(instance, node.rightchild)


def _predict_score(instance, node):
    if node.size <= node.endcriteria:
        if node.size < 2: 
            return float('-inf'), 0
        else:
            temp = []
            for i in range(node.size):
                temp.append(np.linalg.norm(instance - node.data[i]))
            return float('-inf'), sum(temp) / node.size
    rpcad = node.rpca.transform(instance.reshape(1, -1))
    lc = rc = 0
    for j in range(node.npc):
        if rpcad[:, j] < node.split[j]:
            lc += 1
        else:
            rc += 1
    if rc < lc:
        return _predict_score(instance, node.leftchild)
    else:
        return _predict_score(instance, node.rightchild)
    

def predict_proba(data, model, verbos):
    if verbos:
        if isinstance(model, list):
            bar = Bar('  Calculating Probabilities For The Forest...', max=data.shape[0])
        else:
            bar = Bar('  Calculating Probabilities For The Tree...', max=data.shape[0])  
    result = []
    if isinstance(model, list):
        for i in range(data.shape[0]):
            temp = []
            for tree in model:
                temp.append(_predict_proba(data[i, :], tree)[1])
            result.append([1 - np.mean(temp), np.mean(temp)])
            if verbos:
                bar.next()
    else:
        for i in range(data.shape[0]):
            result.append(_predict_proba(data[i, :], model))
            if verbos:
                bar.next()
    if verbos:        
        bar.finish()
    return np.array(result)


def predict_score(data, model, verbos):
    if verbos:
        if isinstance(model, list):
            bar = Bar('  Calculating Scores For The Forest...', max=data.shape[0])
        else:
            bar = Bar('  Calculating Scores For The Tree...', max=data.shape[0])  
    result = []
    if isinstance(model, list):
        for i in range(data.shape[0]):
            temp = []
            for tree in model:
                temp.append(_predict_score(data[i, :], tree)[1])
            result.append([float('-inf'), np.mean(temp)])
            if verbos:
                bar.next()
    else:
        for i in range(data.shape[0]):
            result.append(_predict_score(data[i, :], model))
            if verbos:
                bar.next()
    if verbos:        
        bar.finish()
    return np.array(result)


def predict(predict_criterion, contamination, verbos):
    if verbos:
        bar = Bar('  Deciding Outlier Labels...', max=len(predict_criterion))  
    outlierness_proba = [proba[1] for proba in predict_criterion]
    outliers_labels = np.zeros(len(outlierness_proba))
    sorted_proba = np.sort(outlierness_proba)
    cutoff_index = int((1 - contamination) * len(sorted_proba))
    cutoff_value = sorted_proba[cutoff_index]
    for i in range(len(outlierness_proba)):
        if outlierness_proba[i] >= cutoff_value:
            outliers_labels[i] = 1
        bar.next()
    if verbos:        
        bar.finish()
    return outliers_labels.astype(int)
    

def forest(fittedforest, t, data, npc, ec, verbos, idg):
    if verbos:
        if t == 1:
            bar = Bar('  Fitting The Tree...', max=t)
        else:
            bar = Bar('  Fitting The Trees in The Forest...', max=t) 
    myforest = []
    idx = np.array(range(data.shape[0]))
    for i in range(t):
        retries = 0
        max_retries = 100 #can change it manually, but recursion errors are rare and 100 is more than enough.
        while retries < max_retries:
            try:
                tree = RandomizedPCATree(data, idx, npc, ec, 0, idg.draw())
                tree.fit()
                myforest.append(tree)
                if verbos:
                    bar.next()
                break
            except RecursionError as e:
                retries += 1
                if myforest:
                    myforest.pop()
                print(f" Recursion Error encountered. Retrying... ({retries}/{max_retries})")
        if retries == max_retries:
            print("Failed to fit the tree after maximum retries.")
            return None
    if verbos:
        bar.finish()
    if fittedforest is not None:
        fittedforest.append(myforest)
    else:
        return myforest



def process_dataset(dataset, args, idg):
    print("Processing Dataset " + dataset + "...")
    path_data = args.dataset +  f"/{dataset}/data.csv"
    path_labels = args.dataset + f"/{dataset}/labels.csv"
    data = pd.read_csv(path_data).to_numpy()
    labels = pd.read_csv(path_labels).to_numpy()
    # Adjust labels, based on LMU repository datasets, should be deleted if you have 0 and 1 labels.
    for i in range(labels.shape[0]):
        if labels[i] == "b'no'":
            labels[i] = 0
        if labels[i] == "b'yes'":
            labels[i] = 1
    labels = labels.astype(int)
    if args.threads > args.forestsize:
        args.threads = args.forestsize
    forests = [[] for _ in range(args.threads)]
    myforest = []
    base_size = args.forestsize // args.threads
    remainder =  args.forestsize % args.threads
    threads = []
    for i in range(args.threads):
        size = base_size + (1 if i < remainder else 0)  
        thread = threading.Thread(target=forest, args=(forests[i], size, data, args.principalcomponents, args.leafsize, args.verbos, idg))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    myforest = sum((forest[0] for forest in forests), [])
    
    prob_forest = predict_proba(data, myforest, args.verbos)
    score_forest = predict_score(data, myforest, args.verbos)
    
    prob_forest = [proba[1] for proba in prob_forest]
    score_forest = [score[1] for score in score_forest]
    
    coeffs = np.array(prob_forest)
    scores = np.array(score_forest)
    final_score = coeffs * scores
    
    auc_forest_prob = roc_auc_score(labels, prob_forest)
    auc_forest_score = roc_auc_score(labels, score_forest)
    auc_forest_comb_mp = roc_auc_score(labels, final_score)

    print()
    return {
        'dataset': dataset,
        'labels': labels,
        'fitted_forest': myforest,
        'forest_size': args.forestsize,
        'auc_forest_prob': auc_forest_prob,
        'auc_forest_score': auc_forest_score,
        'auc_forest_comb_mp': auc_forest_comb_mp,
        'prob_forest': prob_forest,
        'score_forest': score_forest,
        'final_score': final_score,
    }


parser = argparse.ArgumentParser(description='OD - Randomized PCA Forest')
parser.add_argument("-d", "--dataset", help="Path to the datasets.", default="./datasets", type=str)
parser.add_argument("-f", "--forestsize", help="Number of trees in the forest.", default=100, type=int)
parser.add_argument("-p", "--principalcomponents", help="Number of principal components to use.", default=5, type=int)
parser.add_argument("-l", "--leafsize", help="Maximum size of a node to be considered a leaf.", default=10, type=int)
parser.add_argument("-c", "--contamination", help="Amount of outliers expected. Should be in range [0, 0.5].", default=0.1, type=float)
parser.add_argument("-r", "--recursionlimit", help="Maximum number of recursions allowed.", default=1000, type=int)
parser.add_argument("-t", "--threads", help="Number of threads used.", default=4, type=int)
parser.add_argument("-v", "--verbos", help="Set it to 1 to enable verbosity, 0 to disable it.", default=1, type=int)
parser.add_argument("-o", "--save", help="Filename to save the results to, 0 to disable saving.", default="0", type=str)
parser.add_argument("-i", "--load", help="Filename to load the results from, 0 to disable loading.", default="0", type=str)
args = parser.parse_args()
sys.setrecursionlimit(args.recursionlimit)

print("OD - Randomized PCA Forest")
print('*'.center(80, '*'))

datasets = [name for name in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, name))]
idg = IdGen(100000) #pool of unique ids
if args.load == "0":
    results = [process_dataset(dataset, args, idg) for dataset in sorted(datasets)]
    if args.save != "0":
        with open(args.save, 'wb') as file:
            pickle.dump(results, file)
else:
    with open(args.load, 'rb') as file:
        results = pickle.load(file)
    # write your own process to view or plot the results.