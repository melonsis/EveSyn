import numpy as np
import itertools
import pandas as pd
import random
import time
import argparse
from mbi import Dataset

def PreferPicker(workload, previous_cliques, prefer_count,name):
    pre_count = 0
    prefer_flag = "./Results/prefered/prefer_"+name+str(int(time.time()))+".csv"
    print("Generating..")
    if previous_cliques is list:
        pre_count = len(previous_cliques)
    prefer_cliques = []

    if pre_count == 0:
        prefer_cliques = random.sample(workload, prefer_count - pre_count)
    elif prefer_count - pre_count >= 0:
        prefer_cliques += previous_cliques
        while len(prefer_cliques)!=prefer_count:
            clique = random.sample(workload, 1)
            if clique not in prefer_cliques:
                prefer_cliques.append(clique)

    prefer_pd = pd.DataFrame(prefer_cliques,columns=None)
    print("Saving to data folder...")
    prefer_pd.to_csv("./data/prefer.csv",index=False)
    print("Saving to results folder...")
    prefer_pd.to_csv(prefer_flag,index=False)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = './data/colorado/colorado_less.csv'
    params['domain'] = './data/colorado/colorado_less-domain.json'
    params['count'] = 10
    params['previous'] = 0
    params['name'] = "colorado"

    return params

if __name__ == "__main__":
    preparse=argparse.ArgumentParser(description="Prefer Select Tools")
    preparse.add_argument('-d','--dataset',help="Input dataset path")
    preparse.add_argument('--domain',help="Input domain")
    preparse.add_argument('-c','--count',type=int,help="Prefer counts")
    preparse.add_argument('-p','--previous',help = "Previous cliques")
    preparse.add_argument('--name',help = "Previous cliques")
    
    preparse.set_defaults(**default_params())
    args = preparse.parse_args()

    data = Dataset.load(args.dataset,args.domain)
    workload = list(itertools.combinations(data.domain , 2))
    workload = [cl for cl in workload if data.domain.size(cl) <= 10000]
    
    print("Target dataset:",args.name)
    if args.previous == 0:
        print("No previous cliques. Generating new...")
        previous_cliques = 0
    else:
        # Load prefer cliques from file
        print("Detected previous cliques,reading...")
        previous_cliques = []
        previous_pd = pd.read_csv(previous_csv).values.tolist()
        for line in previous_pd:
            previous_cliques.append(tuple(line))
    
    PreferPicker(workload,previous_cliques,args.count,args.name)