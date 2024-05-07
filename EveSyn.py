import evmechanisms.evmech
import evmechanisms.evtools
import evmechanisms.evexp
import argparse
from mbi import Dataset, Domain
import pandas as pd
import numpy as np
import itertools
import os
import json

class AlterArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    conf_args = AlterArgs(**config)
    return conf_args

def default_params(): 
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['config'] = None
    params['dataset'] = '../data/colorado.csv'
    params['domain'] = '../data/colorado-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['noise'] = 'laplace'
    params['max_model_size'] = 80
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000
    params['cliques'] = '../data/cliques.csv'
    params['lastsyn'] = None
    params['eta'] = 5
    params['rounds'] = None
    params['pgm_iters'] = 250
    params['mech'] = 'mwem'

    return params

if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--config', help="path of config file")
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')
    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')
    parser.add_argument('--save', type=str, help='path to save synthetic data')
    parser.add_argument('--cliques', help='cliques that used')
    parser.add_argument('--lastsyn', help = 'last synthetic data')
    parser.add_argument('--eta', type=int, help = 'Threshold of eta')
    parser.add_argument('--mech', type=str,help='The mechanism used for test')
    parser.add_argument('--rounds', type=int, help='number of rounds of MWEM to run')
    parser.add_argument('--pgm_iters', type=int, help='number of iterations')

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    if args.config is not None:
        if os.path.exists(args.config):
            config_args = load_config(args.config)
            args = config_args
            print("Loaded config from config file!")
        else:
            print("Config file not found, turn to use default/inputted config")

    data = Dataset.load(args.dataset, args.domain)
    # IncreSyn: Pre-parse mech
    # args.mech = args.mech.lower
    prng = np.random
    # IncreSyn: Prepare Workload
    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
    if args.mech == "aim":
        workload = [(cl, 1.0) for cl in workload]
        
    # IncreSyn: Load last synthetic data
    if args.lastsyn is not None:
        lastsyn_load = Dataset.load(args.lastsyn, args.domain)
    else:
        lastsyn_load = None
    
    # Init log file
    attr_name = []
    attr_name.append("Dataset")
    attr_name.append("UniversalError")
    attr_name.append("PreferedError")
    attr_name.append("TimeConsume")
    log_file_name = evmechanisms.evtools.log_init(attr_name=attr_name)
    log_file = evmechanisms.evtools.info_logger("======Experiment START======")
    mech_para = evmechanisms.evexp.args_handler(args, args.epsilon,log_file=log_file) # This epsilon is for iteration experiments
    exp_results = []
    dataset_name = "Adult"
    
    print("Starting exp round 1...",end="",flush=True)
    
    exp_results = evmechanisms.evexp.original_syn_aae(mech_para=mech_para, mech_type=args.mech, data=data,workload=workload)
    exp_results = evmechanisms.evexp.exp_single(mech_para=mech_para, )
    exp_results.insert(0, dataset_name+"_Original")
    evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])

    
    exp_results = evmechanisms.evexp.exp_single(mech_para=mech_para, mech_type=args.mech, data=data,workload=workload,error_method=args.error_method)
    exp_results.insert(0, str(args.epsilon))
    exp_results.insert(0, dataset_name+"_Optimized")
    evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
    print("Done")





