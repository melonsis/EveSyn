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

# Determine whether the updated data is incremental
def have_intersection(df1, df2):
    intersection = pd.merge(df1, df2, how='inner')
    return not intersection.empty

def default_params(): 
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['config'] = None
    params['dataset'] = './data/colorado.csv'
    params['domain'] = './data/colorado-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['noise'] = 'laplace'
    params['max_model_size'] = 80
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000
    params['cliques'] = './data/cliques.csv'
    params['rounds'] = None
    params['pgm_iters'] = 250
    params['mech'] = 'mwem'
    params['wsize'] = 5

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
    parser.add_argument('--mech', type=str,help='The mechanism used for test')
    parser.add_argument('--rounds', type=int, help='number of rounds of MWEM to run')
    parser.add_argument('--pgm_iters', type=int, help='number of iterations')
    parser.add_argument('--wsize', type=int, help='size of w')

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    if args.config is not None:
        if os.path.exists(args.config):
            config_args = load_config(args.config)
            args = config_args
            print("Loaded config from config file!")
        else:
            print("Config file not found, turn to use default/inputted config")
    # Loading initial data
    data = Dataset.load(args.dataset, args.domain)
    prng = np.random
    # EveSyn: Prepare Workload
    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
    if args.mech == "aim":
        workload = [(cl, 1.0) for cl in workload]
    
    # Init log file
    attr_name = []
    attr_name.append("Dataset")
    attr_name.append("BudgetsRemain")
    attr_name.append("BudgetsThisRound")
    attr_name.append("TimeConsume")
    log_file_name = evmechanisms.evtools.log_init(attr_name=attr_name)
    log_file = evmechanisms.evtools.info_logger("======Experiment START======")
    exp_results = []
    dataset_name = args.dataname

    mech_para = evmechanisms.evexp.args_handler(args, args.epsilon / 2,log_file=log_file) # APBM initialized allocation
    print("Starting data synthesis at timestamp 1...",end="",flush=True)
    # Calling original_syn
    # The error calculated in this timestamp is only for evaluation, so we do not split any budget to protect it.
    # It should not be calculated when used in a real publishing scenario.
    synth_data, exp_results = evmechanisms.evexp.original_syn(mech_para = mech_para, mech_type=args.mech, data=data,workload=workload, error_method=args.error_method)
    synth_data.df.to_csv("./data/synth/synth_0.csv", index=False)
    print("Done")
    rho_remain = args.epsilon / 2
    # Catch remaining budgets
    rho_used = []
    rho_used.append(args.epsilon / 2)
    # We set the test for 10 rounds (2w) in this example
    # Log the results
    exp_results.insert(0, str(rho_used[0]))
    exp_results.insert(0, str(rho_remain))
    exp_results.insert(0, dataset_name+"_Original")
    evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
    # Start updates
    for i in range(1,10):
        print("Starting data synthesis at timestamp"+str(i)+" ...",end="",flush=True)
        rho_per_round = rho_remain / args.wsize
        previous_dataset = Dataset.load("./data/original/original_"+str(i-1)+".csv", args.domain)
        current_dataset = Dataset.load("./data/original/original_"+str(i)+".csv", args.domain)
        last_synth = Dataset.load("./data/synth/synth_"+str(i-1), args.domain)
        rho_used.append(0)

        if not have_intersection(previous_dataset.df, current_dataset.df):
            updated_df = pd.merge(previous_dataset.df, current_dataset.df, how='outer', indicator=True).query('_merge=="right_only"').drop('_merge', axis=1)
            synth_data, exp_results = evmechanisms.evexp.update(mech_para=mech_para, mech_type=args.mech, data = Dataset(updated_df,args.domain),workload=workload,error_method=args.error_method)
            synth_new = pd.concat(last_synth.df, synth_data.df)
            synth_new.to_csv("./data/synth/synth_"+str(i)+".csv")
        else:
            synth_data, exp_results = evmechanisms.evexp.update(mech_para=mech_para, mech_type=args.mech, data = current_dataset,workload=workload,error_method=args.error_method)
        # Calculate error
            errors = []
            errors_p = []
            eps_error = exp_results[0]/2
            sigma = 1.0 / eps_error
            if args.mech == "mwem":
                error_weight = False
            else:
                error_weight = True
            errors = evmechanisms.evmech.error_universal(data=current_dataset, synth=synth_data, workload=workload, weighted=error_weight, method = args.error_method) + np.random.laplace(loc=0, scale=sigma, size=None)
            errors_p = evmechanisms.evmech.error_universal(data=current_dataset, synth=last_synth, workload=workload, weighted=error_weight, method = args.error_method) + np.random.laplace(loc=0, scale=sigma, size=None)
            if errors < errors_p:
                synth_data.df.to_csv("./data/synth/synth_"+str(i)+".csv")
                rho_used.append(rho_per_round)
            else:
                last_synth.df.to_csv("./data/synth/synth_"+str(i)+".csv")
                rho_used.append(0)
            
            
            # Determine remain budgets using only recent w rho_used
            if i > args.wsize:
                rho_remain = args.epsilon - sum(rho_used[i-args.wsize:i+1])
            else:
                rho_remain = args.epsilon - sum(rho_used[0:i+1])
            # Log the results
            exp_results.insert(0, str(rho_remain))
            exp_results.insert(0, dataset_name+"_Optimized")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            print("Done")





