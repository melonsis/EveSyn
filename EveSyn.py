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
# Load configs from file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    conf_args = AlterArgs(**config)
    return conf_args

# Determine whether the updated data is incremental
def have_intersection(df1, df2):
    # Check whether the previous_dataset is a subset of current_dataset
    is_subset = df1.isin(df2).all().all()
    # Check the column
    columns_match = set(df1.columns) == set(df2.columns)
    return is_subset and columns_match

# Determine each g_i
def domainSize(dataObj, saved_marginals):
    cliques = set()
    #EveSyn: Get saved cliques (sames as marginals)
    cliquepd = pd.read_csv(saved_marginals).values.tolist()
    for line in cliquepd:
        if line[1] is np.nan:
            cliques.add(line[0])
        else:
            cliques.update(set(line))
    return np.log(dataObj.domain.size(cliques))
    
# Determine budgets for initial two timestamps  
def ETuning(budget, w, strategy):
    # For high-initial strategy, we allocate half of the remaining budget for each timestamp
    if strategy == "high-initial":
        init_budget = budget / 2
    # For balance strategy, we allocate a balance budget across all timestamps
    elif strategy == "balance":
        init_budget = budget / w
    else:
        print("Wrong strategy inputted. Using high-initial by default")
        init_budget = budget / 2
    return init_budget


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
    # Loading initial data for process workload and domain
    # Note that this dataset will not using in experiments
    data = Dataset.load(args.dataset, args.domain)
    prng = np.random
    # EveSyn: Prepare Workload
    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]
    # Add default weights for AIM
    if args.mech == "aim":
        workload = [(cl, 1.0) for cl in workload]
    
    # Init log file and parameters
    attr_name = []
    attr_name.append("Dataset")
    attr_name.append("BudgetsRemain")
    attr_name.append("BudgetsThisRound")
    attr_name.append("TimeConsume")
    log_file_name = evmechanisms.evtools.log_init(attr_name=attr_name)
    log_file = evmechanisms.evtools.info_logger("======Experiment START======")
    dataset_name = args.dataname
    budget_remain = args.epsilon
    #List for logging used budgets
    budget_used = []
    # A list for storage g_i
    g_list = []
    # Eta for invokes OriginalSyn
    eta = 0

    # We set the test for 10 rounds (2w) in this example
    # Example experiment start
    for i in range(1,11):
        print("Starting data synthesis at timestamp"+str(i)+" ...",end="",flush=True)
        # Load previous data for further updates
        # Each "original_i.csv" corresponds to the "snapshot" of dataset at timestamp i
        current_dataset = Dataset.load("./data/original/original_"+str(i)+".csv", args.domain)

        # Initial two timestamps
        if i<=2:
            # APBM initialized allocation with ETuning
            # We use high-initial strategy in this example
            init_budget = ETuning(budget=budget_remain, w=args.wsize, strategy="high-initial")
            # Allocate budget to current timestamp
            mech_para = evmechanisms.evexp.args_handler(args, init_budget, log_file=log_file)
            # Calling original_syn
            synth_data, exp_results = evmechanisms.evexp.original_syn(mech_para = mech_para, mech_type=args.mech, data=current_dataset, workload=workload)
            synth_data.df.to_csv("./data/synth/synth_"+str(i)+".csv", index=False)
            print("Done")
            # Consume the allocated budget
            budget_remain = budget_remain - init_budget
            # Catch used budgets
            budget_used.append(init_budget)
            # Log the results
            exp_results.insert(0, str(init_budget))
            exp_results.insert(0, str(budget_remain))
            exp_results.insert(0, dataset_name+"_Original")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            # Calculate and log g_i
            g_list.append(domainSize(dataObj=data, saved_marginals=args.cliques))
            continue
        
        # Load previous dataset when i>2
        previous_dataset = Dataset.load("./data/original/original_"+str(i-1)+".csv", args.domain)
        last_synth = Dataset.load("./data/synth/synth_"+str(i-1), args.domain)

        # When budget runs out, use synthetic data in i-1 as the output
        if budget_remain == 0:
            last_synth.df.to_csv("./data/synth/synth_"+str(i)+".csv")
            budget_used.append(0)
            exp_results.insert(0, str(0))
            exp_results.insert(0, str(budget_remain))
            exp_results.insert(0, dataset_name+"_Optimized")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            # Calculate and log g_i
            g_list.append(domainSize(dataObj=data, saved_marginals=args.cliques))
            # Determine remain budgets using only recent w budget_used
            if i > args.wsize:
                budget_remain = args.epsilon - sum(budget_used[i-args.wsize:i+1])
            else:
                budget_remain = args.epsilon - sum(budget_used[0:i+1])
            continue

        # Allocate initial budget for this round
        budget_per_round = budget_remain / args.wsize
        # Tuning the budget and restrict the tuned budget less than remaining budget
        budget_tuned = min(budget_per_round * (g_list[i-1]/g_list[i-2]), budget_remain)
        # Allocate budget to current timestamp
        mech_para = evmechanisms.evexp.args_handler(args, budget_tuned, log_file=log_file)
        
        # If eta >= 5, invokes original_syn 
        if eta >= 5:
            # Invoking original_syn
            synth_data, exp_results = evmechanisms.evexp.original_syn(mech_para = mech_para, mech_type=args.mech, data=current_dataset,workload=workload)
            synth_data.df.to_csv("./data/synth/synth_"+str(i)+".csv", index=False)
            # Consume the allocated budget
            budget_remain = budget_remain - budget_tuned
            budget_used.append(budget_tuned)
            # Determine remain budgets using only recent w budget_used
            if i > args.wsize:
                budget_remain = args.epsilon - sum(budget_used[i-args.wsize:i+1])
            else:
                budget_remain = args.epsilon - sum(budget_used[0:i+1])
            exp_results.insert(0, str(budget_tuned))
            exp_results.insert(0, str(budget_remain))
            exp_results.insert(0, dataset_name+"_Original")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            # Calculate and log g_i
            g_list.append(domainSize(dataObj=data, saved_marginals=args.cliques))
            eta = 0
            continue


        # Determine whether the updated data is incremental only
        if have_intersection(previous_dataset.df, current_dataset.df):
            # Get the incremetal part of updated dataset
            updated_df = current_dataset.df.iloc[len(previous_dataset):].reset_index(drop=True)
            # Synth the data
            synth_data, exp_results, budget_error = evmechanisms.evexp.update(mech_para=mech_para, mech_type=args.mech, data = Dataset(updated_df,args.domain), last_synth=last_synth, workload=workload)
            synth_new = pd.concat(last_synth.df, synth_data.df)
            synth_new.to_csv("./data/synth/synth_"+str(i)+".csv")
            # Catch used budgets, here is 0
            budget_used.append(0)
            # Determine remain budgets using only recent w budget_used
            if i > args.wsize:
                budget_remain = args.epsilon - sum(budget_used[i-args.wsize:i+1])
            else:
                budget_remain = args.epsilon - sum(budget_used[0:i+1])
            # Log the results
            exp_results.insert(0, str(0))
            exp_results.insert(0, str(budget_remain))
            exp_results.insert(0, dataset_name+"_Optimized")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            # Calculate and log g_i
            g_list.append(domainSize(dataObj=data, saved_marginals=args.cliques))
            continue
        # Normal update process start
        else:
            synth_data, exp_results, budget_error = evmechanisms.evexp.update(mech_para=mech_para, mech_type=args.mech, data = current_dataset, last_synth=last_synth, workload=workload)
            # Calculate error
            errors = []
            errors_p = []
            if args.mech == "mwem":
                error_weight = False
                sigma = np.sqrt(2*np.log(1.25/args.delta)) / (budget_error / 2)
            else:
                error_weight = True
                sigma = np.sqrt(budget_error / 2)
            errors = evmechanisms.evmech.error_universal(data=current_dataset, synth=synth_data, workload=workload, weighted=error_weight, method = args.error_method) + np.random.normal(loc=0, scale=sigma, size=None)
            errors_p = evmechanisms.evmech.error_universal(data=current_dataset, synth=last_synth, workload=workload, weighted=error_weight, method = args.error_method) + np.random.normal(loc=0, scale=sigma, size=None)
            # Ensure the data utility
            if errors < errors_p:
                synth_data.df.to_csv("./data/synth/synth_"+str(i)+".csv")
                budget_remain = budget_remain - budget_tuned
                budget_used.append(budget_tuned)
            else:
                last_synth.df.to_csv("./data/synth/synth_"+str(i)+".csv")
                budget_used.append(0)
                eta += 1

            # Determine remain budgets using only recent w budget_used
            if i > args.wsize:
                budget_remain = args.epsilon - sum(budget_used[i-args.wsize:i+1])
            else:
                budget_remain = args.epsilon - sum(budget_used[0:i+1])
            
            # Log the results
            exp_results.insert(0, str(budget_tuned))
            exp_results.insert(0, str(budget_remain))
            exp_results.insert(0, dataset_name+"_Optimized")
            evmechanisms.evtools.log_append(exp_results, log_file_name[0], log_file_name[1])
            g_list.append(domainSize(dataObj=data, saved_marginals=args.cliques))
            print("Done")





