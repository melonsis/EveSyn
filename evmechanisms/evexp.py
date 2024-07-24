import evmechanisms.evmech
import pandas as pd
import numpy as np
import time
import mechanisms.aim
import mechanisms.mwem_pgm
from scipy.special import softmax
from scipy import sparse
import random
from mbi import Dataset, Domain

# Mech_para: DICT
# ArgList: 
#   Mechnism basic:
#       epsilon(from exp), delta,max_model_size
#       cliques_in, rounds, pgm_iters, save


def args_handler(args, budgets,log_file):
    mech_para = {}
    mech_para['delta'] = args.delta
    mech_para['max_model_size'] = args.max_model_size
    mech_para['cliques_in'] = args.cliques
    mech_para['rounds'] = args.rounds
    mech_para['pgm_iters'] = args.pgm_iters
    mech_para['save'] =args.save

    mech_para['budgets'] = budgets
    mech_para['log_file'] = log_file

    return mech_para


def original_syn(mech_para:dict, mech_type, data, workload):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    max_model_size = mech_para['max_model_size']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    log_file = mech_para['log_file']
    clique_save = "./data"

    time_start = time.time()
    if mech_type == "mwem":
        synth = mechanisms.mwem_pgm.mwem_pgm(data, epsilon,clique_save, delta, 
                    rounds= rounds,
                    workload=workload,
                    maxsize_mb= max_model_size,
                    pgm_iters= pgm_iters,
                    log_file = log_file)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))
        # print('Time cost:'+str(time_consume)+' ms.Saving model, cliques and measurements...')

    if mech_type == "aim":
        mech = mechanisms.aim.AIM(epsilon, delta, 
                            max_model_size=max_model_size, 
                            log_file = log_file) 
        synth= mech.run(data, workload,clique_save)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))

    return synth, [time_consume]

def worst_estimated(workload, last_synth, current_dataset_answer, budget, l):
    """
    workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    last_synth: Last round's synthetic data
    workload: The list of candidates to consider in the exponential mechanism
    budget: The privacy budget to use for this step.
    l: The number of select marginals

    """
    errors = np.array([])
    for cl in workload:
        bias = last_synth.domain.size(cl)
        x = current_dataset_answer[cl]
        xsyn = last_synth.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xsyn).sum()-bias)
    sensitivity = 2.0
    prob = softmax(0.5 * budget/sensitivity*(errors - errors.max()))
    keys = np.random.choice(len(errors), p=prob, size=l)
    selected = []
    for key in keys:
        selected.append(workload[key])
    return selected

# Here we set default l = 3
def CTuning(budget, last_synth, current_dataset_answer, workload, cliques_in, strategy, l):
    # Process the cliques by strategy
    # Unchanged will not read or modificate the original cliques
    if strategy == "Unchanged":
        remain_budget = budget
    
    # Add or Replace will modificate the original cliques
    else:
        # Read previous cliques (same as marginals)
        cliques_in_list = []
        history_pd = pd.read_csv(cliques_in).values.tolist()
        for line in history_pd:
            if line[1] is np.nan:
                cliques_in_list.append((line[0],))
            else:
                cliques_in_list.append(tuple(line))

        budget_cl = remain_budget / 10
        remain_budget = remain_budget - budget_cl
        worst_l = worst_estimated(last_synth=last_synth, current_dataset_answer=current_dataset_answer, workload=workload, budget=budget_cl, l=l)
        if strategy == "Add":
            for cl in worst_l:
                cliques_in_list.append(cl)
        elif strategy == "Replace":
            for cl in worst_l:
                random_index = random.randint(0, len(cliques_in_list) - 1)
                cliques_in_list.pop(random_index)
                cliques_in_list.append(cl)
        
        cliquepd = pd.DataFrame(cliques_in_list,columns=None)
        cliquepd.to_csv(cliques_in, index=False) #EveSyn: Save the modified cliques

    return remain_budget


def update(mech_para:dict, mech_type, data, last_synth, workload):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    max_model_size = mech_para['max_model_size']
    cliques_in = mech_para['cliques_in']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    log_file = mech_para['log_file']
    
    current_data_answer = { cl : data.project(cl).datavector() for cl in workload }
    # Tuning the cliques by CTuning
    # Additionally select cliques (strategy Add and Replace) may consume budget 
    epsilon = CTuning(budget=epsilon, last_synth=last_synth, current_dataset_answer=current_data_answer, workload=workload, cliques_in=cliques_in, strategy="Unchanged", l=3)
    time_start = time.time()
    if mech_type == "mwem":
        synth,remaining = evmechanisms.evmech.ev_mwem(data, epsilon, delta, 
                    cliques_in=cliques_in,
                    rounds= rounds,
                    workload=workload,
                    maxsize_mb= max_model_size,
                    pgm_iters= pgm_iters,
                    log_file = log_file)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))

    if mech_type == "aim":
        mech = evmechanisms.evmech.ev_AIM(epsilon, delta, 
                            max_model_size=max_model_size,
                            cliques_in = cliques_in, 
                            log_file = log_file) 
        synth,remaining = mech.run(data, workload)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))
    return synth, [time_consume], remaining