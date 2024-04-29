import ismechanisms.ismech
import pandas as pd
import numpy as np
import time
import mechanisms.aim
import mechanisms.mwem_pgm

# Mech_para: DICT
# ArgList: 
#   Mechnism basic:
#       epsilon(from exp), delta, lastsyn_load, max_model_size
#       cliques_in, eta_max, rounds, pgm_iters, save
#   Exp:
#       budgets, del_percent, incre_percent, incre_count
def exp_single(mech_para:dict, mech_type, data, workload, error_method):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    lastsyn_load = mech_para['lastsyn_load']
    max_model_size = mech_para['max_model_size']
    cliques_in = mech_para['cliques_in']
    eta_max = mech_para['eta_max']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    save = mech_para['save']
    log_file = mech_para['log_file']

    prefer_cliques = []
    errors = []
    errors_p = []
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist()
    for line in prefer_pd:
        prefer_cliques.append(tuple(line))
    time_start = time.time()
    if mech_type == "mwem":
        synth = ismechanisms.ismech.is_mwem(data, epsilon,lastsyn_load, delta, 
                    cliques_in=cliques_in,
                    rounds= rounds,
                    workload=workload,
                    maxsize_mb= max_model_size,
                    pgm_iters= pgm_iters,
                    eta_max = eta_max,
                    log_file = log_file)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))
        # print('Time cost:'+str(time_consume)+' ms.Saving model, cliques and measurements...')
        errors = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=workload, method = error_method)
        errors_p = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=prefer_cliques, method = error_method)
    if mech_type == "aim":
        mech = ismechanisms.ismech.is_AIM(epsilon, delta, lastsyn_load, 
                            max_model_size=max_model_size,
                            cliques_in = cliques_in, 
                            eta_max=eta_max,
                            log_file = log_file) 
        synth = mech.run(data, workload)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))

        prefer_cliques = [(cl, 1.0) for cl in prefer_cliques]
        errors = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=workload, weighted=True, method = error_method)
        errors_p = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=prefer_cliques, weighted=True, method = error_method)

    if save is not None:
        synth.df.to_csv(save, index=False)
    return [np.mean(errors), np.mean(errors_p),time_consume]

def args_handler(args, budgets,log_file):
    mech_para = {}
    mech_para['delta'] = args.delta
    mech_para['lastsyn_load'] = args.lastsyn
    mech_para['max_model_size'] = args.max_model_size
    mech_para['cliques_in'] = args.cliques
    mech_para['eta_max'] = args.eta
    mech_para['rounds'] = args.rounds
    mech_para['pgm_iters'] = args.pgm_iters
    mech_para['save'] =args.save

    mech_para['budgets'] = budgets
    mech_para['log_file'] = log_file

    return mech_para

def original_syn(mech_para:dict, mech_type, data, workload, error_method):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    lastsyn_load = mech_para['lastsyn_load']
    max_model_size = mech_para['max_model_size']
    cliques_in = mech_para['cliques_in']
    eta_max = mech_para['eta_max']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    save = mech_para['save']
    log_file = mech_para['log_file']
    clique_save = "./data"

    prefer_cliques = []
    errors = []
    errors_p = []
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist()
    for line in prefer_pd:
        prefer_cliques.append(tuple(line))
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
        errors = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=workload, method = error_method)
        errors_p = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=prefer_cliques, method = error_method)
    if mech_type == "aim":
        mech = mechanisms.aim.AIM(epsilon, delta, 
                            max_model_size=max_model_size, 
                            log_file = log_file) 
        synth = mech.run(data, workload,clique_save)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))

        prefer_cliques = [(cl, 1.0) for cl in prefer_cliques]
        errors = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=workload, weighted=True, method = error_method)
        errors_p = ismechanisms.ismech.error_universal(data=data, synth=synth, workload=prefer_cliques, weighted=True, method = error_method)

    if save is not None:
        synth.df.to_csv(save, index=False)
    return [np.mean(errors), np.mean(errors_p),time_consume]

def original_syn_aae(mech_para:dict, mech_type, data, workload):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    lastsyn_load = mech_para['lastsyn_load']
    max_model_size = mech_para['max_model_size']
    cliques_in = mech_para['cliques_in']
    eta_max = mech_para['eta_max']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    save = mech_para['save']
    log_file = mech_para['log_file']
    clique_save = "./data"

    prefer_cliques = []
    errors = []
    errors_p = []
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist()
    for line in prefer_pd:
        prefer_cliques.append(tuple(line))
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
        errors = ismechanisms.ismech.error_aae(data=data, synth=synth, workload=workload)
        errors_p = ismechanisms.ismech.error_aae(data=data, synth=synth, workload=prefer_cliques)
    if mech_type == "aim":
        mech = mechanisms.aim.AIM(epsilon, delta, 
                            max_model_size=max_model_size, 
                            log_file = log_file) 
        synth = mech.run(data, workload,clique_save)
        time_end = time.time()
        time_consume=int(round((time_end-time_start) * 1000))

        prefer_cliques = [(cl, 1.0) for cl in prefer_cliques]
        errors = ismechanisms.ismech.error_aae(data=data, synth=synth, workload=workload)
        errors_p = ismechanisms.ismech.error_aae(data=data, synth=synth, workload=prefer_cliques)

    if save is not None:
        synth.df.to_csv(save, index=False)
    return [np.mean(errors), np.mean(errors_p),time_consume]