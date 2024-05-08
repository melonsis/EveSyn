import evmechanisms.evmech
import pandas as pd
import numpy as np
import time
import mechanisms.aim
import mechanisms.mwem_pgm

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

def update(mech_para:dict, mech_type, data, workload):
    epsilon = mech_para['budgets']
    delta = mech_para['delta']
    max_model_size = mech_para['max_model_size']
    cliques_in = mech_para['cliques_in']
    rounds = mech_para['rounds']
    pgm_iters = mech_para['pgm_iters']
    log_file = mech_para['log_file']

    cliques_in_list = []
    history_pd = pd.read_csv(cliques_in).values.tolist()
    for line in history_pd:
        if line[1] is np.nan:
            cliques_in_list.append((line[0],))
        else:
            cliques_in_list.append(tuple(line))
    time_start = time.time()
    if mech_type == "mwem":
        synth,remaining = evmechanisms.evmech.ev_mwem(data, epsilon,delta, 
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
    return synth, [remaining, time_consume]