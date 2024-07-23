import numpy as np
import itertools
from mbi import GraphicalModel, FactoredInference
from mechanisms.mechanism import Mechanism
from collections import defaultdict
from hdmm.matrix import Identity
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import time 
from scipy.special import softmax
from scipy import sparse
from mechanisms.cdp2adp import cdp_rho
import random
import evmechanisms.evtools


def powerset(iterable): # Calculting for powerset
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws): 
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques): 
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20

def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

class ev_AIM(Mechanism):
    def __init__(self,epsilon,delta,lastsyn_load, prng=None,rounds=None,max_model_size=80,structural_zeros={},cliques_in = "./data/cliques.csv",log_file = None):  
        super(ev_AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.cliques_in = cliques_in
        self.lastsyn_load = lastsyn_load
        self.log_file = log_file
        
    
    def run(self, data, W):
        
        rounds = self.rounds or 16*len(data.domain) #EveSyn: Here we using the original rounds limit, to achieve same 1-way calc budget

        cliques = []
        cliquepd = pd.read_csv(self.cliques_in).values.tolist() #EveSyn: Get selected cliques
        for line in cliquepd:
            if line[1] is np.nan:
                cliques.append((line[0],))
            else:
                cliques.append(tuple(line))
        
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)

        oneway = [cl for cl in candidates if len(cl) == 1] 
        rho_used = 0

        sigma = np.sqrt(rounds / (2*0.9*self.rho))

        measurements = []
        if self.log_file == None:
            log_file = evmechanisms.evtools.info_logger("======MECH START======")
        else:
            log_file = self.log_file
        
        # print('Initial Sigma', sigma)
        evmechanisms.evtools.info_logger('[V] Initial Sigma'+str(sigma), log_file[0], log_file[1])
        rho_used += len(oneway)*0.5/sigma**2 
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size) 
            measurements.append((I, y, sigma, cl))
        zeros = self.structural_zeros
        engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros) 
        model = engine.estimate(measurements)
        t = 0
        terminate = False
        
        remaining = self.rho - rho_used
        # EveSyn: After the completion of a 1-way measurements, we reset the maximum number of rounds to be equal to the total length of cliques (with prefer attributes), in order to avoid allocating too much budget for 1-way measurements. 
        # Once this is set, the subsequent process can be considered as allocating a fixed budget per round.
        rounds = len(cliques) + 2
        sigma = np.sqrt(rounds / (2 * remaining)) #EveSyn: Re-design sigma
        evmechanisms.evtools.info_logger('[V] !!!Re-design sigma after one-way!', log_file[0], log_file[1])
        evmechanisms.evtools.info_logger('[V] New sigma:'+str(sigma), log_file[0], log_file[1])
        
        while t < rounds-2 and not terminate:
            t += 1
            cl = None

            rho_used += 0.5/sigma**2 #EveSyn: Remove epsilon here
            cl = cliques[t-1]        #EveSyn: Switch the original select method to reading selected cliques line by line.
            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

            evmechanisms.evtools.info_logger('[V] Selected'+str(cl)+'Size'+str(n)+'Budget Used'+str(rho_used/self.rho), log_file[0], log_file[1])


        evmechanisms.evtools.info_logger('[V] Total rounds:'+str(t), log_file[0], log_file[1])
        engine.iters = 2500
        model = engine.estimate(measurements) #EveSyn: Move the estimation outside of the iteration.

        evmechanisms.evtools.info_logger('[V] Generating Data...', log_file[0], log_file[1])
        synth = model.synthetic_data()

        return synth, self.rho-rho_used


def ev_mwem(data,epsilon, lastsyn_load, delta=0.0, cliques_in=None,rounds=None, workload = None, maxsize_mb = 25, pgm_iters=100, noise='laplace',  log_file = None):

    """
    Implementation of a dynamic update version of MWEM+PGM

    :param data: an *ORIGINAL* mbi.Dataset object
    :param epsilon: privacy budget
    :param delta: privacy parameter (ignored)
    :param cliques: A list of cliques (attribute tuples) which choosen in original synthetic mechanism
    :param rounds: The number of rounds of MWEM to run (default: number of attributes)
    :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
        Used to avoid MWEM+PGM failure modes (intractable model sizes).   
        Set to np.inf if you would like to run MWEM as originally described without this modification 
        (Note it may exceed resource limits if run for too many rounds)

    Implementation Notes:
    - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
        not increase size of the graphical model too much
    """ 
    if workload is None:
        workload = list(itertools.combinations(data.domain, 2))
    
    answers = { cl : data.project(cl).datavector() for cl in workload } #EveSyn: Get workload answers

    cliques = []
    cliquepd = pd.read_csv(cliques_in).values.tolist() #EveSyn:Get selected cliques
    for line in cliquepd:
        cliques.append(tuple(line)) 
    #EveSyn:Add prefer cliques

    if rounds is None:
        rounds = len(cliques)+2
    else:
        rounds += 2

    if noise == 'laplace':
        eps_per_round = epsilon / rounds
        sigma = 1.0 / eps_per_round
        marginal_sensitivity = 2
    else:
        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / (2 * rounds)
        sigma = np.sqrt(0.5 / rho_per_round)
        exp_eps = np.sqrt(8 * rho_per_round)
        marginal_sensitivity = np.sqrt(2)

    domain = data.domain
    total = data.records
    def size(cliques):
        return GraphicalModel(domain, cliques).size * 8 / 2**20
    engine = FactoredInference(data.domain, log=False, iters=pgm_iters, warm_start=True)
    measurements = []
    if log_file == None:
        log_file = evmechanisms.evtools.info_logger("======MECH START======")
    else:
        evmechanisms.evtools.info_logger("======MECH START======", log_file[0], log_file[1])

    for i in range(1, rounds-1):
        ax = cliques[i-1] #EveSyn: Switch the original select method to reading selected cliques line by line.
        info = '[V] Round' + str(i) + 'Selected' + str(ax) + "Eps per round =" + str(eps_per_round)
        evmechanisms.evtools.info_logger(info, log_file[0], log_file[1])
        n = domain.size(ax)
        x = data.project(ax).datavector()
        if noise == 'laplace':
            y = x + np.random.laplace(loc=0, scale=marginal_sensitivity*sigma, size=n)
        else:
            y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
    est = engine.estimate(measurements, total) #EveSyn: Move the estimation outside of the iteration.
    evmechanisms.evtools.info_logger('[V] Generating Data...', log_file[0],log_file[1])
    syn = est.synthetic_data()
    remaining = eps_per_round*2
    return syn, remaining

def error_universal(data, synth, workload, method='mae', weighted=False):
    errors = []
    for item in workload:
        if isinstance(item, tuple) and weighted:
            proj, wgt = item  # Get the weight and project
        else:
            proj = item if not isinstance(item, tuple) else item[0]
            wgt = 1           # Give an initial weight at 1

        # Get the data vector
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()

        # Compute the error with given method
        if method == 'mae':
            norm_X = X / X.sum()
            norm_Y = Y / Y.sum()
            e = 0.5 * wgt * np.linalg.norm(norm_X - norm_Y, 1)
        elif method == 'aae':
            diff = np.abs(X - Y)
            e = np.mean(diff)
        else:
            raise ValueError("Unsupported error calculation method")

        errors.append(e)
    return errors