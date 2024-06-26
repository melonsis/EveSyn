import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from mechanisms.mechanism import Mechanism
from collections import defaultdict
from hdmm.matrix import Identity
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
import time
import evmechanisms.evtools

"""
This file contains a EveSyn construction example in the update phase.
For more details of Private-PGM and its implemention, please visit
https://github.com/ryan112358/private-pgm

Before using this or any other mechanisms in EveSyn, make sure you have
already prepared source code of hdmm and mbi for dependences and put the "src" 
folder's path to PYTHONPATH.
"""

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

def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans

class AIM(Mechanism):
    def __init__(self,epsilon,delta,prng=None,rounds=None,max_model_size=80,structural_zeros={},log_file = None):  
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.log_file = log_file

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt) 

        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, W, cliquesave):
        rounds = self.rounds or 16*len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1] 

        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        epsilon = np.sqrt(8*0.1*self.rho/rounds)
       
        measurements = []
        if self.log_file == None:
            log_file = evmechanisms.evtools.info_logger("======MECH START======")
        else:
            log_file = self.log_file
            evmechanisms.evtools.info_logger("======MECH START======", log_file[0], log_file[1])
        cliques = [] #EveSyn:For clique save

        time_start = time.time()
        # print(self.rho)
        evmechanisms.evtools.info_logger('[V] Rho:'+str(self.rho), log_file[0], log_file[1])
        # print('Initial Sigma', sigma)
        evmechanisms.evtools.info_logger('[V] Initial Sigma'+str(sigma), log_file[0], log_file[1]) 
        rho_used = len(oneway)*0.5/sigma**2 
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
        while not terminate:
            t += 1
            if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                epsilon = np.sqrt(8*0.1*remaining)
                terminate = True

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size*rho_used/self.rho

            small_candidates = filter_candidates(candidates, model, size_limit) 
          
            cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n) 
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()
            cliques.append(cl) 

            model = engine.estimate(measurements) 
            w = model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            evmechanisms.evtools.info_logger('[V] Selected'+str(cl)+'Size'+str(n)+'Budget Used'+str(rho_used/self.rho), log_file[0], log_file[1])
            
            if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n: 
                # print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
                evmechanisms.evtools.info_logger('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma'+str(sigma/2), log_file[0], log_file[1])
                sigma /= 2
                epsilon *= 2

        
        engine.iters = 2500
        model = engine.estimate(measurements)

        #time_end = time.time()
        #time_consume=int(round((time_end-time_start) * 1000))

        #print('Time cost:'+str(time_consume)+' ms.Saving model, cliques and measurements...')

        if cliquesave is not '0':
            cliquepd = pd.DataFrame(cliques,columns=None)
            cliquepd.to_csv(cliquesave+"/cliques.csv",index=False) #EveSyn: Save all selected cliques
        

        #print('Generating Data...')
        evmechanisms.evtools.info_logger('[V] Generating Data...', log_file[0], log_file[1])
        synth = model.synthetic_data()

        return synth

def default_params(): # ~177, Parameter defination. No actually functional code in this segment.Containing parameter initializing, function calling, etc.
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/colorado.csv'
    params['domain'] = '../data/colorado-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['noise'] = 'laplace'
    params['max_model_size'] = 80
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000
    params['cliquesave'] = '../data'

    return params
        
if __name__ == "__main__":

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--max_model_size', type=float, help='maximum size (in megabytes) of model')
    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')
    parser.add_argument('--save', type=str, help='path to save synthetic data')
    parser.add_argument('--cliquesave', type=str, help='path to save cliques')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]

    workload = [(cl, 1.0) for cl in workload]
    mech = AIM(args.epsilon, args.delta, max_model_size=args.max_model_size)
    synth = mech.run(data, workload,args.cliquesave)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*wgt*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors))
