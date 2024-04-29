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
import ismechanisms.istools

def eta_test(data, lastsyn_load, syn, workload):
    """
    Test function for adaptively increse eta
    data: Original data
    lastsyn_load: Last synthetic data
    syn: Synthetic data of this time
    workload: The workload using for test
    """
    errors_last = []
    errors_this = []
    for proj in workload:
        O = data.project(proj).datavector()
        X = lastsyn_load.project(proj).datavector()
        Y = syn.project(proj).datavector()
        elast = 0.5*np.linalg.norm(O/O.sum() - X/X.sum(), 1)
        ethis = 0.5*np.linalg.norm(O/O.sum() - Y/Y.sum(), 1)
        errors_last.append(elast)
        errors_this.append(ethis)

    if np.mean(errors_this) < np.mean(errors_last):
        return 1
    else:
        return 0 
    
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

class is_AIM(Mechanism):
    def __init__(self,epsilon,delta,lastsyn_load,eta_max, prng=None,rounds=None,max_model_size=80,structural_zeros={},cliques_in = "./data/cliques.csv",log_file = None):  
        super(is_AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros
        self.cliques_in = cliques_in
        self.lastsyn_load = lastsyn_load
        self.eta_max = eta_max
        self.log_file = log_file
        
    def worst_approximated_eta(self, candidates, answers, model, eps, sigma, eta):
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
        return self.exponential_mechanism_eta(errors, eta, eps, max_sensitivity)
    

    def run(self, data, W):
        
        rounds = self.rounds or 16*len(data.domain) #IncreSyn: Here we using the original rounds limit, to achieve same 1-way calc budget
        eta = 1 # IncreSyn: Initialzed an eta = 1

        cliques = []
        cliquepd = pd.read_csv(self.cliques_in).values.tolist() #IncreSyn: Get selected cliques
        for line in cliquepd:
            if line[1] is np.nan:
                cliques.append((line[0],))
            else:
                cliques.append(tuple(line))
        #IncreSyn:Load prefer cliques from file
        prefer_pd =  pd.read_csv("./data/prefer.csv").values.tolist()
        for line in prefer_pd:
            if line[1] is np.nan:
                    cliques.append((line[0],))
            else:
                    cliques.append(tuple(line))
        
       
       
        
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1] 
        rho_used = 0

        
        sigma = np.sqrt(rounds / (2*0.9*self.rho))


        measurements = []
        if self.log_file == None:
            log_file = ismechanisms.istools.info_logger("======MECH START======")
        else:
            log_file = self.log_file
        
        # print('Initial Sigma', sigma)
        ismechanisms.istools.info_logger('[V] Initial Sigma'+str(sigma), log_file[0], log_file[1])
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
        # IncreSyn: After the completion of a 1-way measurements, we reset the maximum number of rounds to be equal to the total length of cliques (with prefer attributes), in order to avoid allocating too much budget for 1-way measurements. 
        # Once this is set, the subsequent process can be considered as allocating a fixed budget per round.
        if self.lastsyn_load is None:
            rounds = len(cliques)
        else:
            rounds = len(cliques)+eta
        sigma = np.sqrt(rounds / (2 * remaining)) #IncreSyn: Re-design sigma
         #print("!!!Re-design sigma after one-way!")
        #print("New sigma:",sigma)
        ismechanisms.istools.info_logger('[V] !!!Re-design sigma after one-way!', log_file[0], log_file[1])
        ismechanisms.istools.info_logger('[V] New sigma:'+str(sigma), log_file[0], log_file[1])
        

          #IncreSyn: When the last synthetic data is given, run the select step once
        if self.lastsyn_load is not None: 
            print('Last synthetic data detected, adding selection')
            epsilon = np.sqrt(8*0.1*self.rho/rounds)
            #rho_used += epsilon
            choice_cl = self.worst_approximated_eta(candidates, answers, self.lastsyn_load, epsilon, sigma,eta)
            for cl in choice_cl:
                if cl not in cliques:
                    cliques.append(cl)
                else:
                    rounds = rounds-1
            remaining = remaining - 1/sigma**2 
            sigma = np.sqrt(rounds / (2 * remaining))  #IncreSyn: Re-design sigma after selection
            print("!!!Re-design sigma after selection!")
            print("New sigma:",sigma)
        

        while t < rounds and not terminate:
            t += 1
            cl = None
            if (self.rho - rho_used <0.5/sigma**2): #IncreSyn: Change the limitation
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                # We do not needs epsilon here
                # epsilon = np.sqrt(8*0.1*remaining) 
                terminate = True
            rho_used += 0.5/sigma**2 #IncreSyn: Remove epsilon here
            cl = cliques[t-1]        #IncreSyn: Switch the original select method to reading selected cliques line by line.
            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            ismechanisms.istools.info_logger('[V] Selected'+str(cl)+'Size'+str(n)+'Budget Used'+str(rho_used/self.rho), log_file[0], log_file[1])


        ismechanisms.istools.info_logger('[V] Total rounds:'+str(t), log_file[0], log_file[1])
        # print("Total rounds:",t)
        engine.iters = 2500
        model = engine.estimate(measurements) #IncreSyn: Move the estimation outside of the iteration.

        # print('Generating Data...')
        ismechanisms.istools.info_logger('[V] Generating Data...', log_file[0], log_file[1])
        synth = model.synthetic_data()
        if self.lastsyn_load is not None:
            error_comp = eta_test(data=data, lastsyn_load=self.lastsyn_load, syn=synth, workload=workload) #IncreSyn:Test for whether eta should gets bigger or not
            if (error_comp == 1) and eta < self.eta_max:
                eta +=1
                print("Eta increased to "+str(eta))

        return synth

def worst_approximated_eta_mwem(workload_answers, est, workload, eps, eta, penalty=True):
    """ Select eta (noisy) worst-approximated marginal for measurement.
    
    :param workload_answers: a dictionary of true answers to the workload
        keys are cliques
        values are numpy arrays, corresponding to the counts in the marginal
    :param est: a GraphicalModel object that approximates the data distribution
    :param: workload: The list of candidates to consider in the exponential mechanism
    :param eps: the privacy budget to use for this step.
    :param eta: the number of selected cliques
    """
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum()-bias)
    sensitivity = 2.0
    prob = softmax(0.5*eps/sensitivity*(errors - errors.max()))
    keys = np.random.choice(len(errors), p=prob,size = eta)
    choice_cl = []
    for key in keys:
        choice_cl.append(workload[key])
    return choice_cl

def is_mwem(data,epsilon, lastsyn_load, delta=0.0, cliques_in=None,rounds=None, workload = None, maxsize_mb = 25, pgm_iters=100, noise='laplace', eta_max=5, log_file = None):

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
    eta = 1 # IncreSyn: Initialzed an eta = 1
    if workload is None:
        workload = list(itertools.combinations(data.domain, 2))
    
    answers = { cl : data.project(cl).datavector() for cl in workload } #IncreSyn: Get workload answers

    cliques = []
    cliquepd = pd.read_csv(cliques_in).values.tolist() #IncreSyn:Get selected cliques
    for line in cliquepd:
        cliques.append(tuple(line)) 
    #IncreSyn:Add prefer cliques
    prefer_cliques = []
    #IncreSyn:Load prefer cliques from file
    prefer_pd = pd.read_csv("./data/prefer.csv").values.tolist() #IncreSyn: Get prefer cliques
    for line in prefer_pd:
            prefer_cliques.append(tuple(line))
    #IncreSyn:Add prefer cliques to original cliques
    cliques += prefer_cliques

    if rounds is None:
        if lastsyn_load is None:
            rounds = len(cliques)
        else:
            rounds = len(cliques)+eta

    if noise == 'laplace':
        eps_per_round = epsilon / (2 * rounds)
        sigma = 1.0 / eps_per_round
        exp_eps = eps_per_round
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
        log_file = ismechanisms.istools.info_logger("======MECH START======")
    else:
        ismechanisms.istools.info_logger("======MECH START======", log_file[0], log_file[1])

    #IncreSyn: When the last synthetic data is given, run the select step once
    if lastsyn_load is not None: 
        print('Last synthetic data detected, adding selection')
        choice_cl = worst_approximated_eta_mwem(workload_answers = answers, est = lastsyn_load, workload = workload, eta=eta, eps = exp_eps)

        for cl in choice_cl:
            if cl not in cliques:
                cliques.append(cl)
            else:
                rounds = rounds-1
        
        eps_per_round = (epsilon - exp_eps) / (2 * rounds)
        sigma = 1.0 / eps_per_round #IncreSyn: Re-calculate sigma

    for i in range(1, rounds+1):
        ax = cliques[i-1] #IncreSyn: Switch the original select method to reading selected cliques line by line.
        # print('Round', i, 'Selected', ax, "Eps per round =",eps_per_round)
        info = '[V] Round' + str(i) + 'Selected' + str(ax) + "Eps per round =" + str(eps_per_round)
        ismechanisms.istools.info_logger(info, log_file[0], log_file[1])
        n = domain.size(ax)
        x = data.project(ax).datavector()
        if noise == 'laplace':
            y = x + np.random.laplace(loc=0, scale=marginal_sensitivity*sigma, size=n)
        else:
            y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
    est = engine.estimate(measurements, total) #IncreSyn: Move the estimation outside of the iteration.
    ismechanisms.istools.info_logger('[V] Generating Data...', log_file[0],log_file[1])
    # print('Generating Data...')
    syn = est.synthetic_data()
    if lastsyn_load is not None:
        error_comp = eta_test(data=data, lastsyn_load=lastsyn_load, syn=syn, workload=workload) #IncreSyn:Test for whether eta should gets bigger or not
        if (error_comp == 1) and eta < eta_max:
            eta +=1
            print("Eta increased to "+str(eta))
    return syn

def prefer_picker(workload, previous_cliques, prefer_count,name):
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

def error_universal(data, synth, workload, method='mae', weighted=False):
    errors = []
    for item in workload:
        if isinstance(item, tuple) and weighted:
            proj, wgt = item  # 解包投影和权重
        else:
            proj = item if not isinstance(item, tuple) else item[0]
            wgt = 1           # 默认权重为1

        # 获取数据向量
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()

        # 根据选择的方法计算误差
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