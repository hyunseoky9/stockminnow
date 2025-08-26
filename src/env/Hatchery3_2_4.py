import itertools
from scipy.stats import multivariate_hypergeom, norm, beta
import pickle
import numpy as np
from math import floor
import random
import pandas as pd
import sys
import os
from AR1_normalized import AR1_normalized


class Hatchery3_2_4:
    """
    Same as hatchery 3.2, but all the fall actions are taken at once, and spring decision is not taken. It's assumed that maximum capacity is produced every year.
    action is a vector of 4, where the first three are stocking proportions in angostura, isleta, and san acacia, and the last one is the discard proportion.
    Another fix from Hatchery3_2 is that calculating wild Ne uses only wild origin N0 (N0CF; N0 counter-factual). This prevents double dipping in the R-L equation
    ensuring that Ne,w in the R-L only includes wild-origin spawners. This was not possible in 3.2 because stocked fish were baked into N0 during fall timesteps (t=1,2,3)
    """
    def __init__(self,initstate,parameterization_set,discretization_set,LC_prediction_method, param_uncertainty=0):
        """
        same as 3.2.2 but the simulation doesn't terminate. The population falls to zero if the population goes below some threshold.
        """
        self.envID = 'Hatchery3.2.4'
        self.partial = True
        self.episodic = True
        self.absorbing_cut = True # has an absorbing state and the episode should be cut shortly after reaching it.
        self.discset = discretization_set
        self.param_uncertainty = param_uncertainty # if 1, parameters will be sampled from the posterior distribution from Yackulic et al. 2023. at every reset.
        self.contstate = False
        self.special_stacking = False

        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        # for reach index, 1 = angostura, 2 = isleta, 3 = san acacia
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_hatchery3.1.csv'
        paramdf = pd.read_csv(parameterization_set_filename)
        self.n_reach = 3
        self.alpha = paramdf['alpha'][self.parset] 
        self.alphavar = paramdf['alphavar'][self.parset]
        self.beta = paramdf['beta'][self.parset] 
        self.Lmean = paramdf['Lmean'][self.parset]
        self.mu = np.array([paramdf['mu_a'][self.parset],paramdf['mu_i'][self.parset],paramdf['mu_s'][self.parset]])
        self.sd = paramdf['sd'][self.parset]
        self.beta_2 = paramdf['beta_2'][self.parset]
        #self.beta_stk = paramdf['beta_stk'][self.parset]
        self.tau = paramdf['tau'][self.parset]
        self.r0 = paramdf['r0'][self.parset]
        self.r1 = paramdf['r1'][self.parset]
        self.delfall = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.deldiff = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.delfall[0] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall1_i'][self.parset], paramdf['delfall1_s'][self.parset]])
        self.delfall[1] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall2_i'][self.parset], paramdf['delfall2_s'][self.parset]])
        self.deldiff[0] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff1_i'][self.parset], paramdf['deldiff1_s'][self.parset]])
        self.deldiff[1] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff2_i'][self.parset], paramdf['deldiff2_s'][self.parset]])
        self.phifall = np.array([paramdf['phifall_a'][self.parset],paramdf['phifall_i'][self.parset],paramdf['phifall_s'][self.parset]])
        self.phidiff = np.array([paramdf['phidiff_a'][self.parset],paramdf['phidiff_i'][self.parset],paramdf['phidiff_s'][self.parset]])
        self.lM0mu = np.array([paramdf['lM0mu_a'][self.parset],paramdf['lM0mu_i'][self.parset],paramdf['lM0mu_s'][self.parset]])
        self.lM1mu = np.array([paramdf['lM1mu_a'][self.parset],paramdf['lM1mu_i'][self.parset],paramdf['lM1mu_s'][self.parset]])
        self.lMwmu = np.array([paramdf['lMwmu_a'][self.parset],paramdf['lMwmu_i'][self.parset],paramdf['lMwmu_s'][self.parset]])
        self.lM0sd = np.array([paramdf['lM0sd_a'][self.parset],paramdf['lM0sd_i'][self.parset],paramdf['lM0sd_s'][self.parset]])
        self.lM1sd = np.array([paramdf['lM1sd_a'][self.parset],paramdf['lM1sd_i'][self.parset],paramdf['lM1sd_s'][self.parset]])
        self.lMwsd = np.array([paramdf['lMwsd_a'][self.parset],paramdf['lMwsd_i'][self.parset],paramdf['lMwsd_s'][self.parset]])
        self.irphi = paramdf['irphi'][self.parset]
        self.p0 = paramdf['p0'][self.parset]
        self.p1 = paramdf['p1'][self.parset]
        self.sz = paramdf['sz'][self.parset]
        self.fpool_f = np.array([paramdf['fpoolf_a'][self.parset],paramdf['fpoolf_i'][self.parset],paramdf['fpoolf_s'][self.parset]])
        self.fpool_s = np.array([paramdf['fpools_a'][self.parset],paramdf['fpools_i'][self.parset],paramdf['fpools_s'][self.parset]])
        self.frun_f = np.array([paramdf['frunf_a'][self.parset],paramdf['frunf_i'][self.parset],paramdf['frunf_s'][self.parset]])
        self.frun_s = np.array([paramdf['fruns_a'][self.parset],paramdf['fruns_i'][self.parset],paramdf['fruns_s'][self.parset]])
        self.avgeff_fr = np.array([paramdf['avgeff_fr_a'][self.parset],paramdf['avgeff_fr_i'][self.parset],paramdf['avgeff_fr_s'][self.parset]])
        self.avgeff_fp = np.array([paramdf['avgeff_fp_a'][self.parset],paramdf['avgeff_fp_i'][self.parset],paramdf['avgeff_fp_s'][self.parset]])
        self.thetaf = np.array([paramdf['thetaf_a'][self.parset],paramdf['thetaf_i'][self.parset],paramdf['thetaf_s'][self.parset]])
        self.thetas = np.array([paramdf['thetas_a'][self.parset],paramdf['thetas_i'][self.parset],paramdf['thetas_s'][self.parset]])
        self.n_genotypes = paramdf['n_genotypes'][self.parset]
        self.n_locus = paramdf['n_locus'][self.parset]
        self.n_cohorts = paramdf['n_cohorts'][self.parset]
        self.b = np.array([paramdf['b1'][self.parset],paramdf['b2'][self.parset],paramdf['b3'][self.parset],paramdf['b4'][self.parset]])
        self.fc = np.array([paramdf['fc1'][self.parset],paramdf['fc2'][self.parset],paramdf['fc3'][self.parset],paramdf['fc4'][self.parset]])
        self.s0egg = paramdf['s0egg'][self.parset] # survival rate of egg to hatchery
        self.s0larvae = paramdf['s0larvae'][self.parset] # survival rate of hatchery to age 0 cohort
        self.eggcollection_max = paramdf['eggcollection_max'][self.parset] # maximum number of eggs that can be collected 
        self.larvaecollection_max = paramdf['larvaecollection_max'][self.parset] # maximum number of larvae that can be collected 
        self.sc = np.array([paramdf['s1'][self.parset],paramdf['s2'][self.parset],paramdf['s3'][self.parset]]) # cohort survival rate by age group
        self.rlen = np.array([paramdf['angolen'][self.parset], paramdf['isllen'][self.parset], paramdf['salen'][self.parset]]) # reach length in km
        self.dth = 10 # density threshold
        self.Nth_local = self.rlen* self.dth
        self.Nth = np.sum(self.Nth_local)
        self.c = 12
        # no termination: a subpopulation falls to 0 if it goes below its local threshold. The simulation does not terminate.
        self.objective_type = 'no termination' 

        print(f'Nth: {self.Nth}, Nth_local: {self.Nth_local}, c: {self.c}, objective_type: {self.objective_type}')
        self.extant = paramdf['extant'][self.parset] # reward for not being
        self.prodcost = paramdf['prodcost'][self.parset] # production cost in spring if deciding to produce
        self.unitcost = paramdf['unitcost'][self.parset] # unit production cost.
        self.maxcap = paramdf['maxcap'][self.parset] # maximum carrying capacity of the hatchery
        self.Ne2Nratio = paramdf['Ne2Nratio'][self.parset] # ratio of effective population size to total population size in hatchery
        # for monitoring simulation (alternate parameterization for )
        self.sampler = self.sz
        
        # discount factor
        self.gamma = 0.99

        # parameter posterior distribution 
        if self.param_uncertainty:
            param_uncertainty_filename = 'uncertain_parameters_posterior_samples4POMDP.csv'
            self.param_uncertainty_df = pd.read_csv(param_uncertainty_filename)
            self.paramsampleidx = None # initiate sample idx.

        # start springflow simulation model and springflow-to-"Larval carrying capacity" model.
        self.flowmodel = AR1_normalized()
        self.Otowi_minus_ABQ_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[1] # difference between Otowi and ABQ springflow
        self.Otowi_minus_SA_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[2] # difference between Otowi and San Acacia springflow
        self.LC_prediction_method = LC_prediction_method # 0=HMM, 1=GAM
        if self.LC_prediction_method == 0: # HMM
            self.LC_ABQ = pd.read_csv('springflow2LC_hmm_ABQ.csv')
            self.LC_SA = pd.read_csv('springflow2LC_hmm_San Acacia.csv')
        elif self.LC_prediction_method == 1: # linear GAM
            with open('LC_GAM_ABQ.pkl','rb') as handle:
                self.LC_ABQ = pickle.load(handle)
            with open('LC_GAM_Isleta.pkl','rb') as handle:
                self.LC_IS = pickle.load(handle)
            with open('LC_GAM_San Acacia.pkl','rb') as handle:
                self.LC_SA = pickle.load(handle)

        # get some parameters for effective population size calculation
        ## get log(kappa) probability distribution and domain values
        with open('lkappa_dataset.pkl', 'rb') as handle:
            lkappa_dataset = pickle.load(handle)
        self.lkappa_prob = lkappa_dataset['lkappaprob']
        self.angolkappa_midvalues = lkappa_dataset['angolkappa_midvalues']
        self.isllkappa_midvalues = lkappa_dataset['isllkappa_midvalues']
        A,B             = np.meshgrid(self.angolkappa_midvalues,
                                    self.isllkappa_midvalues,
                                    indexing='ij')
        mask            = self.lkappa_prob > 1e-3            # keep only useful combos
        self.kappa_exp  = np.exp(np.stack([A[mask], B[mask], B[mask]], axis=-1))  # (nκ,3)
        self.kappa_prob = self.lkappa_prob[mask]             # (nκ,)

        ## calculate different alpha values (number of eggs produced per spawner) and their probabilities in the posterior distribution
        ## (used for calculating sigma^2_{b_w} in the document Hatchery 3.2).
        # define ends (95% CI) of the posterior distribution
        alphainterval = np.linspace(self.alpha - 1.96*np.sqrt(self.alphavar), self.alpha + 1.96*np.sqrt(self.alphavar), 11)
        # calculate the probability for each bin
        self.alphaprob = norm.cdf(alphainterval[1:], loc=self.alpha, scale=np.sqrt(self.alphavar)) - norm.cdf(alphainterval[:-1], loc=self.alpha, scale=np.sqrt(self.alphavar))
        self.alpha_centers = (alphainterval[:-1] + alphainterval[1:]) / 2
        self.alpha_vec  = np.hstack((self.alpha, self.alpha_centers))  # (nα,)
        ## get the probability distribution of river drying, delfall, and avg deldiff.
        bins = 5
        intervals = np.linspace(0,1,bins+1) # bin boundaries
        delfall_i = np.array([self.delfall[0,1],self.delfall[1,1]]) # isleta alpha beta
        delfall_s = np.array([self.delfall[0,2],self.delfall[1,2]]) # san acacia alpha beta
        delfallprob_i = beta.cdf(intervals[1:], a=delfall_i[0], b=delfall_i[1]) - beta.cdf(intervals[:-1], a=delfall_i[0], b=delfall_i[1])
        delfallprob_s = beta.cdf(intervals[1:], a=delfall_s[0], b=delfall_s[1]) - beta.cdf(intervals[:-1], a=delfall_s[0], b=delfall_s[1])
        delmid_values = (intervals[:-1] + intervals[1:]) / 2 # get the mid value for each bin
        deldiffavg = np.zeros(self.n_reach)
        deldiffavg[1:] = self.deldiff[0,1:] / np.sum(self.deldiff,axis=0)[1:] # average value for deldiff. angostura value is 0

        ## get expected survival rates for age1+ and age 0
        M0 = np.exp(self.lM0mu)
        M1 = np.exp(self.lM1mu)
        Mw = np.exp(self.lMwmu)
        combo = np.array(list(itertools.product(delmid_values, delmid_values)))
        self.combo_delfallprob = np.array(list(itertools.product(delfallprob_i, delfallprob_s)))
        self.combo_delfallprob = np.prod(self.combo_delfallprob, axis=1) # product of delfall probabilities
        valididx = np.where(self.combo_delfallprob > 1e-2)[0] # only keep the combinations with probability greater than 1e-3
        self.combo_delfallprob = self.combo_delfallprob[valididx] # only keep the valid combinations
        delfall = np.column_stack((np.zeros(combo.shape[0]),combo))
        delfall = delfall[valididx,:] # only keep the valid combinations
        self.sj = np.exp(-124*M0 - 150*Mw)*((1 - delfall) + self.tau*delfall*deldiffavg + (1 - self.tau)*self.r0*np.mean(self.phidiff))
        self.sa = np.exp(-124*M1 - 150*Mw)*((1 - delfall) + self.tau*delfall + (1 - self.tau)*self.r1*np.mean(self.phifall))
        self.sj =  np.prod(self.sj,axis=1)**(1/self.sj.shape[1]) # geometric mean of sj
        self.sa =  np.prod(self.sa,axis=1)**(1/self.sa.shape[1]) # geometric mean of sa

        ## get average of age 2+ fish 
        self.avgsa = np.sum(self.sa * self.combo_delfallprob)
        self.AVGage_of_age2plus = 2*(1/(1+self.avgsa)) + 3*(self.avgsa/(1+self.avgsa)) # assume that the fish only lives till age 3
        
        # number of broodstock used for producing maximum capacity. Assumes maximum capacity is produced every year.
        self.stockreadyfish_per_female = np.median([645.4969697,962.1485714,743.7636364,354.9875,634.92]) # first four values from bio park, 5th value from dexter. 
        self.Nb = 2*self.maxcap/self.stockreadyfish_per_female # the value 1000 is Thomas' ballpark estimate of stock-ready fish produced per female #  2*self.maxcap/self.fc[1]
        # the value 1000 is Thomas' ballpark estimate of stock-ready fish produced per female

        # observation related parameters
        self.avgp = np.mean([self.p0, self.p1]) # average p
        self.avgfallf = (self.fpool_f + self.frun_f)/2 # f is the proportion of RGSM in the river segment exposed to sampling
        self.popsize_1cpue = 1/(self.avgfallf*self.avgp*self.thetaf*(100/(self.avgeff_fp+self.avgeff_fr))) # average population size that corresponds to 1 cpue given average p, f (fall), and theta (fall) parameter values.



        # range for each variables
        self.N0minmax = [0,1e7] 
        self.N1minmax = [0,1e7] # N1 and N1 minmax are the total population minmax.
        self.qminmax = [self.flowmodel.flowmin[0], self.flowmodel.flowmax[0]] # springflow in Otowi (Otowi gauge)
        self.Neminmax = [0, 1e7]
        self.aminmax = [0, 300000]
        # dimension for each variables
        self.N0_dim = (self.n_reach)
        self.N1_dim = (self.n_reach)
        self.q_dim = (1)
        self.Ne_dim = (1)
        self.statevar_dim = (self.N0_dim, self.N1_dim, self.q_dim, self.Ne_dim)
        self.obsvar_dim = (self.N0_dim, self.N1_dim, self.q_dim, self.Ne_dim)
        self.action_dim = (1,1,1,1) # 4 actions: stocking proportion in angostura, isleta, and san acacia, and discard rest.

        # starting 3.0, discretization for discrete variables and ranges for continuous variables will be defined in a separate function, state_discretization.
        discretization_obj = self.state_discretization(discretization_set)
        self.states = discretization_obj['states']
        self.observations = discretization_obj['observations']
        self.actions = discretization_obj['actions']

        # define how many discretizations each variable has.
        if self.discset == -1: 
            self.statespace_dim = list(np.ones(np.sum(self.statevar_dim))*-1) # continuous statespace is not defined (marked as -1)
            self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
            self.obsspace_dim = list(np.ones(np.sum(self.obsvar_dim))*-1)
        else:
            self.statespace_dim = np.concatenate((np.array(list(map(lambda x: len(x[1]), self.states.items())))*np.array([np.ones(self.statevar_dim[i]) for i in range(len(self.statevar_dim))],dtype='object')))
            #self.statespace_dim = list(np.array(
            #    list(map(lambda x: len(x[1]), self.states.items()))
            #) * np.array([
            #    self.N0_dim, self.N1_dim, self.Nh_dim, self.q_dim, self.p_dim, np.prod(self.ph_dim), self.ph0_dim, self.G_dim, self.t_dim
            #]))
            self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
            self.obsspace_dim = np.concatenate((np.array(list(map(lambda x: len(x[1]), self.observations.items())))*np.array([np.ones(self.obsvar_dim[i]) for i in range(len(self.obsvar_dim))],dtype='object')))
            #self.obsspace_dim = list(np.array(
            #    list(map(lambda x: len(x[1]), self.observations.items()))
            #) * np.array([
            #    self.N0_dim, self.N1_dim, self.Nh_dim, self.q_dim, 
            #    self.G_dim, self.t_dim
            #]))
            # for discrete springflow, get the index of the discrete springflow values in the LC to springflow mapping table.
            self.ABQq = np.minimum(np.maximum(self.states['q'] - self.Otowi_minus_ABQ_springflow, self.flowmodel.flowmin[1]), self.flowmodel.flowmax[1])
            self.SAq = np.minimum(np.maximum(self.states['q'] - self.Otowi_minus_SA_springflow, self.flowmodel.flowmin[2]), self.flowmodel.flowmax[2])
            if self.LC_prediction_method == 0:
                ABQqfrac = (self.ABQq - self.LC_ABQ['springflow'].values[0])/(self.LC_ABQ['springflow'].values[-1] - self.LC_ABQ['springflow'].values[0])
                SAqfrac = (self.SAq - self.LC_SA['springflow'].values[0])/(self.LC_SA['springflow'].values[-1] - self.LC_SA['springflow'].values[0])

                self.disc_sf_idxs_abq = np.round(ABQqfrac * (len(self.LC_ABQ['springflow']) - 1)).astype(int)
                self.disc_sf_idxs_sa = np.round(SAqfrac * (len(self.LC_SA['springflow']) - 1)).astype(int)

        # idx in the state and observation list for each state/observation variables
        self.sidx = {}
        self.oidx = {}
        self.aidx = {}
        for idx, key in enumerate(self.states.keys()):
            if idx ==0:
                self.sidx[key] = np.arange(0,self.statevar_dim[idx])
            else:
                self.sidx[key] = np.arange(np.sum(self.statevar_dim[0:idx]),np.sum(self.statevar_dim[0:idx]) + self.statevar_dim[idx])
        for idx, key in enumerate(self.observations.keys()):
            if idx ==0:
                self.oidx[key] = np.arange(0,self.obsvar_dim[idx])
            else:
                self.oidx[key] = np.arange(np.sum(self.obsvar_dim[0:idx]),np.sum(self.obsvar_dim[0:idx]) + self.obsvar_dim[idx])
        for idx, key in enumerate(self.actions.keys()):
            if idx ==0:
                self.aidx[key] = np.arange(0,self.action_dim[idx])
            else:
                self.aidx[key] = np.arange(np.sum(self.action_dim[0:idx]),np.sum(self.action_dim[0:idx]) + self.action_dim[idx])

        # Initialize state and observation
        self.state, self.obs = self.reset(initstate)

    def reset(self, initstate=None):
        if type(initstate) is not np.ndarray:
            initstate = np.ones(len(self.statevar_dim))*-1
        
        # Initialize state variables
        new_state = []
        new_obs = []

        # N0 & ON0
        N0val, N1val = self.init_pop_sampler()
        if initstate[0] == -1:
            # N0val = random.choices(list(np.arange(1, len(self.states['N0']))), k = self.statevar_dim[0])
            new_state.append(N0val) # don't start from the smallest population size
            new_obs.append(N0val)
        else:
            new_state.append(np.array([initstate[0]]))
            new_obs.append(np.array([initstate[0]]))
        # N1 & ON1
        if initstate[1] == -1:
            # N1val = random.choices(list(np.arange(1, len(self.states['N1']))), k = self.statevar_dim[1])
            new_state.append(N1val) # don't start from
            new_obs.append(N1val)
        else:
            new_state.append(np.array([initstate[1]]))
            new_obs.append(np.array([initstate[1]]))
        # q & qhat
        if initstate[2] == -1:
            if self.discset == -1:
                qval = np.random.uniform(size=1)*(self.states['logq'][1] - self.states['logq'][0]) + self.states['logq'][0]
            else:
                qval = random.choices(list(np.arange(0,len(self.states['q']))), k = self.statevar_dim[3])
            new_state.append(qval)
            new_obs.append(qval)
        else:
            new_state.append(np.array([initstate[2]]))
            new_obs.append(np.array([initstate[2]]))
        # Ne & ONe
        if self.discset == -1:
            Neval = np.array([np.sum(np.exp(N0val)-1 + np.exp(N1val)-1)*0.6])
            #Neval,_,_ = self.NeCalc(np.exp(N0val)-1, np.exp(N1val)-1, pval[0], Nbval[0])
        else:
            Neval = np.array([np.sum(np.exp(N0val)-1 + np.exp(N1val)-1)*0.6])
            #Neval,_,_ = self.NeCalc(np.exp(N0val)-1, np.exp(N1val)-1, pval, Nbval)
            Neval = self._discretize_idx(Neval, self.states['Ne'])
        new_state.append(np.log(Neval+1))
        new_obs.append(np.log(Neval+1))

        # sample
        if self.param_uncertainty:
            self.paramsampleidx = np.random.randint(0, self.param_uncertainty_df.shape[0]) #np.random.choice([178,3898]) #178 #np.random.randint(0, self.param_uncertainty_df.shape[0])
            paramvals = self.parameter_reset() # resample parameters from the posterior distribution

        self.state = np.concatenate(new_state)
        self.obs = np.concatenate(new_obs)

        return self.obs, self.state

    def step(self, a, current_strategy = 0):
        """
        Take an action and return the next state, reward, done flag, and extra information.
        a is a vector of 4, where the first three are stocking proportions in angostura, isleta, and san acacia, and the last one is the discard proportion.
        current_strategy=1 takes the production action and stocking action based on the currently carried out heuristic stocking strategy.
        """
        extra_info = {}
        if self.discset == -1:
            N0 = np.exp(np.array(self.state)[self.sidx['logN0']]) - 1
            N1 = np.exp(np.array(self.state)[self.sidx['logN1']]) - 1
            q = np.exp(np.array(self.state)[self.sidx["logq"]]) - 1
            Ne = np.exp(np.array(self.state)[self.sidx["logNe"]]) - 1
        else:
            N0 = np.array(self.states["N0"])[np.array(self.state)[self.sidx['N0']]]
            N1 = np.array(self.states["N1"])[np.array(self.state)[self.sidx['N1']]]
            q = np.array(self.states["q"])[np.array(self.state)[self.sidx['q']]]
            Ne = np.array(self.states['Ne'])[np.array(self.state)[self.sidx['Ne']]]
        totN0 = np.sum(N0)
        totN1 = np.sum(N1)
        Nr = N0 + N1 # population size in each reach
        if current_strategy == 1: 
            #a = self.stocking_decision() # stocking decision based on monitoring samples in september/october
            a = self.stocking_decision2(N0,N1) # stocking decision based on current strategy when assuming that you can observe the population size through IPM.
            extra_info['current_strat_action'] = a
        a = a[0:self.n_reach] # only take the first n_reach elements of the action vector
        totpop = totN0 + totN1

        # demographic stuff (stocking and winter survival)
        Mw = np.exp(self.lMwmu) #np.exp(np.random.normal(self.lMwmu, self.lMwsd))
        stockedNsurvived = a*self.maxcap*self.irphi
        N0CF = N0.copy()*np.exp(-150*Mw) # counterfactual N0, if no stocking had been done. Also equivalent to wild-origin spawners.
        N0 = N0 + stockedNsurvived # stocking san acacia (t=3) in the fall
        N0 = np.minimum(N0*np.exp(-150*Mw),np.ones(self.n_reach)*self.N0minmax[1]) # stocking san acacia (t=3) in the fall
        N1 = N1*np.exp(-150*Mw)
        p = stockedNsurvived*np.exp(-150*Mw) # Total number of fish stocked in a season that make it to breeding season
        Ne_score, Neh, Ne_base = self.NeCalc0(N0,N1,p, self.Nb,None,None,1)
        extra_info['Ne_score'] = Ne_score # Ne_score is the Ne until you stock in the next fall.
        # demographic stuff (reproductin and summer survival)

        delfall = np.concatenate(([self.delfall[0][0]],np.random.beta(self.delfall[0][1:],self.delfall[1][1:])))
        deldiff = np.concatenate(([self.deldiff[0][0]],np.random.beta(self.deldiff[0][1:],self.deldiff[1][1:])))
        L, abqsf, sasf = self.q2LC(q)
        extra_info['L'] = L
        extra_info['abqsf'] = abqsf
        extra_info['sasf'] = sasf
        natural_capacity = np.random.normal(self.mu, self.sd)
        kappa = np.exp(self.beta*(L - self.Lmean) + natural_capacity)
        extra_info['natural_capacity'] = natural_capacity
        extra_info['kappa'] = kappa
        # local extinction if the population goes below the local threshold
        for r in range(self.n_reach):
            if N0[r] + N1[r] < self.Nth_local[r]:
                N0[r], N1[r] = 0, 0
        effspawner = N0 + self.beta_2*N1 # effective number of spawners
        P1 = (self.alpha*N0)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 1 fish that newly became adults
        P2 = (self.alpha*self.beta_2*N1)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 2+ fish
        P = (self.alpha*effspawner)/(1 + self.alpha*effspawner/kappa)
        M0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
        M1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
        if np.sum(P)>0:
            genT = (np.sum(P1) + np.sum(P2)*self.AVGage_of_age2plus)/np.sum(P)  # generation time
            N0_next = np.minimum(P*np.exp(-124*M0)*((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff),np.ones(self.n_reach)*self.N0minmax[1])
            N1_next = np.minimum((N0+N1)*np.exp(-215*M1)*((1-delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall),np.ones(self.n_reach)*self.N1minmax[1])
            # Ne calculation
            #Ne_CF, _, _ = self.NeCalc0(N0CF,N1,p,self.Nb,genT,kappa,0) # Ne if no stocking had been done
            if np.sum(N0CF+N1)>0:
                Ne_next, _, _ = self.NeCalc0(N0CF,N1,p,self.Nb,genT,kappa,0) # N0CF is used because we need to keep track of the wild effective population size. 
            else: 
                Ne_next = np.array([0])
            extra_info['Ne'] = Ne_next # Ne_wild is the Ne until you stock in the next fall.
        else: 
            N0_next = N0
            N1_next = N1
            Ne_next = np.array([0])
        juvmortality = np.exp(-124*M0-150*Mw)*((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff)
        adultmortality = np.exp(-215*M1-150*Mw)*((1 - delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall)
        extra_info['juvM'] = juvmortality
        extra_info['adultM'] = adultmortality
        extra_info['P'] = P

        # hydrological stuff
        q_next = self.flowmodel.nextflow(q) # springflow and forecast in spring
        q_next = q_next[0][0]
        #extra_info['Ne_imp'] = ((np.log(Ne_score)[0] - np.log(Ne_base)) + (np.log(Ne_next)[0] - np.log(Ne_CF)[0])) # Ne_CF is the Ne if no stocking had been done.
        #if ((np.log(Ne_score)[0] - np.log(Ne_base)) + (np.log(Ne_next)[0] - np.log(Ne_CF)[0])) >=0:
        #    print(f'negative impact on Ne smaller than positive impact on Ne: {(np.log(Ne_score)[0] - np.log(Ne_base) + np.log(Ne_next)[0] - np.log(Ne_CF)[0]):.3f}')
        #else:
        #    print(f'negative impact on Ne larger than positive impact on Ne: {(np.log(Ne_score)[0] - np.log(Ne_base) + np.log(Ne_next)[0] - np.log(Ne_CF)[0]):.3f}')
        # reward & done
        if Ne_score ==0 or Ne_base==0:
            genetic_reward = (np.log(Ne_score+1)[0] - np.log(Ne_base+1))
        else:
            genetic_reward = (np.log(Ne_score)[0] - np.log(Ne_base)) # + (np.log(Ne_next)[0] - np.log(Ne_CF)[0])
        persistence_reward = np.sum(self.c/3*((Nr>self.Nth_local).astype(int)))
        extra_info['genetic_reward'] = genetic_reward
        extra_info['persistence_reward'] = persistence_reward
        reward = persistence_reward + genetic_reward
        # np.sum(self.c/3*((Nr>self.Nth_local).astype(int))) + genetic_reward
        # self.c + genetic_reward 
        # np.sum(self.c/3*((Nr>self.popsize_1cpue).astype(int))) + genetic_reward 
        # self.c + genetic_reward  
        #np.sum(c/3*((Nr>Nth_local).astype(int))) + ((np.log(Ne_score)[0] - np.log(Ne_base)) + (np.log(Ne_next)[0] - np.log(Ne_CF)[0])) 
        # np.log(np.sum(N0_next+N1_next)) #1 + ((np.log(Ne_score)[0] - np.log(Ne_base)) + (np.log(Ne_next)[0] - np.log(Ne_CF)[0]))  
        # 100 + np.log(Ne_score)[0]   
        # self.extant +  #self.extant*(1/(1+np.exp(-0.001*(np.sum(N0+N1) - (np.log(1/0.01 - 1)/0.001) + self.Nth)))) # 0.001 = k, 0.01 = percentage of self.extant at Nth
        done = False

        # update state & obs
        if self.discset == -1:
            logN0_next = np.log(N0_next+1)
            logN1_next = np.log(N1_next+1)
            logNe_next = np.log(Ne_next+1)
            logq_next = np.array([np.log(q_next+1)])
            self.state = np.concatenate([logN0_next, logN1_next, logq_next, logNe_next])
            self.obs = np.concatenate([logN0_next, logN1_next, logq_next, logNe_next])
        else:
            N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
            N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
            q_next_idx = [self._discretize_idx(q_next, self.states['q'])]
            Ne_next_idx = [self._discretize_idx(Ne_next, self.states['Ne'])]
            self.state = np.concatenate([N0_next_idx, N1_next_idx , q_next_idx, Ne_next_idx]).astype(int)
            self.obs = np.concatenate([N0_next_idx, N1_next_idx, q_next_idx, Ne_next_idx]).astype(int)
        return self.obs, reward, done, extra_info

    def state_discretization(self, discretization_set):
        """
        input: discretization id.
        output: dictionary with states, observations, and actions.
        """

        if discretization_set == 0:
            states = {
                "N0": list(np.linspace(self.N0minmax[0], self.N0minmax[1], 31)), # population size dim:(3)
                "N1": list(np.linspace(self.N1minmax[0], self.N1minmax[1], 31)), # population size (3)
                "q": list(np.linspace(self.qminmax[0], self.qminmax[1], 11)), # spring flow in Otowi (1)
                "Ne": list(np.linspace(self.Neminmax[0], self.Neminmax[1], 11)), # Effective population size of the wild population BEFORE stocking (1)
            }

            observations = {
                "ON0": states['N0'],
                "ON1": states['N1'], 
                "Oq": states['q'],
                "ONe": states['Ne'],
            }
        elif discretization_set == -1: # continuous
            states = {
                "logN0": list(np.log(np.array(self.N0minmax)+1)), # log population size for age 0 dim:(3)
                "logN1": list(np.log(np.array(self.N1minmax)+1)), # log population size for age 1+ (3)
                "logq": list(np.log(np.array(self.qminmax)+1)), # log spring flow in Otowi (Otowi) (1)
                "logNe": list(np.log(np.array(self.Neminmax)+1)), # Effective population size of the wild population BEFORE stocking (1)
            }
            observations = {
                "OlogN0": states['logN0'],
                "OlogN1": states['logN1'],
                "Ologq": states['logq'],
                "OlogNe": states['logNe'],
            }
        # action space is 4 dimensional and each dimension is continuous between 0 and 1.
        actions = {
            "a_a": [0,1], # proportion of fish stocked in Angostura (1)
            "a_i": [0,1], # proportion of fish stocked in Isleta (1)
            "a_s": [0,1], # proportion of fish stocked in San Acacia (1)
            "a_d": [0,1], # proportion of fish discarded (1)
        }

        return {'states': states,'observations': observations,'actions': actions}

    def init_pop_sampler(self):
        """
        output:
            (if continuous) population size: list of length n_reach (using dirichlet distribution)
            OR
            (if discrete) index of population size: list of index of population size
        """
        if self.discset == -1:
            init_totpop = np.exp(np.random.uniform(size=1)[0]*(self.states['logN0'][1] - np.log(self.Nth+1)) + np.log(self.Nth+1)) # initial total population size (don't let it be lowest state)
            init_ageprop = np.random.uniform(size=1)[0] # initial age proportion
            init_prop0 = init_ageprop * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 0
            init_prop1 = (1 - init_ageprop) * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 1+
            init_pop0 = init_prop0*init_totpop
            init_pop1 = init_prop1*init_totpop
            if any(init_pop0 + init_pop1 < self.Nth):
                init_pop0 = np.maximum(init_pop0, (self.Nth + 100) - init_pop1)
            return np.log(init_pop0+1), np.log(init_pop1+1)
        else:
            init_totpop_idx = random.choices(np.arange(1,len(self.states['N0'])),k=1)[0] # initial total population size (don't let it be lowest state)
            init_ageprop = np.random.uniform(size=1)[0] # initial age proportion
            init_prop0 = init_ageprop * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 0
            init_prop1 = (1 - init_ageprop) * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 1+
            init_pop0 = init_prop0*self.states['N0'][init_totpop_idx]
            init_pop1 = init_prop1*self.states['N0'][init_totpop_idx]
            return self.pop_discretize(init_pop0,init_pop1,init_totpop_idx)
        
    def pop_discretize(self, pop0, pop1, totpop_idx):
        """
        intput:
            popX: list of population size of age 0 or 1+ for the 3 reaches.
            totpop: total population size
        output:
            index of population size for age 0 and age 1+
        Uses Knuth's “rounding to a given sum” trick and runs in O(nlogn) due to one sort.
        """
        freq0 = pop0/self.states['N0'][totpop_idx]
        freq1 = pop1/self.states['N0'][totpop_idx]
        scaledfreq0 = freq0*((totpop_idx+1) - 1)
        scaledfreq1 = freq1*((totpop_idx+1) - 1)
        scaledfreq0_flr = np.floor(scaledfreq0)
        scaledfreq1_flr = np.floor(scaledfreq1)
        margin = np.sum(scaledfreq0_flr) + np.sum(scaledfreq1_flr) - np.sum(scaledfreq0) - np.sum(scaledfreq1)
        if margin < 0:
            scaledfrac0 = scaledfreq0 - scaledfreq0_flr
            scaledfrac1 = scaledfreq1 - scaledfreq1_flr
            scaledfrac = np.concatenate((scaledfrac0,scaledfrac1))
            new_scaledfreq_flr = np.concatenate((scaledfreq0_flr,scaledfreq1_flr))
            new_scaledfreq_flr[scaledfrac.argsort()[::-1][0:np.abs(round(margin))]] += 1
            scaledfreq0_flr = list(new_scaledfreq_flr[0:len(scaledfreq0_flr)].astype(int))
            scaledfreq1_flr = list(new_scaledfreq_flr[len(scaledfreq0_flr):(len(scaledfreq0_flr)+len(scaledfreq1_flr))].astype(int))
        return scaledfreq0_flr, scaledfreq1_flr

    def _discretize(self, x, possible_states):
        # get the 2 closest values in the possible_states to x 
        # and then get the weights disproportionate to the distance to the x
        # then use those weights as probabilities to chose one of the two states.
        if x < possible_states[0]:
            return possible_states[0]
        elif x > possible_states[-1]:
            return possible_states[-1]
        else:
            lower, upper = np.array(possible_states)[np.argsort(abs(np.array(possible_states) - x))[:2]]
            weights = np.array([upper - x, x - lower])/(upper - lower)  
            return random.choices([lower, upper], weights=weights,k=1)[0]
        
    def _discretize_idx(self, x, possible_states):
        # get the 2 closest values in the possible_states to x
        # and then get the weights disproportionate to the distance to the x
        # then use those weights as probabilities to chose one of the two states.
        if x <= possible_states[0]:
            return 0
        elif x >= possible_states[-1]:
            return len(possible_states) - 1
        else:
            frac = (x - np.array(possible_states)[0])/(np.array(possible_states)[-1] - np.array(possible_states)[0])
            fracscaled = frac*(len(possible_states) - 1)
            lower_idx = np.floor(fracscaled).astype(int)
            upper_idx = np.ceil(fracscaled).astype(int)
            if lower_idx == upper_idx:
                return lower_idx
            else:
                weights = np.array([upper_idx - fracscaled, fracscaled - lower_idx])
            return random.choices([lower_idx, upper_idx], weights=weights,k=1)[0]


    def q2LC(self, q):
        """
        input:
            q: springflow in Otowi (1)
            method: 0=HMM, 1=GAM
        output:
            LC: larval carrying capacity (3)
        """
        if self.LC_prediction_method == 0: # hmm. This option is deprecated and is not updated. don't use.
            if self.discset == -1: # continuous
                # get springflow at ABQ and SA for given springflow at Otowi
                abqsf = np.minimum(np.maximum(q - self.Otowi_minus_ABQ_springflow, self.flowmodel.allowedmin[1]), self.flowmodel.allowedmax[1])
                sasf = np.minimum(np.maximum(q - self.Otowi_minus_SA_springflow, self.flowmodel.allowedmin[2]), self.flowmodel.allowedmax[2])
                # get the index of the springflow in the LC to springflow mapping table
                abqsf_idx = np.round((abqsf - self.LC_ABQ['springflow'][0])/(self.LC_ABQ['springflow'].iloc[-1] - self.LC_ABQ['springflow'][0]) * (len(self.LC_ABQ['springflow']) - 1)).astype(int)
                sasf_idx = np.round((sasf - self.LC_SA['springflow'][0])/(self.LC_SA['springflow'].iloc[-1] - self.LC_SA['springflow'][0]) * (len(self.LC_SA['springflow']) - 1)).astype(int)
                # get the potential LC values for the given springflow
                LCsamplesABQ = self.LC_ABQ.iloc[abqsf_idx]
                LCsamplesSA = self.LC_SA.iloc[sasf_idx]
            else: # discrete
                # use pre-computed index of springflow in the LC to springflow mapping table
                LCsamplesABQ = self.LC_ABQ.iloc[self.disc_sf_idxs_abq[np.array(self.state)[self.sidx['q']]]]
                LCsamplesSA = self.LC_SA.iloc[self.disc_sf_idxs_sa[np.array(self.state)[self.sidx['q']]]]
            angoLC = np.random.choice(LCsamplesABQ.values[0], size=1)
            saLC = np.random.choice(LCsamplesSA.values[0], size=1)
            L = np.array([angoLC, saLC, saLC]).T[0]
        elif self.LC_prediction_method == 1: # gam
            # calculate the error for the LC prediction
            angoLC_error = np.clip(np.random.normal(0, self.LC_ABQ['std']), -1.96*self.LC_ABQ['std'], 1.96*self.LC_ABQ['std'])
            islLC_error = np.clip(np.random.normal(0, self.LC_IS['std']), -1.96*self.LC_IS['std'], 1.96*self.LC_IS['std'])
            saLC_error = np.clip(np.random.normal(0, self.LC_SA['std']), -1.96*self.LC_SA['std'], 1.96*self.LC_SA['std'])
            if self.discset == -1:
                # get springflow at ABQ and SA for given springflow at Otowi
                abqsf = np.minimum(np.maximum(q - self.Otowi_minus_ABQ_springflow, self.flowmodel.allowedmin[1]), self.flowmodel.allowedmax[1])
                sasf = np.minimum(np.maximum(q - self.Otowi_minus_SA_springflow, self.flowmodel.allowedmin[2]), self.flowmodel.allowedmax[2])

                # predict the LC using the GAM model
                angoLC = np.maximum(self.LC_ABQ['model'].predict(abqsf) + angoLC_error, 0) # make sure LC is not negative
                islLC = np.maximum(self.LC_IS['model'].predict(sasf) + islLC_error, 0) # make sure LC is not negative
                saLC = np.maximum(self.LC_SA['model'].predict(sasf) + saLC_error,0) # make sure LC is not negative

            else:
                angoLC = np.maximum(self.LC_ABQ['model'].predict(self.ABQq[np.array(self.state)[self.sidx['q']]]) + angoLC_error, 0) # make sure LC is not negative
                islLC = np.maximum(self.LC_IS['model'].predict(self.SAq[np.array(self.state)[self.sidx['q']]]) + islLC_error, 0) # make sure LC is not negative
                saLC = np.maximum(self.LC_SA['model'].predict(self.SAq[np.array(self.state)[self.sidx['q']]]) + saLC_error, 0) # make sure LC is not negative
            L = np.array([angoLC, islLC, saLC]).T[0]
        return L, abqsf, sasf
    
    
    def production_target(self):
        """
        quantifies the amount of fish to produce in the hatchery based on the forecast
        the values here for the model is from 'Spring augmnetation planning.pdf'
        """
        qhat = np.exp(np.array(self.obs)[self.oidx['logqhat']]) if self.discset == -1 else np.array([self.observations['qhat'][self.obs[self.oidx['qhat'][0]]]])
        qhat_kaf = qhat[0]/1233480.0 # convert cubic meter to kaf
        X = np.array([1,qhat_kaf])
        V = np.array([[1.662419546,-3.284657e-03],[-0.003284657,7.883848e-06]])
        se = np.sqrt(X@V@X.T)
        fit = -0.005417*(qhat_kaf) + 2.321860
        production_target = 1/(1 + np.exp(-(fit + 1.739607*se)))*299000 # the glm model predicts the percentage of the max population capacity which was 299000 in the planning document
        aidx = self._discretize_maxstock_idx(production_target, self.actions['a'],1)
        return aidx
    
    def stocking_decision(self):
        """
        quantifies how many fish to stock in each reach based on the monitoring data
        """
        mdata = self.monitoring_sample() # monitoring catch per effort data.
        stock = np.zeros(self.n_reach)
        augment = 0
        reachlen = np.array([12333473,8748359,8527714])/100 # this is a length in 100m^2 because cpue is in 100m^2
        # figure out which reach needs augmentation and how much
        for i in range(self.n_reach):
            meanCPUE = np.mean(mdata[i])
            #if len(np.where(mdata[i] > 0)[0]) > np.floor(len(mdata[i]/2)): # within a reach, are >= 50% of the sites occupied?
            if meanCPUE > 1.0: # is the reach-wide average CPUE >= 1.0?
                augment = 0 # no augmentation needed
            else:
                augment = 1
            #else:
            #    augment = 1
            if augment == 1:
                stock[i] = (1 - meanCPUE) * reachlen[i]

        stock_prop = stock/np.sum(stock)
        return stock_prop

    def stocking_decision2(self, N0, N1):
        """
        same as stocking_decision but it assumes that the manager observes the actual population size or at least gets the estimate of the population size from Charles' IPM model. 
        The manager tries to stock enough to meet the 1.0 CPUE target in each reach. So get the corresponding average population size for 1 cpue for each reach and then stock if the 
        current population size is below that.
        """
        stock = np.zeros(self.n_reach + 1)
        Nr = N0 + N1 # popsize in each reach
        # figure out which reach needs augmentation and how much
        stock[0:self.n_reach] = np.maximum(self.popsize_1cpue[0:self.n_reach] - Nr,0)
        stock_prop = stock/self.maxcap
        if np.sum(stock_prop) >= 1:
            stock_prop = stock_prop/np.sum(stock_prop)
        else:
            stock_prop[-1] = 1 - np.sum(stock_prop[0:self.n_reach])
        return stock_prop

    def monitoring_sample(self):
        """
        simulate fall monitoring catch data from the model state
        output:
            monitoring catch data for each site (5, 6, 9 sites for angostura, isleta, and san acacia respectively)
        the values for number of sites is from 'Spring augmnetation planning.pdf'
        the average effort value is from pop_model_param_output.R, variable avgeff_f
        """

        sitenum = np.array([5, 6, 9])
        #sitenum = np.array([1000, 1000, 1000])
        avg_effort = [1257.0977, 891.8601, 1850.5951] # average area sampled at each site (square meters)
        if self.discset == -1:
            aidx = [self.sidx['logN0'][0],self.sidx['logN1'][0]]
            iidx = [self.sidx['logN0'][1],self.sidx['logN1'][1]]
            sidx = [self.sidx['logN0'][2],self.sidx['logN1'][2]]
            popsize = np.array([np.sum(np.exp(np.array(self.state)[aidx])), np.sum(np.exp(np.array(self.state)[iidx])), np.sum(np.exp(np.array(self.state)[sidx]))])
        else:
            aidx = [self.sidx['N0'][0],self.sidx['N1'][0]]
            iidx = [self.sidx['N0'][1],self.sidx['N1'][1]]
            sidx = [self.sidx['N0'][2],self.sidx['N1'][2]]
            popsize = np.array([np.sum(np.array(self.states['N0'])[np.array(self.state)[[0,3]]]), np.sum(np.array(self.states['N0'])[np.array(self.state)[[1,4]]]), np.sum(np.array(self.states['N0'])[np.array(self.state)[[2,5]]])])
        # calculate cpue from popsize
        avgp = np.mean([self.p0, self.p1])
        avgfallf = (self.fpool_f + self.frun_f)/2 # is the proportion of RGSM in the river segment exposed to sampling
        avgcatch = popsize*avgp*avgfallf*self.thetaf
        p = self.sampler / (avgcatch + self.sampler)
        cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/(self.avgeff_fr[i]+self.avgeff_fp[i])*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.

        # test code below (calculates mean cpue for each reach for different reach population sizes)
        #sitenum = np.array([1000, 1000, 1000])
        #popsizes = [10e2, 10e3, 10e4, 10e5, 10e6, 10e7]
        #mcpue = np.zeros((len(popsizes), self.n_reach))
        #for j in range(len(popsizes)):
        #    popsize = np.ones(3) * popsizes[j]
        #    avgp = np.mean([self.p0, self.p1])
        #    avgfallf = (self.fpool_f + self.frun_f)/2 # is the proportion of RGSM in the river segment exposed to sampling
        #    avgcatch = popsize*avgp*avgfallf*self.thetaf
        #    p = self.sampler / (avgcatch + self.sampler)
        #    cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/avg_effort[i]*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.
        #    mcpue[j] = np.array([np.mean(cpue[i]) for i in range(len(sitenum))], dtype=object) # mean cpue for each reach
        #print(self.fpool_f)
        #print(self.frun_f)
        return cpue#, mcpue


    def _discretize_maxstock_idx(self, x, possible_actions, lower_or_uppper):
        '''
        same as _discretize_idx but for actions, when x is in between possible_actions, it always returns the lower one.
        if lower_or_upper = 0, then it returns the lower one.
        if lower_or_upper = 1, then it returns the upper one.
        '''
        if x <= possible_actions[0]:
            return 0
        elif x >= possible_actions[-1]:
            return len(possible_actions) - 1
        else:  
            frac = (x - np.array(possible_actions)[0])/(np.array(possible_actions)[-1] - np.array(possible_actions)[0])
            fracscaled = frac*(len(possible_actions) - 1)
            if lower_or_uppper == 1:
                return np.ceil(fracscaled).astype(int)
            else:
                return np.floor(fracscaled).astype(int)
    
    def _discretize_stocking_idx(self, max_aidx, stock):
        # discretize the stocking amount to the nearest action choices
        freq = stock/self.actions['a'][max_aidx]
        stock_scaled = freq*max_aidx
        stock_scaled_flr = np.floor(stock_scaled)
        margin = np.sum(stock_scaled_flr) - np.sum(stock_scaled)
        if margin < 0:
            scaledfrac = stock_scaled - stock_scaled_flr
            stock_scaled_flr[np.argsort(scaledfrac)[::-1][0:np.abs(round(margin))]] += 1
        return list(stock_scaled_flr.astype(int))

    def NeCalc0(self, N0, N1, p, Nb, genT, kappa, season):
        """
        Calculate the effective population size (Ne). 
        intput:
            p: total number of fish stocked and survived to breeding season
            Nb: number of broodstock used for production
            N0: total population size of age 0 fish (1)
            N1: total population size of age 1+ fish (1)
            kappa: larval carrying capacity (3)
            season: 0= spring, 1=fall, in spring, only Ne = New, in fall Ne = f(New,Neh)
        output: Ne value 
        """
        if season == 0: # spring
            # calculate wild population's Ne
            totN0, totN1  = N0.sum(), N1.sum()
            effspawner    = N0 + self.beta_2*N1                      # (3,)
            alph          = self.alpha_vec[:,None,None]              # (nα,1,1)
            kappa         = self.kappa_exp.T[None,:,:]               # (1,3,nκ)
            denom         = 1 + alph*effspawner[:,None]/kappa               # (nα,3,nκ)
            if totN0 == 0:
                f1 = np.zeros((alph.shape[0],denom.shape[2]))
            else:
                f1 = (alph*N0[:,None]/denom).sum(1)/totN0        # (nα,nκ)
            if totN1 == 0:
                fa = np.zeros((alph.shape[0],denom.shape[2]))
            else:
                fa = (alph*self.beta_2*N1[:,None]/denom).sum(1)/totN1
            bvals         = (self.sj[:,None,None]*( (f1*totN0+fa*totN1)/(totN0+totN1) )).transpose(1,0,2) # (nα,nκ,ndel)
            b             = bvals[0]                                 # mean-α row
            recruitvar    = ((bvals[1:] - b)**2 * self.alphaprob[:,None,None]).sum(0)
            grate         = self.sa[:,None] + b/2
            var_dg        = self.sa[:,None]*(1-self.sa[:,None]) + b/4 + recruitvar/4
            factor = (var_dg/(grate**2) * self.kappa_prob * self.combo_delfallprob[:,None]).sum()

            New = (totN0+totN1)/factor
            New = np.array([New / genT]) # generation time adjusted wild Ne.
            return New, 0, New
        else:
            New = np.exp(self.state[self.sidx['logNe'][0]]) - 1
            # calculate hatchery population's Ne
            if np.sum(p) == 0: # if no fish are stocked, then Ne = New
                Ne = np.array([New])
                Neh = np.array([0])
            else:
                stocked_cont = np.sum(p) # stocked contribution
                total_cont = np.sum(N0) # wild contribution
                x = stocked_cont/total_cont
                #effspawner = N0 + self.beta_2*N1 # effective number of spawners
                #stocked_cont = np.sum((self.alpha*p)/(1 + self.alpha*effspawner/kappa)) # stocked fish contribution
                #total_cont =  np.sum((self.alpha*(effspawner))/(1 + self.alpha*effspawner/kappa)) # wild fish contribution
                #x = stocked_cont/(total_cont)
                #mu_k = self.fc[1]*self.irphi*np.exp(-150*np.prod(np.exp(self.lMwmu))**(1/3))
                #Neh = np.maximum(mu_k*(2*Nb - 1)/4, 0) # variance effective population size of hatchery population
                Neh = Nb * self.Ne2Nratio
                # apply Ryman-Laikre effect to calculate effective population size
                if New == 0:
                    Ne = np.array([Neh])
                else:
                    Ne = np.array([1/(x**2/Neh + (1-x)**2/(New))])
            return Ne, Neh, New

    def parameter_reset(self):
        """
        reset the parameters of the model to the initial values.
        """
        # reset 
        self.alpha = self.param_uncertainty_df['alpha'].iloc[self.paramsampleidx]
        self.beta = self.param_uncertainty_df['beta'].iloc[self.paramsampleidx]
        self.mu = np.array([self.param_uncertainty_df['mu_a'].iloc[self.paramsampleidx], 
                            self.param_uncertainty_df['mu_i'].iloc[self.paramsampleidx], 
                            self.param_uncertainty_df['mu_s'].iloc[self.paramsampleidx]])
        self.sd = self.param_uncertainty_df['sd'].iloc[self.paramsampleidx]
        self.beta_2 = self.param_uncertainty_df['beta_2'].iloc[self.paramsampleidx]
        self.tau = self.param_uncertainty_df['tau'].iloc[self.paramsampleidx]
        self.r0 = self.param_uncertainty_df['r0'].iloc[self.paramsampleidx]
        self.r1 = self.param_uncertainty_df['r1'].iloc[self.paramsampleidx]
        self.lM0mu = np.array([self.param_uncertainty_df['lM0mu_a'].iloc[self.paramsampleidx], 
                               self.param_uncertainty_df['lM0mu_i'].iloc[self.paramsampleidx], 
                               self.param_uncertainty_df['lM0mu_s'].iloc[self.paramsampleidx]])
        self.lM1mu = np.array([self.param_uncertainty_df['lM1mu_a'].iloc[self.paramsampleidx], 
                               self.param_uncertainty_df['lM1mu_i'].iloc[self.paramsampleidx], 
                               self.param_uncertainty_df['lM1mu_s'].iloc[self.paramsampleidx]])
        self.lMwmu = np.array([self.param_uncertainty_df['lMwmu_a'].iloc[self.paramsampleidx],  
                               self.param_uncertainty_df['lMwmu_i'].iloc[self.paramsampleidx], 
                               self.param_uncertainty_df['lMwmu_s'].iloc[self.paramsampleidx]])
        self.irphi = self.param_uncertainty_df['irphi'].iloc[self.paramsampleidx]
        paramset = np.concatenate(([self.alpha], [self.beta], self.mu, [self.sd], [self.beta_2], [self.tau], [self.r0], [self.r1], self.lM0mu, self.lM1mu, self.lMwmu, [self.irphi]))
        return paramset