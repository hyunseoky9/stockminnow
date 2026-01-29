import pickle
import numpy as np
import random
import pandas as pd
from whitenoise_normalized_otowi import whitenoise_normalized_otowi

class Hatchery3_3_7:
    def __init__(self,initstate,parameterization_set,discretization_set,LC_prediction_method, param_uncertainty=0, Rinfo=None):
        """
        latent process same  as 3.3.6, includes observation process with catch data now.
        """
        self.envID = 'Hatchery3.3.7'
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
        # parameter posterior distribution 
        if self.param_uncertainty:
            param_uncertainty_filename = 'uncertain_parameters_posterior_samples4POMDP.csv'
            self.param_uncertainty_df = pd.read_csv(param_uncertainty_filename)
            self.paramsampleidx = None # initiate sample idx.

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
        self.delfallp = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.deldiffp = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.delfallp[0] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall1_i'][self.parset], paramdf['delfall1_s'][self.parset]])
        self.delfallp[1] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall2_i'][self.parset], paramdf['delfall2_s'][self.parset]])
        self.deldiffp[0] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff1_i'][self.parset], paramdf['deldiff1_s'][self.parset]])
        self.deldiffp[1] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff2_i'][self.parset], paramdf['deldiff2_s'][self.parset]])
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
        self.rsz = np.mean(self.param_uncertainty_df['rsz'])
        self.bankfull = np.mean(self.param_uncertainty_df[['bankfull_a','bankfull_i','bankfull_s']], axis=0)
        self.lA0_perpool = np.mean(self.param_uncertainty_df['lA0_perpool'])
        self.AtQ_perpool = np.mean(self.param_uncertainty_df['AtQ_perpool'])
        self.alpha0_int = np.mean(self.param_uncertainty_df['alpha0_int'])
        self.alpha1_int = np.mean(self.param_uncertainty_df['alpha1_int'])
        self.alpha0_max = np.mean(self.param_uncertainty_df['alpha0_max'])
        self.alpha1_max = np.mean(self.param_uncertainty_df['alpha1_max'])
        self.lsl_width = np.mean(self.param_uncertainty_df[['lsl_width_a','lsl_width_i','lsl_width_s']], axis=0)

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
        self.dth = paramdf['dth'][self.parset] # 10 # density threshold
        self.Nth_local = self.rlen* self.dth
        self.Nth = np.sum(self.Nth_local)
        if Rinfo is None:
            self.Rinfo = {'c':1, 'no_genetics':0}
        else:
            self.Rinfo = Rinfo
        self.c = self.Rinfo['c']
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


        # start springflow simulation model and springflow-to-"Larval carrying capacity" model.
        #self.flowmodel = AR1_normalized()
        self.flowmodel = whitenoise_normalized_otowi()
        self.Otowi_minus_ABQ_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[1] # difference between Otowi and ABQ springflow
        self.Otowi_minus_SA_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[2] # difference between Otowi and San Acacia springflow
        #self.ABQ_minus_SA_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[1] # difference between ABQ and San Acacia springflow
        self.LC_prediction_method = LC_prediction_method # 0=HMM, 1=GAM

        #self.flowmodel = AR1_normalized()
        #self.flowmodel2 = whitenoise_normalized()
        #self.Otowi_minus_ABQ_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[1] # difference between Otowi and ABQ springflow
        #self.Otowi_minus_SA_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[2] # difference between Otowi and San Acacia springflow
        #self.ABQ_minus_SA_springflow = self.flowmodel2.constants[0] - self.flowmodel2.constants[1] # difference between ABQ and San Acacia springflow
        #self.LC_prediction_method = LC_prediction_method # 0=HMM, 1=GAM

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


        # monitoring related data. made from monitoring_simumlation_pre-analysis.ipynb
        with open('monitoring_sim_essentials.pkl', 'rb') as f:
            self.monitoring_sim_essentials = pickle.load(f)

        # observation related parameters
        self.avgp = np.mean([self.p0, self.p1]) # average p
        self.avgfallf = (self.fpool_f + self.frun_f)/2 # f is the proportion of RGSM in the river segment exposed to sampling
        self.popsize_1cpue = 1/(self.avgfallf*self.avgp*self.thetaf*(100/(self.avgeff_fp+self.avgeff_fr))) # average population size that corresponds to 1 cpue given average p, f (fall), and theta (fall) parameter values.
        self.monthidx_dict = {'apr':0,'may':1,'jun':2,'jul':3,'aug':4,'sep':5,'oct':6,'nov':7}
        self.monthnum_dict = {'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11}
        self.springmonitoring_months = [4,10,11] # [4]
        self.relm, self.relmetrics = self.relevant_monthsNmetrics4obs(self.Rinfo['obsvars']) # relevant months and metrics
        if 9 not in self.relm: 
            self.relm = np.concatenate((self.relm, np.array([9]))) # make sure september is included for calculating carryover effect.
        if 11 not in self.relm: 
            self.relm = np.concatenate((self.relm, np.array([11]))) # make sure november is included for calculating carryover effect.
        self.novInRelm = np.isin(11, self.relm)
        self.octInRelm = np.isin(10, self.relm)
        self.lastyrM0 = np.zeros(self.n_reach) # initiate lastyrM0
        self.lastyrM1 = np.zeros(self.n_reach) # initiate lastyrM1
        self.lastyrage0_drysurvival = np.zeros(self.n_reach) # initiate lsatyr age0 dry survival
        self.lastyrage1_drysurvival = np.zeros(self.n_reach) # initiate lsatyr age1+ dry survival
        self.reach_area_per100sqm = np.array([123334.73, 87483.59, 85277.14]) # reach area per 100 sqm for stocking calculation
        self.sample_all_months = Rinfo.get('sample_all_months', False) # if True, simulate all monitoring months every year.
        self.sample_multiplier = Rinfo.get('sample_multiplier', 1) # multiplier for number of samples in each monitoring session.
        
        # range for each variables
        self.N0minmax = [0,1e7] 
        self.N1minmax = [0,1e7] # N1 and N1 minmax are the total population minmax.
        self.Nhminmax = [0,self.maxcap] # hatchery population minmax
        self.qminmax = [self.flowmodel.flowmin[0], self.flowmodel.flowmax[0]] # springflow in Otowi (Otowi gauge) unit: m^3
        self.aminmax = [0, self.maxcap]
        self.tminmax = [0,1]
        # dimension for each variables
        self.N0_dim = (self.n_reach)
        self.N1_dim = (self.n_reach)
        self.Nh_dim = (1)
        self.q_dim = (1)
        self.t_dim = (1)
        self.statevar_dim = (self.N0_dim, self.N1_dim, self.Nh_dim, self.q_dim, self.t_dim)
        #obsservation dim for each variables.
        monitoringvardim = []
        for names in self.Rinfo['obsvars']:
            if '_r' in names:
                monitoringvardim.append(self.n_reach)
            else:
                monitoringvardim.append(1)
        monitoringvardim = np.array(monitoringvardim)
        self.obsvar_dim = np.concatenate((monitoringvardim, np.array([self.Nh_dim]), np.array([self.q_dim]), np.array([self.t_dim])))
        self.action_dim = (1,1,1,1) # 4 actions: proportion of capacity produced + stocking proportion in angostura, isleta, and san acacia.

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

    def reset(self, initstate=None,paramsampleidx =None):
        """
        Reset the environment to an initial state and return the initial observation. Always start from spring.
        """
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
        else:
            new_state.append(np.array([initstate[0]]))

        # N1 & ON1
        if initstate[1] == -1:
            # N1val = random.choices(list(np.arange(1, len(self.states['N1']))), k = self.statevar_dim[1])
            new_state.append(N1val) # don't start from
        else:
            new_state.append(np.array([initstate[1]]))

        # draw springflow and forecast for Nh, q, and qhat
        if self.discset == -1:
            qvalNforecast = self.flowmodel.nextflowNforecast()
            qval = np.array([np.log(qvalNforecast[0] + 1)])
            otowi_forecast = np.array([qvalNforecast[1][1]])
        else:
            qval = random.choices(list(np.arange(0,len(self.states['q']))), k = self.statevar_dim[3])

        # Nh
        if initstate[2] == -1:
            production = 0 # start with no fish. #self.production_target(otowi_forecast) # production target based on the initiated springflow forecast
            new_state.append(np.array([np.log(production + 1)])) 
            new_obs.append(np.array([np.log(production + 1)]))
        else:
            new_state.append(np.array([initstate[2]]))
            new_obs.append(np.array([initstate[2]]))

        # q & qhat (observed flow)
        if initstate[3] == -1:
            new_state.append(qval)
            new_obs.append(qval) # in fall springflow is already observed without error
        else:
            new_state.append(np.array([initstate[3]]))
            new_obs.append(np.array([initstate[3]]))

        # t. Always start from fall.
        new_state.append(np.array([1]))
        new_obs.append(np.array([1]))

        # sample
        if self.param_uncertainty:
            self.paramsampleidx = np.random.randint(0, self.param_uncertainty_df.shape[0]) #np.random.choice([178,3898]) #178 #np.random.randint(0, self.param_uncertainty_df.shape[0])
            paramvals = self.parameter_reset(paramsampleidx) # resample parameters from the posterior distribution

        self.state = np.concatenate(new_state)

        # monitoring data simulation
        delfall = np.concatenate(([self.delfallp[0][0]],np.random.beta(self.delfallp[0][1:],self.delfallp[1][1:])))
        deldiff = np.concatenate(([self.deldiffp[0][0]],np.random.beta(self.deldiffp[0][1:],self.deldiffp[1][1:])))
        age0_drysurvival = ((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff)
        age1_drysurvival = ((1-delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall)
        delfall = np.concatenate(([self.delfallp[0][0]],np.random.beta(self.delfallp[0][1:],self.delfallp[1][1:])))
        deldiff = np.concatenate(([self.deldiffp[0][0]],np.random.beta(self.deldiffp[0][1:],self.deldiffp[1][1:])))
        self.lastyrage0_drysurvival = ((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff)
        self.lastyrage1_drysurvival = ((1-delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall)
        M0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
        M1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
        self.lastyrM0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
        self.lastyrM1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
        

        Mw = np.exp(np.random.normal(self.lMwmu, self.lMwsd))
        self.monitor(M0, M1, Mw, age0_drysurvival, age1_drysurvival, np.exp(N0val), np.exp(N1val), None, None, initializing=True)
        obsvars = self._construct_obs(initializing=True)
        self.obs = np.concatenate((obsvars, np.concatenate(new_obs)))

        return self.obs, self.state
    


    def monitor(self, M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing=False):
        """
        Simulate monitoring data based on the current state.
        """
        if not initializing:
            if self.state[self.sidx['t']][0] == 1: # fall, set up the new monitoring dataframe and fill in spring
                # set up
                self.mdata = self.monitoringdata_setup(datatype=1)
                #self.rescue = self.monitoringdata_setup(datatype=2)
                # fill in april monitoring data if there's monitoring session.
                self.sample_catch(M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing, springmonitoring=True)
            else: # spring, fill in monitoring data
                self.sample_catch(M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing, springmonitoring=False) 
        else: # initialization, N0 and N1 is fall population, not spring population when initializing. 
            # set up
            self.mdata = self.monitoringdata_setup(datatype=1)
            #self.rescue = self.monitoringdata_setup(datatype=2)
            # fill in catch data
            self.sample_catch(M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing, springmonitoring=True)
            self.sample_catch(M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing, springmonitoring=False)
        self.monitoring_summary = self.summarize_monitoring()

    def summarize_monitoring(self):
        """
        summarize the monitoring data
        """
        m = self.mdata
        relm = np.asarray(self.relm, dtype=np.int16)
        nmonths = relm.size
        month = m["month"].astype(np.int16, copy=False)
        reach = m["reach"].astype(np.int8, copy=False)
        # map month values to 0..nmonths-1; ignore months not in relm
        mi = np.searchsorted(relm, month)
        # reach index 0..2; ignore anything outside 1..3
        ri = reach - 1
        # Allocate aggregates
        catch_sum = np.zeros((nmonths, 3), dtype=np.float32)
        effort_sum = np.zeros((nmonths, 3), dtype=np.float32)
        numsamples = np.zeros((nmonths, 3), dtype=np.int64)
        zeros_count = np.zeros((nmonths, 3), dtype=np.int64)
        maxcatch = np.zeros((nmonths, 3), dtype=np.int64)
        # Values
        catch01 = m["catch01"].astype(np.int64, copy=False)
        effort = m["effort"].astype(np.float32, copy=False)
        # Sum aggregates
        np.add.at(catch_sum, (mi, ri), catch01)
        np.add.at(effort_sum, (mi, ri), effort)
        np.add.at(numsamples, (mi, ri), 1)
        # Build DataFrames (cheap compared to groupby)
        summary = {
            "relm": relm,
            "month_to_i": {int(mon): int(i) for i, mon in enumerate(relm)},
            # core arrays
            "catch": catch_sum,
            "effort": effort_sum,
            "numsamples": numsamples,
            }
        # Fill requested metrics
        for metric in self.relmetrics:
            if metric == 'logcatch0':
                catch0 = m['catch0'].astype(np.int64, copy=False)
                catch0_sum = np.zeros((nmonths, 3), dtype=np.float32)
                np.add.at(catch0_sum, (mi, ri), catch0)
                summary['catch0'] = catch0_sum
            elif metric == 'logcatch1':
                catch1 = m['catch1'].astype(np.int64, copy=False)
                catch1_sum = np.zeros((nmonths, 3), dtype=np.float32)
                np.add.at(catch1_sum, (mi, ri), catch1)
                summary['catch1'] = catch1_sum
            elif metric == "prop0":
                # Prop0 pieces
                np.add.at(zeros_count, (mi, ri), (catch01 == 0).astype(np.int64, copy=False))
                summary[metric] = zeros_count.astype(np.int64) # save count of zeros instead of proportion. proportion calculated during observation construction to avoid divide by zero.
            elif metric == 'logmaxcatch':
                # Max catch pieces
                np.maximum.at(maxcatch, (mi, ri), catch01)
                summary['maxcatch'] = maxcatch.astype(np.float32)
            elif metric == "poolprop":
                # Pool proportion pieces
                pool_effort = np.zeros((nmonths, 3), dtype=np.float32)
                if "habitat" in m and ('poolprop' in self.relmetrics):
                    habitat = m["habitat"].astype(np.int8, copy=False)
                    pool_mask = (habitat == 1)
                    if np.any(pool_mask):
                        mi_p = mi[pool_mask]
                        ri_p = (m["reach"][pool_mask].astype(np.int8, copy=False) - 1)
                        pool_eff = m["effort"][pool_mask].astype(np.float32, copy=False)
                        np.add.at(pool_effort, (mi_p, ri_p), pool_eff)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pool_prop = pool_effort / np.where(effort_sum == 0.0, np.nan, effort_sum)
                summary[metric] = pool_prop.astype(np.float32)
            elif metric == "logecatch":
                # Expected catch
                ecatch_sum = None
                # compute only if needed; safe to compute always, it's cheap
                ecatch_sum = np.zeros((nmonths, 3), dtype=np.float32)
                eC01 = m["eC01"].astype(np.float32, copy=False)
                np.add.at(ecatch_sum, (mi, ri), eC01)
                if ecatch_sum is None:
                    ecatch_sum = np.zeros((nmonths, 3), dtype=np.float32)
                summary["ecatch"] = ecatch_sum
        return summary



    def sample_catch(self, M0, M1, Mw, age0_drysurvival, age1_drysurvival, N0, N1, novN0, novN1, initializing, springmonitoring):
        '''
        sample catch data for monitoring and rescue data.
        if springmonitoring = True, sample for april monitoring data only, else sample for may through november monitoring data only + rescue data.
        '''
        month = self.mdata["month"]
        reach = self.mdata["reach"]
        julian = self.mdata["julian"]
        discharge = self.mdata["discharge"]
        habitat = self.mdata["habitat"]
        effort = self.mdata["effort"]
        
        # monitoring data
        if springmonitoring:
            lastoctidx = np.isin(month, self.springmonitoring_months[1])
            lastnovidx = np.isin(month, self.springmonitoring_months[2])
            aprilidx   = np.isin(month, self.springmonitoring_months[0])
            idx = lastoctidx | lastnovidx | aprilidx
        else:
            idx = ~np.isin(month, self.springmonitoring_months) # non_april_idx
        if(np.sum(idx)>0):
            ## get detectability parameters 
            ### theta
            reachlenyridx = np.random.choice(np.arange(0,self.monitoring_sim_essentials['reach_wetlen'].shape[2]), 1)[0]
            theta = 0.2/(self.monitoring_sim_essentials['reach_wetlen'][reach[idx]-1,julian[idx]-1,reachlenyridx])
            ### f
            xi_p0 = self.alpha0_int*discharge[idx]/(1 + self.alpha0_int*discharge[idx]/self.alpha0_max)
            xi_p1 = self.alpha1_int*discharge[idx]/(1 + self.alpha1_int*discharge[idx]/self.alpha1_max)
            xi_m0 = xi_p0 * ((habitat[idx] - 1)* (1/xi_p0 - 1) + 1) # this makes pool samples have xi_m0 = 1 and run samples have xi_m0 = xi_p0
            xi_m1 = xi_p1 * ((habitat[idx] - 1)* (1/xi_p1 - 1) + 1)
            # Explicit logit: log(p / (1 - p)), avoiding scipy.special.logit
            A0_perpool = np.log(self.lA0_perpool / (1 - self.lA0_perpool))
            sl_width = np.exp(self.lsl_width[reach[idx]-1])
            areavar1 = np.exp(A0_perpool + self.AtQ_perpool*discharge[idx])
            areavar2 = 200*sl_width*discharge[idx]/(1+sl_width*discharge[idx]/self.bankfull[reach[idx]-1])
            totpool = areavar1*areavar2
            totrun = (1-areavar1)*areavar2
            f0 = np.minimum(effort[idx]*xi_m0/(xi_p0*totpool + totrun),1)
            f1 = np.minimum(effort[idx]*xi_m1/(xi_p1*totpool + totrun),1)
            ## get mortality rates on the sampling day
            if initializing: # when initializing, N0 and N1 are fall population sizes, need to backtrack to spring population sizes
                N0 = np.minimum(N0/(np.exp(-M0 * 124) * age0_drysurvival), np.exp(self.states['logN0'][1])) # age 0 population in spring
                N1 = np.minimum(N1/(np.exp(-M1 * 215) * age1_drysurvival), np.exp(self.states['logN1'][1])) # age 1 population in spring
                if springmonitoring:
                    lastyrnovpop = np.minimum(N1/np.exp(-Mw * 150), np.exp(self.states['logN1'][1])) # age 1 population in last year november
                    novN0 = lastyrnovpop*0.9 # age0 in last year november, assume 90% are age 0
                    novN1 = lastyrnovpop*0.1 # age1 in last year november, assume 10% are age 1

            ## get expected cathch
            if springmonitoring == False:
                natural_m0 = np.exp(-M0[reach[idx]-1] * (julian[idx] - 91)) # age 0 natural mortality adjustment from fall to sampling day
                natural_m1 = np.exp(-M1[reach[idx]-1] * julian[idx]) # age 0 natural mortality adjustment from fall to sampling day
                N0_adjusted = N0[reach[idx]-1] * natural_m0 * age0_drysurvival[reach[idx]-1]
                eC0 = N0_adjusted * theta *  self.p0 * f0
                C0 = np.random.negative_binomial(self.sz, self.sz/(self.sz + eC0))
                mayjune_idx = (month[idx]==5) | (month[idx]==6)
                C0[mayjune_idx] = 0 # no age 0 in may and june
                N1_adjusted = N1[reach[idx]-1] * natural_m1 * age1_drysurvival[reach[idx]-1]
                eC1 = N1_adjusted * theta *  self.p1 * f1
                C1 = np.random.negative_binomial(self.sz, self.sz/(self.sz + eC1))
            else: 
                N0lastnov_adjusted = novN0[reach[lastnovidx]-1]
                N1lastnov_adjusted = novN1[reach[lastnovidx]-1]
                natural_m1 = np.exp(-M1[reach[aprilidx]-1] * julian[aprilidx]) # age 0 natural mortality adjustment from fall to sampling day
                N0lastoct_adjusted = novN0[reach[lastoctidx]-1]/(np.exp(-self.lastyrM0[reach[lastoctidx]-1] * (124 - (julian[lastoctidx]-91))) * self.lastyrage0_drysurvival[reach[lastoctidx]-1])
                N1lastoct_adjusted = novN1[reach[lastoctidx]-1]/(np.exp(-self.lastyrM1[reach[lastoctidx]-1] * (215 - julian[lastoctidx])) * self.lastyrage1_drysurvival[reach[lastoctidx]-1])
                N0april_adjusted = np.zeros(np.sum(aprilidx))
                N1april_adjusted = N1[reach[aprilidx]-1] * natural_m1 * age1_drysurvival[reach[aprilidx]-1]
                N0_adjusted = np.concatenate((N0lastoct_adjusted, N0lastnov_adjusted, N0april_adjusted))
                N1_adjusted = np.concatenate((N1lastoct_adjusted,N1lastnov_adjusted, N1april_adjusted))
                eC0 = N0_adjusted * theta *  self.p0 * f0
                eC1 = N1_adjusted * theta *  self.p1 * f1
                C0 = np.random.negative_binomial(self.sz, self.sz/(self.sz + eC0))
                C1 = np.random.negative_binomial(self.sz, self.sz/(self.sz + eC1))
                
            self.mdata["catch0"][idx] = C0
            self.mdata["catch1"][idx] = C1
            self.mdata["catch01"][idx] = C0 + C1

            self.mdata["eC0"][idx] = eC0
            self.mdata["eC1"][idx] = eC1
            self.mdata["eC01"][idx] = eC0 + eC1

            self.mdata["f0"][idx] = f0
            self.mdata["f1"][idx] = f1
            self.mdata["theta"][idx] = theta
            self.mdata["p0"][idx] = self.p0

            # N0_adjusted / N1_adjusted differ by branch; ensure they exist
            self.mdata["N0_adjusted"][idx] = N0_adjusted
            self.mdata["N1_adjusted"][idx] = N1_adjusted
            self.mdata["N_adjusted"][idx] = N0_adjusted + N1_adjusted
            self.mdata["N0"][idx] = N0[reach[idx] - 1]
            self.mdata["N1"][idx] = N1[reach[idx] - 1]
            self.mdata["N"][idx] = self.mdata["N0"][idx] + self.mdata["N1"][idx]

            if springmonitoring:
                # mimic your special-case: last Oct/Nov catches not distinguishable by age
                self.mdata["catch0"][lastnovidx] = 0
                self.mdata["catch1"][lastnovidx] = 0
                self.mdata["catch0"][lastoctidx] = 0
                self.mdata["catch1"][lastoctidx] = 0

        # rescue catch data
        # WARNING: issues with rescue data simualtion. 
        # rescuetheta will be 0 for most samples, because most days in the data have no drying. This makes the simulated sample rescue catch always 0.
        # This zero-inflated data looks very different from actual rescue data. 
        # Another issue is that when abundance is really large and rescuetheta is not zero, the eR0 values can be really large,
        # leading to negative binomial sampling errors due to numerical instability in the function.
        # not doing rescue data for now (12/3/2025)
        if 1==0:
            if springmonitoring:
                idx = np.isin(self.mdata['month'].values, self.springmonitoring_months) # april_idx
            else:
                idx = ~np.isin(self.mdata['month'].values, self.springmonitoring_months) # non_april_idx
            if np.sum(idx)>0:
                ## rescue theta is just the proportion of river dried that day. 
                ### find a year where sum of delfall absolute difference accross 3 reaches is samllest
                mindiffyridx = np.argmin(np.sum(np.abs(self.delfall[:,None] - self.monitoring_sim_essentials['total_dryingprop']),axis=0))
                ### use the normalize drying proportion of the chosen year to the simulated delfall.
                obstotal_drying_denom = self.monitoring_sim_essentials['total_dryingprop'][self.rescue['reach'].values[idx]-1,mindiffyridx]
                obstotal_drying_denom[obstotal_drying_denom==0] = 1 # to avoid zero division
                rescuetheta = (self.monitoring_sim_essentials['poportion_dried'][self.rescue['reach'].values[idx]-1,self.rescue['julian'].values[idx]-1, mindiffyridx]*
                    (self.delfall[self.rescue['reach'].values[idx]-1]/obstotal_drying_denom))
                ## calculate abundance on the rescue days.
                if initializing:
                    N0 = N0/(np.exp(-M0 * 124) * age0_drysurvival) # age 0 population in spring
                    N1 = N1/(np.exp(-M1 * 215) * age1_drysurvival) # age 1 population in spring
                natural_m0 = np.exp(-M0[self.rescue['reach'].values[idx]-1] * (self.rescue['julian'].values[idx] - 91)) # age 0 natural mortality adjustment from fall to sampling day
                natural_m1 = np.exp(-M1[self.rescue['reach'].values[idx]-1] * self.rescue['julian'].values[idx]) # age 0 natural mortality adjustment from fall to sampling day
                N0_adjusted = N0[self.rescue['reach'].values[idx]-1] * natural_m0 * age0_drysurvival[self.rescue['reach'].values[idx]-1]
                N1_adjusted = N1[self.rescue['reach'].values[idx]-1] * natural_m1 * age1_drysurvival[self.rescue['reach'].values[idx]-1]
                ## calculate expected rescue catch
                eR0 = self.r0 * rescuetheta * N0_adjusted * (1-self.tau)
                eR1 = self.r1 * rescuetheta * N1_adjusted * (1-self.tau)
                ## sample rescue catch
                R0 = np.random.negative_binomial(self.rsz, self.rsz/(self.rsz + eR0))
                R1 = np.random.negative_binomial(self.rsz, self.rsz/(self.rsz + eR1))
                self.rescue.loc[idx, 'catch0'] = R0
                self.rescue.loc[idx, 'catch1'] = R1
                self.rescue.loc[idx, 'catch01'] = R0 + R1
    
    def monitoringdata_setup(self, datatype):
        """
        Set up the monitoring data dataframe structure.
        datatype: 1 = regular monitoring data, 2 = rescue data
        """
        if datatype == 1:
            # set up monitoring data dataframe
            # choose months of monitoring.
            if self.sample_all_months:
                numsesh = 6
            else:
                numsesh = np.random.choice(self.monitoring_sim_essentials['num_sesh_dist'] -1,1) # -1 is added because September session is assumed to always happen.
            seshmonth = np.sort(np.random.choice(np.array([4,5,6,7,8,10]),numsesh, replace=False, p=self.monitoring_sim_essentials['samplenum_prop_bymonth']))
            octexist = 1 if self.octInRelm and seshmonth[-1] == 10 else 0 # october session exists.
            seshmonth = np.concatenate((np.array([10]), seshmonth[0:-1], np.array([9]))) if octexist else np.concatenate((seshmonth, np.array([9])))
            seshmonth = seshmonth[np.isin(seshmonth,self.relm)] # get rid of months that are not in the observation variables. 
            numsesh = len(seshmonth)
            # number of sample pairs for each sesh (one for run and one for pool)
            halfnumsamples = np.random.choice(self.monitoring_sim_essentials['halfnum_sample_dist']*self.sample_multiplier, numsesh-1) #(np.ones(numsesh)*22).astype(int)
            halfnumsamples = np.concatenate((halfnumsamples, np.random.choice(self.monitoring_sim_essentials['halfnum_sample_dist_sep']*self.sample_multiplier, 1)))
            # julians for each samples
            
            julians = []
            sample_reaches = []
            discharge = []
            # for each session, generate julians and sample reaches. November gets added first because the fall timestep is october and the first data collected after fall step is november data.
            # add november session. 
            if self.novInRelm:
                halfnumsamples_nov = np.random.choice(self.monitoring_sim_essentials['halfnum_sample_dist_nov']*self.sample_multiplier, 1) #np.array([22]) # np.random.choice(self.monitoring_sim_essentials['halfnum_sample_dist_nov'], 1) #np.array([23])
                julians.append(np.ones(halfnumsamples_nov)*215) # all november samples on day 215 (Nov 1)
                # all 3 reaches should have at least one sample in november
                sample_reach_nov = np.random.multinomial(halfnumsamples_nov[0] - 3, self.monitoring_sim_essentials['sampledist_reach_nov']) #np.array([5,10,7]); -3 because each reach will get 1 sample to ensure each reach has at least one sample.
                sample_reach_nov = sample_reach_nov + 1 # add 1 back to each reach
                sample_reach_nov = np.concatenate(([1]*sample_reach_nov[0], [2]*sample_reach_nov[1], [3]*sample_reach_nov[2]))
                sample_reaches.append(sample_reach_nov)
                discharge_sample = np.random.choice(self.monitoring_sim_essentials['discharge_data_nov'], halfnumsamples_nov[0])
                discharge.append(discharge_sample)
            else:
                halfnumsamples_nov = np.array([0])
            # add other months
            for i in range(len(halfnumsamples)):
                sumsampledaymorethan30 = 1
                while sumsampledaymorethan30:
                    sampledaydiff = np.random.choice(self.monitoring_sim_essentials['days_since_last_sample_dist'], halfnumsamples[i]-1)
                    if np.sum(sampledaydiff) < 30:
                        sumsampledaymorethan30 = 0
                startdate = self.monitoring_sim_essentials['start_of_month_julian'][seshmonth[i]-4]
                julian = np.concatenate((np.array([startdate]), startdate + np.cumsum(sampledaydiff)))
                julian[np.where(julian - julian[0] > 29)[0]] = julian[0] + 29 # cap at 30 days
                if seshmonth[i] == 9: # all 3 reaches should have at least one sample in september
                    sample_reach = np.random.multinomial(halfnumsamples[i] - 3, self.monitoring_sim_essentials['sampledist_reach']) #np.array([5,8,9]) ; -3 because each reach will get 1 sample to ensure each reach has at least one sample.
                    sample_reach = sample_reach + 1 # add 1 back to each reach
                else:
                    sample_reach = np.random.multinomial(halfnumsamples[i], self.monitoring_sim_essentials['sampledist_reach']) #np.array([5,8,9]) 
                sample_reach = np.concatenate(([1]*sample_reach[0], [2]*sample_reach[1], [3]*sample_reach[2]))
                unique_julians, inverse_indices = np.unique(julian, return_inverse=True)
                discharge_unique = np.random.choice(self.monitoring_sim_essentials['discharge_data'][seshmonth[i]-4], len(unique_julians))
                discharge_sample = discharge_unique[inverse_indices]
                if i == 0 and octexist == 1:
                    julians.insert(0,julian)
                    sample_reaches.insert(0,sample_reach)
                    discharge.insert(0,discharge_sample)
                else:
                    julians.append(julian)
                    sample_reaches.append(sample_reach)
                    discharge.append(discharge_sample)

            # multiply the data by 2 for run and pool samples
            # flow
            discharge = np.concatenate(discharge)
            discharge = np.repeat(discharge,2) # for run and pool
            # repeat other variables for each habitat. 
            if octexist == 1:
                repeated_months = np.concatenate((np.ones(halfnumsamples[0]*2)*10,np.ones(halfnumsamples_nov[0]*2)*11,np.repeat(seshmonth[1:], halfnumsamples[1:]*2))).astype(int)
            else:
                repeated_months = np.concatenate((np.ones(halfnumsamples_nov[0]*2)*11,np.repeat(seshmonth, halfnumsamples*2))).astype(int)
            julians_flat = np.concatenate(julians)
            julians_flat = np.repeat(julians_flat,2)
            julians_flat = julians_flat.astype(int)
            reaches_flat = np.repeat(np.concatenate(sample_reaches),2).astype(int)
            habitat = np.tile(np.array([1,2]), np.sum(halfnumsamples)+halfnumsamples_nov[0]) # 1 = pool, 2 = run
            effort = np.zeros(np.sum(halfnumsamples)*2 + halfnumsamples_nov[0]*2)
            # 0 and even numbers get pool effort, odd numbers get run effort
            N = repeated_months.shape[0]
            effort[1:len(effort):2] = np.random.choice(self.monitoring_sim_essentials['effort_run_dist'], len(effort)//2)
            effort[0:len(effort):2] = np.random.choice(self.monitoring_sim_essentials['effort_pool_dist'], len(effort)//2)

            zf = np.zeros(N, dtype=np.float32)
            zi = np.zeros(N, dtype=np.int64)
            # create dataframe
            mdata = {'month':repeated_months,
                            'julian':julians_flat,
                            'reach':reaches_flat,
                            'discharge':discharge,
                            'habitat':habitat,
                            'effort':effort,

                            # dynamic (filled in sample_catch)
                            "catch0": zi.copy(),
                            "catch1": zi.copy(),
                            "catch01": zi.copy(),
                            "eC0": zf.copy(),
                            "eC1": zf.copy(),
                            "eC01": zf.copy(),
                            "f0": zf.copy(),
                            "f1": zf.copy(),
                            "theta": zf.copy(),
                            "p0": zf.copy(),
                            "N0_adjusted": zf.copy(),
                            "N1_adjusted": zf.copy(),
                            "N_adjusted": zf.copy(),
                            "N0": zf.copy(),
                            "N1": zf.copy(),
                            "N": zf.copy()                            
                        }
            #with pd.option_context('display.max_rows', None):
            #    print(self.mdata)
            return mdata
        else: #datatype == 2
            random_indices = np.random.randint(0, len(self.monitoring_sim_essentials['samplesize_permonth_rescue']), size=self.monitoring_sim_essentials['samplesize_permonth_rescue'].shape[1])
            sampled_values = self.monitoring_sim_essentials['samplesize_permonth_rescue'].values[random_indices, np.arange(self.monitoring_sim_essentials['samplesize_permonth_rescue'].shape[1])]
            if(np.all(sampled_values == 0)): # if all sampled values are zero, sample a positive value from July
                julypositive_values = self.monitoring_sim_essentials['samplesize_permonth_rescue'].iloc[:,3] 
                julypositive_values = julypositive_values[julypositive_values > 0]
                sampled_values[3] = np.random.choice(julypositive_values,1)[0]
            months = []
            reaches = []
            julians = []
            sample_sizes_nozeros = []
            for i in range(len(sampled_values)):
                if sampled_values[i] > 0:
                    sample_sizes_nozeros.append(sampled_values[i])
                    reach = []
                    month = i+4
                    months.append(month) # rescue months from april (4) to october (10)
                    startdate = np.random.choice(np.arange(0,30-sampled_values[i]+1),1) + self.monitoring_sim_essentials['start_of_month_julian'][i]
                    julian = np.arange(startdate, startdate + sampled_values[i])
                    julians.append(julian)
                    firstsamplereach = np.random.binomial(1, self.monitoring_sim_essentials['reach_proportions_rescue'][1]) + 2
                    for j in range(sampled_values[i]):
                        if j == 0:
                            reach.append(firstsamplereach)
                        else:
                            if reach[-1] == 2:
                                reach.append(np.random.binomial(1, self.monitoring_sim_essentials['reach_transition_matrix_rescue'][0][1]) + 2)
                            else:
                                reach.append(np.random.binomial(1, self.monitoring_sim_essentials['reach_transition_matrix_rescue'][1][1]) + 2)
                    reaches.append(reach)
            months = np.repeat(np.array(months), sample_sizes_nozeros)
            reaches = np.concatenate(reaches)
            julians = np.concatenate(julians)
            self.rescue = pd.DataFrame({'month':months,
                                'julian':julians,
                                'catch0':np.zeros(len(reaches)),
                                'catch1':np.zeros(len(reaches)),
                                'reach':reaches,
                                })
            #with pd.option_context('display.max_rows', None):
            #    print(self.rescue)

            return self.rescue

    def step(self, a, current_strategy = 0, referencemonth='sep'):
        """
        Take an action and return the next state, reward, done flag, and extra information.
        a is a vector of 4, where the first is the proportion of production and the rest three are stocking proportions in angostura, isleta, and san acacia.
        current_strategy=1 takes the production action and stocking action based on the currently carried out heuristic stocking strategy.
        current_strategy=1 is used for using current production and stocking strategy. It ignores a input.
        referencemonth is used to determine the stocking decision when current_strategy=1. It can include other months this way: 'jul+aug+sep'. Of course, that observation have to have been initialized.
        """
        extra_info = {}
        if self.discset == -1:
            N0 = np.exp(np.array(self.state)[self.sidx['logN0']]) - 1
            N1 = np.exp(np.array(self.state)[self.sidx['logN1']]) - 1
            Nh = np.exp(np.array(self.state)[self.sidx["logNh"]]) - 1
            q = np.exp(np.array(self.state)[self.sidx["logq"]]) - 1
            t = self.state[self.sidx['t']][0]
        totN0 = np.sum(N0)
        totN1 = np.sum(N1)
        Nr = N0 + N1 # population size in each reach
        if current_strategy == 1: 
            if t == 0: # spring
                prod_target = self.production_target(np.exp(self.obs[self.oidx['Ologq']])-1)
                prod_target = np.array([min(prod_target, self.maxcap)/self.maxcap])
                a = np.concatenate((prod_target,[0,0,0])) # production target based on the observed springflow forecast
            else:
                a = np.concatenate(([0],self.stocking_decision3_2(referencemonth))) # stocking decision based on current strategy when assuming that you can observe the population size through IPM.
            extra_info['current_strat_action'] = a
        a_prod = a[0] # production action
        a_stock = a[1:]
        a_stock = a_stock[0:self.n_reach]
        totpop = totN0 + totN1
        # switch season 
        t_next = np.array([(t + 1) % 2]) # 0 is spring and 1 is fall
        if np.sum(Nr) > 0:
            if t == 1: # fall
                # demographic stuff (stocking and winter survival)
                Mw = np.exp(self.lMwmu) #np.exp(np.random.normal(self.lMwmu, self.lMwsd))
                stockedNsurvived = np.round(a_stock*Nh)*self.irphi
                #print(f'x in fall {np.sum(stockedNsurvived)/np.sum((N0+stockedNsurvived))}')
                N0og = N0.copy()
                N1og = N1.copy()
                N0CF = N0.copy()*np.exp(-150*Mw) # counterfactual N0, if no stocking had been done. Also equivalent to wild-origin spawners.
                extra_info['Mw'] = Mw
                N0 = N0 + stockedNsurvived # stocking san acacia (t=3) in the fall
                N0 = np.minimum(N0*np.exp(-150*Mw),np.ones(self.n_reach)*self.N0minmax[1]) # stocking san acacia (t=3) in the fall
                N1 = N1*np.exp(-150*Mw)

                # local extinction if the population goes below the local threshold
                for r in range(self.n_reach):
                    if N0[r] + N1[r] < self.Nth_local[r]:
                        N0[r], N1[r], N0CF[r] = 0, 0, 0
                Nr_spring = N0 + N1
                # reward & done
                persistence_reward = np.sum(self.c/3*((Nr_spring>self.Nth_local).astype(int)))
                extra_info['persistence_reward'] = persistence_reward
                reward = persistence_reward

                # hydrological stuff
                qNforecast = self.flowmodel.nextflowNforecast() # springflow and forecast in spring
                #q_next = q_next[0][0]
                q_next = qNforecast[0]
                qhat_next = qNforecast[1][1] # otowi springflow forecast
                N0_next = N0
                N1_next = N1
                Nh_next = np.array([0])
                # monitoring dataframe set up & April monitoring
                ## sample delfall, deldiff, and M0 and M1 for the spring step now cuz you need it for april monitoring.
                self.delfall = np.concatenate(([self.delfallp[0][0]],np.random.beta(self.delfallp[0][1:],self.delfallp[1][1:])))
                self.deldiff = np.concatenate(([self.deldiffp[0][0]],np.random.beta(self.deldiffp[0][1:],self.deldiffp[1][1:])))
                age0_drysurvival = ((1 - self.delfall) + self.tau*self.delfall*self.deldiff + (1 - self.tau)*self.r0*self.phidiff)
                age1_drysurvival = ((1 - self.delfall) + self.tau*self.delfall + (1 - self.tau)*self.r1*self.phifall)
                self.M0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
                self.M1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
                self.monitor(self.M0, self.M1, Mw, age0_drysurvival, age1_drysurvival, np.zeros(self.n_reach), N0_next+N1_next, N0og, N1og,initializing=False)
                
            else: # spring
                # demographic stuff (reproductin and summer survival)
                L, abqsf, sasf = self.q2LC(q)   
                #extra_info['L'] = L
                #extra_info['abqsf'] = abqsf
                #extra_info['sasf'] = sasf
                natural_capacity = np.random.normal(self.mu, self.sd)
                kappa = np.exp(self.beta*(L - self.Lmean) + natural_capacity)
                #extra_info['natural_capacity'] = natural_capacity
                #extra_info['kappa'] = kappa

                effspawner = N0 + self.beta_2*N1 # effective number of spawners
                P1 = (self.alpha*N0)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 1 fish that newly became adults
                P2 = (self.alpha*self.beta_2*N1)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 2+ fish
                P = (self.alpha*effspawner)/(1 + self.alpha*effspawner/kappa)
                if np.sum(P)>0:
                    age0_drysurvival = ((1 - self.delfall) + self.tau*self.delfall*self.deldiff + (1 - self.tau)*self.r0*self.phidiff)
                    age1_drysurvival = ((1-self.delfall) + self.tau*self.delfall + (1 - self.tau)*self.r1*self.phifall)
                    N0_next = np.minimum(P*np.exp(-124*self.M0)*age0_drysurvival,np.ones(self.n_reach)*self.N0minmax[1])
                    N1_next = np.minimum((N0+N1)*np.exp(-215*self.M1)*age1_drysurvival,np.ones(self.n_reach)*self.N1minmax[1])
                else:
                    N0_next = N0
                    N1_next = N1
                    #juvmortality = np.exp(-124*M0-150*Mw)*((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff)
                    #adultmortality = np.exp(-215*M1-150*Mw)*((1 - delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall)
                    #extra_info['juvM'] = juvmortality
                    #extra_info['adultM'] = adultmortality
                    #extra_info['P'] = P

                # monitoring data simulation.
                self.monitor(self.M0, self.M1, None, age0_drysurvival, age1_drysurvival, P, N0+N1, None, None, initializing=False)
                self.lastyrM0 = self.M0.copy()
                self.lastyrM1 = self.M1.copy()
                self.lastyrage0_drysurvival = age0_drysurvival
                self.lastyrage1_drysurvival = age1_drysurvival
                # hatchery production for next year
                Nh_next = np.array([np.round(a_prod * self.maxcap)]) # production target based on the springflow forecast
                # flow stuff
                q_next = q[0] # no change
                qhat_next = q[0] # real flow is observed without error in the fall
                # reward
                reward = -Nh_next[0]/self.maxcap # economic cost (financial & labor cost)
            done = False
            # update state & obs
            if self.discset == -1:
                logN0_next = np.log(N0_next+1)
                logN1_next = np.log(N1_next+1)
                logNh_next = np.log(Nh_next+1)
                logq_next = np.array([np.log(q_next+1)])
                logqhat_next = np.array([np.log(qhat_next+1)])
                self.state = np.concatenate([logN0_next, logN1_next, logNh_next, logq_next, t_next])
                obsvars = self._construct_obs()
                self.obs = np.concatenate([obsvars, logNh_next, logqhat_next, t_next])
        else: #extinct. terminate
            reward = 0
            extra_info['persistence_reward'] = 0
            done = True
        return self.obs, reward, done, extra_info

    def _construct_obs(self, initializing=False):
        """
        construct observation variables from the monitoring data (mdata & rescue).
        List of observation variables are in Rinfo['observation_vars'].
        """
        season = self.state[self.sidx['t']][0]
        ms = self.monitoring_summary
        m2i = ms["month_to_i"]
        obsvars = []

        for varname in self.Rinfo['obsvars']:
            varnamesplit = varname.split('_')

            reachspecific = True if varnamesplit[1] == 'r' else False # whether the variable is reach specific
            multimonth = True if '+' in varnamesplit[-1] else False # whether the variable is multi-month metric
            varmonth = varnamesplit[-1] if multimonth==False else varnamesplit[-1].split('+') # month or list of months for the variable
            varmetric = varnamesplit[0] # metric of the variable
            monthnum = np.array([self.monthnum_dict[m] for m in varmonth], dtype=np.int16) if multimonth else int(self.monthnum_dict[varmonth])


            if not initializing:
                if season == 0: # spring season. skip updating non april/nov monitoring vars
                    if np.all(~np.isin(monthnum,self.springmonitoring_months)): # put the current variable values.
                        obsvar = self.obs[self.oidx[varname]]
                        obsvars.append(obsvar)
                        continue
                else: # fall season. skip updating april/nov monitoring vars
                    if np.all(np.isin(monthnum,self.springmonitoring_months)): # put the current variable values.
                        obsvar = self.obs[self.oidx[varname]]
                        obsvars.append(obsvar)
                        continue
            if multimonth: # check if any month has no samples fot all or some reaches
                month_is = [m2i.get(int(m), None) for m in monthnum]
                month_is = np.array(month_is, dtype=np.int64)
                nums = ms["numsamples"][month_is].sum(axis=0)
                nosample = (nums == 0)
                nomonth = (np.sum(~nosample) == 0)
            else:
                mi = m2i.get(int(monthnum), None)
                nums = ms["numsamples"][mi]
                nosample = (nums == 0)
                nomonth = (np.sum(~nosample) == 0)

            NAval = -1
            ## 12/19 start fixing from here.
            # --- Compute obsvar by metric (same behavior as old .loc version) ---
            if varmetric == 'logcatch':
                if not nomonth:
                    if multimonth:
                        obsvar = np.log(ms["catch"][month_is].sum(axis=0) + 1.0)
                    else:
                        obsvar = np.log(ms["catch"][mi] + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else: # no data for all months
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'effort':
                if not nomonth:
                    if multimonth:
                        obsvar = ms["effort"][month_is].sum(axis=0).astype(np.float32, copy=False)
                    else:
                        obsvar = ms["effort"][mi].astype(np.float32, copy=False).copy()
                    obsvar[nosample] = 0
                else:
                    obsvar = (0 * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([0], dtype=np.float32)
            elif varmetric == 'logcpue':
                if not nomonth:
                    if multimonth:
                        csum = ms["catch"][month_is].sum(axis=0)
                        esum = ms["effort"][month_is].sum(axis=0) + 1e-5
                        obsvar = np.log((csum / esum) * 100.0 + 1.0)
                    else:
                        e = ms["effort"][mi] + 1e-5
                        obsvar = np.log((ms["catch"][mi] / e) * 100.0 + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logcpue0':
                if not nomonth:
                    if multimonth:
                        csum = ms["catch0"][month_is].sum(axis=0)
                        esum = ms["effort"][month_is].sum(axis=0) + 1e-5
                        obsvar = np.log((csum / esum) * 100.0 + 1.0)
                    else:
                        e = ms["effort"][mi] + 1e-5
                        obsvar = np.log((ms["catch0"][mi] / e) * 100.0 + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logcpue1':
                if not nomonth:
                    if multimonth:
                        csum = ms["catch1"][month_is].sum(axis=0)
                        esum = ms["effort"][month_is].sum(axis=0) + 1e-5
                        obsvar = np.log((csum / esum) * 100.0 + 1.0)
                    else:
                        e = ms["effort"][mi] + 1e-5
                        obsvar = np.log((ms["catch1"][mi] / e) * 100.0 + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logcatch0':
                if not nomonth:
                    if multimonth:
                        obsvar = np.log(ms["catch0"][month_is].sum(axis=0) + 1.0)
                    else:
                        obsvar = np.log(ms["catch0"][mi] + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logcatch1':
                if not nomonth:
                    if multimonth:
                        obsvar = np.log(ms["catch1"][month_is].sum(axis=0) + 1.0)
                    else:
                        obsvar = np.log(ms["catch1"][mi] + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logmaxcatch':
                if not nomonth:
                    if multimonth:
                        obsvar = np.log(ms["maxcatch"][month_is].max(axis=0) + 1.0)
                    else:
                        obsvar = np.log(ms["maxcatch"][mi] + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'numsamples':
                if multimonth:
                    obsvar = nums.astype(np.float32, copy=False)
                else:
                    obsvar = nums.astype(np.float32, copy=False).copy()
                # (no special missing handling in your old version)
            elif varmetric == 'prop0':
                # You used -1 as sentinel for "no samples"
                if not nomonth:
                    if "prop0" not in ms:
                        raise KeyError("monitoring_summary is missing 'prop0' (needed by _construct_obs).")
                    if multimonth:
                        # weighted avg by numsamples
                        w = ms["prop0"][month_is]
                        obsvar = w.sum(axis=0) / (nums + 1e-5)
                    else:
                        obsvar = ms["prop0"][mi]/nums[mi].copy()
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'poolprop':
                # You used -1 as sentinel for "no samples"
                if not nomonth:
                    if "poolprop" not in ms:
                        raise KeyError("monitoring_summary is missing 'poolprop' (needed by _construct_obs).")
                    if multimonth:
                        w = ms["poolprop"][month_is] * ms["numsamples"][month_is]
                        obsvar = w.sum(axis=0) / (nums + 1e-5)
                    else:
                        obsvar = ms["poolprop"][mi].copy()
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            elif varmetric == 'logecatch':
                if not nomonth:
                    if "ecatch" not in ms:
                        raise KeyError("monitoring_summary is missing 'ecatch' (needed by _construct_obs).")
                    if multimonth:
                        obsvar = np.log(ms["ecatch"][month_is].sum(axis=0) + 1.0)
                    else:
                        obsvar = np.log(ms["ecatch"][mi] + 1.0)
                    obsvar = obsvar.astype(np.float32, copy=False)
                    obsvar[nosample] = NAval
                else:
                    obsvar = (NAval * np.ones(self.n_reach, dtype=np.float32)) if reachspecific else np.array([NAval], dtype=np.float32)
            obsvars.append(obsvar)
        return np.concatenate(obsvars)
    
    def relevant_monthsNmetrics4obs(self,obsvarnames):
        """
        input: litst of observation variable names
        output: months where observation variables are collected & metrics collected in observation variables
        """
        months = self.monthidx_dict.keys()
        relmonths = []
        relmetrics = []
        for name in obsvarnames:
            # split by _
            namesplit = name.split('_')
            varmonths = namesplit[-1].split('+') if '+' in namesplit[-1] else [namesplit[-1]]
            relmetrics.append(namesplit[0])
            for month in months:
                if month in varmonths:
                    relmonths.append(self.monthidx_dict[month]+4)
                    
        return np.unique(np.array(relmonths)), np.unique(np.array(relmetrics))

    def state_discretization(self, discretization_set):
        """
        input: discretization id.
        output: dictionary with states, observations, and actions.
        """

        if discretization_set == 0: # deprecated
            states = {
                "N0": list(np.linspace(self.N0minmax[0], self.N0minmax[1], 31)), # population size dim:(3)
                "N1": list(np.linspace(self.N1minmax[0], self.N1minmax[1], 31)), # population size (3)
                "q": list(np.linspace(self.qminmax[0], self.qminmax[1], 11)), # spring flow in Otowi (1) unit: m^3
            }

            observations = {
                "ON0": states['N0'],
                "ON1": states['N1'], 
                "Oq": states['q'],
            }
        elif discretization_set == -1: # continuous
            states = {
                "logN0": list(np.log(np.array(self.N0minmax)+1)), # log population size for age 0 dim:(3)
                "logN1": list(np.log(np.array(self.N1minmax)+1)), # log population size for age 1+ (3)
                "logNh": list(np.log(np.array(self.Nhminmax)+1)), # log spring flow in Otowi (Otowi) (1)
                "logq": list(np.log(np.array(self.qminmax)+1)), # log spring flow in Otowi (Otowi) (1)
                "t": [0,1], # season 0=spring, 1=fall (1)
            }
            self.Rinfo['obsvars']
            observations = {}
            for varname in self.Rinfo['obsvars']:
                observations[varname] = [0,1] # there's really no need to put down actual ranges for these variables...
            observations['OlogNh'] = states['logNh']
            observations['Ologq'] = states['logq']
            observations['Ot'] = states['t']
        # action space is 4 dimensional and each dimension is continuous between 0 and 1.
        actions = {
            "a_p": [0,1], # proportion of maximum capacity to produce (1)
            "a_a": [0,1], # proportion of fish stocked in Angostura (1)
            "a_i": [0,1], # proportion of fish stocked in Isleta (1)
            "a_s": [0,1]  # proportion of fish stocked in San Acacia (1)
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
                # get springflow at ABQ and SA for given springflow at Otowi
                abqsf = np.minimum(np.maximum(q - self.Otowi_minus_ABQ_springflow, self.flowmodel.allowedmin[1]), self.flowmodel.allowedmax[1])
                sasf = np.minimum(np.maximum(q - self.Otowi_minus_SA_springflow, self.flowmodel.allowedmin[2]), self.flowmodel.allowedmax[2])
                #abqsf = np.minimum(np.maximum(self.states['q'], self.flowmodel.allowedmin[0]), self.flowmodel.allowedmax[0])
                #sasf = np.minimum(np.maximum(self.states['q'] - self.ABQ_minus_SA_springflow, self.flowmodel.allowedmin[1]), self.flowmodel.allowedmax[1])

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
                #abqsf = np.minimum(np.maximum(q, self.flowmodel.allowedmin[0]), self.flowmodel.allowedmax[0])
                #sasf = np.minimum(np.maximum(q - self.ABQ_minus_SA_springflow, self.flowmodel.allowedmin[1]), self.flowmodel.allowedmax[1])

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
    
    
    def production_target(self,forecast=None):
        """
        quantifies the amount of fish to produce in the hatchery based on the forecast
        the values here for the model is from 'Spring augmnetation planning.pdf'
        """
        if forecast is not None:
            qhat = forecast
        else:
            qhat = np.exp(np.array(self.obs)[self.oidx['Ologq']]) if self.discset == -1 else np.array([self.observations['qhat'][self.obs[self.oidx['qhat'][0]]]])
        qhat_kaf = qhat[0]/1233480.0 # convert cubic meter to kaf
        X = np.array([1,qhat_kaf])
        V = np.array([[1.662419546,-3.284657e-03],[-0.003284657,7.883848e-06]])
        se = np.sqrt(X@V@X.T)
        fit = -0.005417*(qhat_kaf) + 2.321860
        production_target = 1/(1 + np.exp(-(fit + 1.739607*se)))*299000 # the glm model predicts the percentage of the max population capacity which was 299000 in the planning document
        #aidx = self._discretize_maxstock_idx(production_target, self.actions['a'],1)
        return production_target
    
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

    def stocking_decision3(self, N0, N1):
        """
        same as stocking_decision but it assumes that the manager observes the actual population size or at least gets the estimate of the population size from Charles' IPM model. 
        The manager tries to stock enough to meet the 1.0 CPUE target in each reach. 
        If there are any reaches below expected 1.0 CPUE equivalent popsize
            only stock in those reaches to put them back to 1.0 CPUE equivalent popsize. Any leftover are distributed are stocked among reaches with inverse proportional weights.
            inverse proportional weights: if each reach's population is 100,200,300, then the weights are inverse population / sum of inverse population = (1/100, 1/200, 1/300)/sum(1/100, 1/200, 1/300)
        """
        stock = np.zeros(self.n_reach)
        Nr = N0 + N1 # popsize in each reach
        # figure out which reach needs augmentation because they're below 1.0CPUE and how much
        stock[0:self.n_reach] = np.maximum(self.popsize_1cpue[0:self.n_reach] - Nr,0)
        Nh = np.round(np.exp(np.array(self.state)[self.sidx['logNh'][0]]) - 1)
        if np.sum(stock) <= Nh:
            if Nh == 0:
                return (1/(Nr + 1))/np.sum(1/(Nr + 1)) # add 1 to avoid division by zero
            else:
                leftover = Nh - np.sum(stock)
                inverseweight = (1/(Nr + 1))/np.sum(1/(Nr + 1)) # add 1 to avoid division by zero
                leftoverstock = inverseweight*leftover
                stock[0:self.n_reach] = stock[0:self.n_reach] + leftoverstock
        stock_prop = stock/np.sum(stock)
        return stock_prop

    def stocking_decision3_2(self,referencemonth='sep'):
        """
        same stocking distribution logic as stocking_decision3, but based on actual cpue instead of population size.
        """
        stock = np.zeros(self.n_reach)
        cpue = np.exp(self.obs[self.oidx[f'logcpue_r_{referencemonth}']]) - 1 # assumes that logcpue_r_oct variable is constructed
        cpuediff = np.maximum(1 - cpue, 0)
        stock[0:self.n_reach] = cpuediff * self.reach_area_per100sqm # from Archdeacon et al. 2022 augmentation plan
        Nh = np.round(np.exp(np.array(self.state)[self.sidx['logNh'][0]]) - 1)
        if np.sum(stock) <= Nh:
            if Nh == 0:
                return (1/(cpue + 1))/np.sum(1/(cpue+1)) # add 1 to avoid division by zero
            else:
                leftover = Nh - np.sum(stock)
                inverseweight = (1/(cpue + 1))/np.sum(1/(cpue + 1)) # add 1 to avoid division by zero
                leftoverstock = inverseweight*leftover
                stock[0:self.n_reach] = stock[0:self.n_reach] + leftoverstock
        stock_prop = stock/np.sum(stock)
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


    def parameter_reset(self, paramsampleidx=None):
        """
        reset the parameters of the model to the initial values.
        """
        if paramsampleidx is not None:
            self.paramsampleidx = paramsampleidx
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
        sdsample = self.param_uncertainty_df['sd_lM'].iloc[self.paramsampleidx]
        self.lM0sd = np.array([sdsample, sdsample, sdsample])
        self.lM1sd = np.array([sdsample, sdsample, sdsample])
        self.irphi = self.param_uncertainty_df['irphi'].iloc[self.paramsampleidx]
        self.dth = self.param_uncertainty_df['dth'].iloc[self.paramsampleidx] # extinction threshold
        ## rest sampling parameters
        self.bankfull = np.array([self.param_uncertainty_df['bankfull_a'].iloc[self.paramsampleidx],
                                self.param_uncertainty_df['bankfull_i'].iloc[self.paramsampleidx],
                                self.param_uncertainty_df['bankfull_s'].iloc[self.paramsampleidx]])
        self.lA0_perpool = self.param_uncertainty_df['lA0_perpool'].iloc[self.paramsampleidx]
        self.AtQ_perpool = self.param_uncertainty_df['AtQ_perpool'].iloc[self.paramsampleidx]
        self.alpha0_int = self.param_uncertainty_df['alpha0_int'].iloc[self.paramsampleidx]
        self.alpha1_int = self.param_uncertainty_df['alpha1_int'].iloc[self.paramsampleidx]
        self.alpha0_max = self.param_uncertainty_df['alpha0_max'].iloc[self.paramsampleidx]
        self.alpha1_max = self.param_uncertainty_df['alpha1_max'].iloc[self.paramsampleidx]
        self.sz = self.param_uncertainty_df['sz'].iloc[self.paramsampleidx]
        self.rsz = self.param_uncertainty_df['rsz'].iloc[self.paramsampleidx]
        self.p0 = self.param_uncertainty_df['p0'].iloc[self.paramsampleidx]
        self.p1 = self.param_uncertainty_df['p1'].iloc[self.paramsampleidx]
        self.lsl_width = np.array([self.param_uncertainty_df['lsl_width_a'].iloc[self.paramsampleidx],
                                    self.param_uncertainty_df['lsl_width_i'].iloc[self.paramsampleidx],
                                    self.param_uncertainty_df['lsl_width_s'].iloc[self.paramsampleidx]])
        
        

        paramset = np.concatenate(([self.alpha], [self.beta], self.mu, [self.sd], [self.beta_2],
                                    [self.tau], [self.r0], [self.r1], self.lM0mu, self.lM1mu,
                                      self.lMwmu, [self.irphi], [self.dth], self.bankfull,
                                      [self.lA0_perpool], [self.AtQ_perpool], [self.alpha0_int],
                                      [self.alpha1_int], [self.alpha0_max], [self.alpha1_max], [self.sz], [self.p0], [self.p1], self.lsl_width))
        return paramset