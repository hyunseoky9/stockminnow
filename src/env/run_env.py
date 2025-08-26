import numpy as np
from scipy.stats import hmean
import os
import sys
import pickle
from datetime import datetime
from performance_analysis_tools import *
from Hatchery3_2_4 import Hatchery3_2_4 


# take username input from sys
username = sys.argv[1]
# get date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

env = Hatchery3_2_4(None,1,-1,1,1)
numsteps = 0
stock_prop = [] # proportion of fish stocked   
ltNe = 0 # long-term Ne (harmonic mean)
local_dips = np.zeros(env.n_reach)
actiondist = np.zeros(len(env.actionspace_dim))
actiondistcount = 0
maxstep = 50

rewards = []
total_reward = 0
states = []
env.reset()

Nes = [] # Ne after stocking (Ne_score)
local_dips_epi = []
extinction_period = -1
# standardize
done = False
t = 0
while done == False:
    #stateid = env._flatten(env.state)
    rellogpop, logpop = relative_logpopsize(env)
    local_dips_epi.append(logpop < np.log(env.rlen * env.dth))
    states.append(env.state)
    N0 = np.exp(np.array(env.state)[env.sidx['logN0']]) - 1
    N1 = np.exp(np.array(env.state)[env.sidx['logN1']]) - 1
    q = np.exp(np.array(env.state)[env.sidx["logq"]]) - 1
    Ne = np.exp(np.array(env.state)[env.sidx["logNe"]]) - 1
    print(f't: {t}\nlog(population size): {np.round(np.log(N0+N1+1),2)}\nlog(Ne_wild): {np.round(np.log(Ne+1),2)}\nlog(q): {np.round(np.log(q+1),2)}')
    action_input = input('Stocking action:')
    action = np.array(eval(action_input))
    _, reward, done, info = env.step(action)
    print(f'reward: {reward} (persistence reward: {info["persistence_reward"]}, genetic reward: {info["genetic_reward"]})')
    #print(f'reward: {reward}')
    if done == False:
        Nes.append(info['Ne_score'])
    stock_prop.append(action)
    actiondist += action
    actiondistcount += 1
    rewards.append(reward)
    total_reward += reward
    t += 1
    if t >= maxstep:
        done = True
extinct_period = t
local_dips_epi = np.array(local_dips_epi)
local_dips = np.sum(local_dips_epi,axis=0)/local_dips_epi.shape[0]
if len(Nes) > 0:
    ltNe = hmean(np.concatenate(Nes))
numsteps = extinct_period
actiondist = actiondist / actiondistcount

outputdict = {'rewards': rewards, 'maxstep': maxstep, 'numsteps': numsteps, 
                'stock_prop': actiondist, 'stock_prop_each_timestep': stock_prop,
                'states':states, 'ltNe': ltNe, 'local_dips': local_dips}
# save outputdict as a pickle
with open(f"../../data/performance_output_{env.envID}_{username}_{current_time}.pkl", "wb") as f:
    pickle.dump(outputdict, f)

# return rewards_receptacle, numsteps, actiondist, ltNe, Nvals, stock_prop, Newvals,local_dips, logpopsizes