from call_paramset import call_env
import numpy as np
import pickle

def var_means_stds(env):
    # calculate or load standardization means and stds for obsrevation variables for env's with monitoring-based observations (e.g. 3.3.7)
    filename = f'obsvar_mean_{env.envID}.pkl'
    with open(filename, 'rb') as f:
        df = pickle.load(f) # pre-calculated means and stds for Hatchery3.3.7
    vars = env.Rinfo['obsvars']
    outputlen = np.sum((np.char.find(np.array(vars), '_r') != -1) * 2 + 1)
    means = np.zeros(outputlen)
    stds = np.zeros(outputlen)

    # find the means and stds from a pickled dictionary. 
    existingvaridx = []
    notexistingvaridx = []
    existingvarrealidx = []
    notexistingvarrealidx = []
    realidxcounter = 0
    for i, var in enumerate(vars):
        # if the var is not in the dictionary, calculate it through simulation
        if var in df.keys():
            existingvaridx.append(i)
            existingvarrealidx.append(np.array([0,1,2])+realidxcounter if '_r' in var else np.array([0])+realidxcounter)
        else:
            notexistingvaridx.append(i)
            notexistingvarrealidx.append(np.array([0,1,2])+realidxcounter if '_r' in var else np.array([0])+realidxcounter)
        realidxcounter += 3 if '_r' in var else 1
    if len(notexistingvaridx) > 0:
        print(f'Calculating mean and std for variable: {np.array(vars)[notexistingvaridx]}')
        # need to add september cpue observation for current strategy
        Rinfo = env.Rinfo.copy()
        obsvar_plus = env.Rinfo['obsvars'] + ['logcpue_r_sep'] if 'logcpue_r_sep' in env.Rinfo['obsvars'] else env.Rinfo['obsvars']
        Rinfo['obsvars'] = obsvar_plus
        config = {'init': None, 'paramset': 1, 'discretization': -1, 'LC': 1, 'uncertainty': 1, 'Rinfo': Rinfo}
        param = {'envid': env.envID, 'envconfig': str(config)}
        newenv = call_env(param)
        notexistingoidxpre = [newenv.oidx[vars[i]] for i in notexistingvaridx]
        notexistingoidx = np.concatenate(np.array(notexistingoidxpre))
        obs_vals = []
        num_episodes = 3000 #3000
        tmax = 50
        for epi in range(num_episodes):
            obs = newenv.reset()
            done = False
            t = 0
            while not done:
                obs, reward, done, info = newenv.step(np.array([0,0,0,0]),1) # enact current strategy
                obs_vals.append(obs[notexistingoidx])
                t+=1
                if t >= tmax:
                    done = True
            if (epi+1) % 1000 == 0:
                print(f'Finished {epi+1} episodes for calculating means and stds.')
        mean_val = np.zeros(len(notexistingoidx))
        std_val = np.zeros(len(notexistingoidx))
        obs_vals = np.array(obs_vals)
        # calculate mean and std for each variable
        counter = 0
        for i in range(len(notexistingvaridx)):
            if ('effort' in vars[notexistingvaridx[i]]) or ('numsamples' in vars[notexistingvaridx[i]]):
                for j in range(len(notexistingoidxpre[i])):
                    #valid_mask = obs_vals[:,counter] > 0 
                    maxval = np.max(np.log1p(obs_vals[:,counter]))
                    # set mean to 0 and std to maxval so when standardizing, the value is between 0 and 1 instead of doing z-score standardization like all the other variables.
                    mean_val[counter] = 0 
                    std_val[counter] = maxval
                    counter += 1
            else:
                for j in range(len(notexistingoidxpre[i])):
                    valid_mask = obs_vals[:,counter] >= -900
                    mean_val[counter] = obs_vals[:,counter][valid_mask].mean()
                    std_val[counter] = obs_vals[:,counter][valid_mask].std()
                    counter += 1

    for idx,i in enumerate(existingvaridx):
        means[existingvarrealidx[idx]] = df[vars[i]][0]
        stds[existingvarrealidx[idx]] = df[vars[i]][1]
    if len(notexistingvaridx) > 0:
        means[np.concatenate(notexistingvarrealidx)] = mean_val
        stds[np.concatenate(notexistingvarrealidx)] = std_val
    # save the updated means and stds back to the pickle file
    for i in range(len(notexistingvaridx)):
        df[vars[notexistingvaridx[i]]] = (means[notexistingvarrealidx[i]], stds[notexistingvarrealidx[i]])
    with open(filename, 'wb') as f:
        pickle.dump(df, f)

    return means, stds
