import pandas as pd
from itertools import product
def call_paramset(filename,id):
    # basic processing
    data = pd.read_csv(filename, header=None)
    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(0)   
    paramdf = data.iloc[id].to_dict()
    paramdflist = []
    # if there are semicolons, separate them and make all combinations of paramdf
    keys = paramdf.keys()
    tunekeys = []
    tunekeyvals = []
    # get which keys have multiple values to try out for tuning
    for key in keys:
        if key=='notes' or key=='score':
            continue
        if ';' in paramdf[key]:
            tunekeys.append(key)
            vals = paramdf[key].split(';')
            tunekeyvals.append(vals)
    # Generate all combinations of tuning parameters
    for combination in product(*tunekeyvals):
        temp_paramdf = paramdf.copy()
        for i, key in enumerate(tunekeys):
            temp_paramdf[key] = combination[i]
        paramdflist.append(temp_paramdf)
    return paramdflist


from Hatchery3_3_7 import Hatchery3_3_7
def call_env(param):
    config = eval(param['envconfig'])
    if param['envid'] == 'Env1.0':
        return Env1_0(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env1.1':
        return Env1_1(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env1.2':
        return Env1_2(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.0':
        return Env2_0(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.1':
        return Env2_1(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.2':
        return Env2_2(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.3':
        return Env2_3(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.4':
        return Env2_4(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.5':
        return Env2_5(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.6':
        return Env2_6(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.7':
        return Env2_7(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Hatchery3.0':
        return Hatchery3_0(config['init'], config['paramset'], config['discretization'],config['LC'])
    elif param['envid'] == 'Hatchery3.1':
        return Hatchery3_1(config['init'], config['paramset'], config['discretization'],config['LC'])
    elif param['envid'] == 'Hatchery3.2':
        return Hatchery3_2(config['init'], config['paramset'], config['discretization'],config['LC'])
    elif param['envid'] == 'Hatchery3.2.2':
        return Hatchery3_2_2(config['init'], config['paramset'], config['discretization'],config['LC'],config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.2.3':
        return Hatchery3_2_3(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.2.4':
        return Hatchery3_2_4(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.2.5':
        return Hatchery3_2_5(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.2.6':
        return Hatchery3_2_6(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.2.7':
        return Hatchery3_2_7(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.1':
        return Hatchery3_3_1(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.2':
        return Hatchery3_3_2(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.2.2':
        return Hatchery3_3_2_2(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.3':
        return Hatchery3_3_3(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.4':
        return Hatchery3_3_4(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.5':
        return Hatchery3_3_5(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.6':
        return Hatchery3_3_6(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.3.7':
        return Hatchery3_3_7(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Hatchery3.4.1':
        return Hatchery3_4_1(config['init'], config['paramset'], config['discretization'],config['LC'], config['uncertainty'], config['Rinfo'])
    elif param['envid'] == 'Tiger':
        return Tiger()
    else:
        raise ValueError("Unknown environment ID")
        