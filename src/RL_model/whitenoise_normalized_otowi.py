from scipy.special import logit, expit
import pickle
import numpy as np
import random
class whitenoise_normalized_otowi:
    """
    Same as whitenoise_normalized but outputs otowi flow and forecast for nextflow and nextflowNforecast methods.
    """
    def __init__(self):

        # load parameters
        with open('white_noise_params_ABQ.pkl', 'rb') as handle:
            abqdf = pickle.load(handle)
        with open('white_noise_params_San Acacia.pkl', 'rb') as handle:
            sanacaciadf = pickle.load(handle)
        with open('white_noise_params_Otowi.pkl', 'rb') as handle:
            otowidf = pickle.load(handle)
        self.flowmin = np.array([otowidf['flowmin'], abqdf['flowmin'], sanacaciadf['flowmin']])
        self.flowmax = np.array([otowidf['flowmax'], abqdf['flowmax'], sanacaciadf['flowmax']])
        self.allowedmin = self.flowmin * 0.9
        self.allowedmax = self.flowmax * 1.1
        self.otowiconstant = otowidf['constant']
        self.constants = np.array([otowidf['constant'], abqdf['constant'], sanacaciadf['constant']])
        self.abqparams = {
            'mu': abqdf['mu'],
            'std': abqdf['std'],
            'flowmin': abqdf['flowmin'],
            'flowmax': abqdf['flowmax'],
            'allowed_flowmin': abqdf['flowmin']*0.9,
            'allowed_flowmax': abqdf['flowmax']*1.1
        }
        self.saparams = {
            'mu': sanacaciadf['mu'],
            'std': sanacaciadf['std'],
            'flowmin': sanacaciadf['flowmin'],
            'flowmax': sanacaciadf['flowmax'],
            'allowed_flowmin': sanacaciadf['flowmin']*0.9,
            'allowed_flowmax': sanacaciadf['flowmax']*1.1
        }
        self.otowiparams = {
            'mu': otowidf['mu'],
            'std': otowidf['std'],
            'flowmin': otowidf['flowmin'],
            'flowmax': otowidf['flowmax'],
            'allowed_flowmin': otowidf['flowmin']*0.9,
            'allowed_flowmax': otowidf['flowmax']*1.1
        }
        # load forecast bias parameters
        with open('nrcs_forecast_bias_stats.pkl', 'rb') as handle:
            self.bias_params = pickle.load(handle)
        self.bias_mean = self.bias_params['mean_bias']
        self.bias_std = self.bias_params['std_bias']
        self.bias_95interval = [self.bias_mean - 1.96 * self.bias_std, self.bias_mean + 1.96 * self.bias_std]



    def nextflow(self,q):
        '''
        Generate the next time step of flow for both gages.
        * it's actually wrong to model two gages separately. but the RL environment (Hatchery3.x) only takes the first argument
        '''
        otowi_initial = np.random.normal(self.otowiparams['mu'], self.otowiparams['std'])
        # back transform
        otowi_flow = expit(otowi_initial) * (self.otowiparams['allowed_flowmax'] - self.otowiparams['allowed_flowmin']) + self.otowiparams['allowed_flowmin']
        return np.array([otowi_flow, -1]) # -1 is just a placeholder
    
    def nextflowNforecast(self):
        '''Generate the next time step of flow for both gages and apply forecast bias.
        '''
        otowi_initial = np.random.normal(self.otowiparams['mu'], self.otowiparams['std'])
        # back transform
        otowi_flow = expit(otowi_initial) * (self.otowiparams['allowed_flowmax'] - self.otowiparams['allowed_flowmin']) + self.otowiparams['allowed_flowmin']

        # otowi forecast with bias
        bias = np.clip(
            np.random.normal(loc=self.bias_mean, scale=self.bias_std, size=1),
            self.bias_95interval[0],
            self.bias_95interval[1]
        )
        #bias = np.array([0]) # activate this line to turn off forecast bias
        # apply forecast bias
        otowi_forecast = otowi_flow + bias # bias 0 for now
        # make sure forecast is within range
        otowi_forecast = np.maximum(otowi_forecast, self.otowiparams['flowmin'] + self.bias_95interval[0])
        otowi_forecast = np.minimum(otowi_forecast, self.otowiparams['flowmax'] + self.bias_95interval[1])
        otowi_forecast = np.maximum(otowi_forecast, 0)
        forecast = np.array([np.array([-1]), otowi_forecast]).T[0] # -1 is just a placeholder
        return [otowi_flow, forecast]
