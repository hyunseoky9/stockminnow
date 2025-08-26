from scipy.special import logit, expit
import pickle
import numpy as np
import random
class AR1_normalized:
    '''
    Same as AR1 but uses the model that's trained on normalized-logit transformed data
    when doing minmax normalizing, the min is 90% of the minimum springflow and the max is 110% of the maximum springflow. 
    I did this because otherwise the minimum springflow would be 0 and the maximum springflow would be 1, which would give -inf and inf when logit transforming.

    model that simulates spring flow
    (total volume from march thru july) at 3 gauges 
    (otowi, abq, san acacia) using AR1 model fitted
    to the 1997-2024 gauge data.
    '''
    def __init__(self):
        # gauge order
        self.order = {'otowi': 0, 'abq': 1, 'sanacacia': 2}

        # load parameters
        with open('ar1_normalized_params.pkl', 'rb') as handle:
            self.params = pickle.load(handle)
        self.otowiNconstant = self.params['otowiNconst']
        self.constants = np.array([self.params['otowiconst'], self.params['abqconst'], self.params['sanacaciaconst']])
        self.phi_1 = self.params['phi_1']
        self.sigma2 = self.params['sigma2']
        self.flowmin = np.array(self.params['flowmin'])
        self.flowmax = np.array(self.params['flowmax'])
        self.allowedmin = self.flowmin * 0.9
        self.allowedmax = self.flowmax * 1.1
        # load bias parameters
        with open('nrcs_forecast_bias_stats.pkl', 'rb') as handle:
            self.bias_params = pickle.load(handle)
        self.bias_mean = self.bias_params['mean_bias']
        self.bias_std = self.bias_params['std_bias']
        self.bias_95interval = [self.bias_mean - 1.96 * self.bias_std, self.bias_mean + 1.96 * self.bias_std]

    def nextflow(self, otowispringflow):
        # input: unnormalized otowi springflow last year
        # output: this year springflow at 3 otowi, abq, san acacia + nrcs springflow forecast at otowi

        # minmax normalize and then logit transform otowi springflow

        Notowispringflow = (otowispringflow - self.allowedmin[0])/ (self.allowedmax[0] - self.allowedmin[0])  # minmax normalize otowi springflow
        NLotowispringflow = logit(Notowispringflow)  # logit transform otowi springflow
        NLOtowival = self.otowiNconstant + (self.phi_1 * NLotowispringflow + np.random.normal(0, np.sqrt(self.sigma2)))
        Otowival =  self.allowedmin[0] + (self.allowedmax[0] - self.allowedmin[0])*expit(NLOtowival)
        Otowival = np.maximum(np.minimum(Otowival, self.allowedmax[0]), self.allowedmin[0])
        #Otowival = self.constants[0] + (self.phi_1 * otowispringflow + np.random.normal(0, np.sqrt(self.sigma2)))
        ABQval = Otowival + self.constants[1] - self.constants[0]
        SAval = Otowival + self.constants[2] - self.constants[0]
        vals = np.array([Otowival, ABQval, SAval])
        return vals

    def nextflowNforecast(self, otowispringflow):
        # input: unnormalized otowi springflow last year
        # output: this year springflow at 3 otowi, abq, san acacia + nrcs springflow forecast at otowi

        # minmax normalize and then logit transform otowi springflow

        Notowispringflow = (otowispringflow - self.allowedmin[0])/ (self.allowedmax[0] - self.allowedmin[0])  # minmax normalize otowi springflow
        NLotowispringflow = logit(Notowispringflow)  # logit transform otowi springflow
        NLOtowival = self.otowiNconstant + (self.phi_1 * NLotowispringflow + np.random.normal(0, np.sqrt(self.sigma2)))
        Otowival =  self.allowedmin[0] + (self.allowedmax[0] - self.allowedmin[0])*expit(NLOtowival)
        Otowival = np.maximum(np.minimum(Otowival, self.allowedmax[0]), self.allowedmin[0])
        #Otowival = self.constants[0] + (self.phi_1 * otowispringflow + np.random.normal(0, np.sqrt(self.sigma2)))
        ABQval = Otowival + self.constants[1] - self.constants[0]
        SAval = Otowival + self.constants[2] - self.constants[0]
        vals = np.array([Otowival, ABQval, SAval])

        bias = np.clip(
            np.random.normal(loc=self.bias_mean, scale=self.bias_std, size=1),
            self.bias_95interval[0],
            self.bias_95interval[1]
        )
        
        forecast = vals[self.order['otowi']] + bias
        forecast = np.maximum(forecast, self.flowmin - self.bias_95interval[1])
        forecast = np.minimum(forecast, self.flowmax + self.bias_95interval[1])

        # make sure forecast is not negative
        forecast = np.max(forecast, 0)
        return vals, forecast #, np.maximum(NLOtowival,0)
