
import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.linear_model import BayesianRidge
from deconvolution import deconvolve_series
from scipy import stats
from scipy.stats import norm, poisson, nbinom



class Params:
    """
    Set training sets, the study period, shedding length(m) and additional params to plot
    """
    def __init__(self):
        self.training_Woodland = [['2022-01-13', '2022-02-02'], ['2021-12-11', '2021-12-30']]
        self.training_Davis = [['2022-01-09', '2022-02-05'], ['2021-12-12', '2022-01-08']]
        self.training_UCDavis = [['2022-01-12', '2022-02-03'], ['2021-12-12', '2022-01-03']]
        self.training_date = {'Davis': self.training_Davis, 'Davis (sludge)': self.training_Davis, 'Woodland': self.training_Woodland, 'UCDavis':self.training_UCDavis}
        self.params = {'Davis': {'test_set': 0, 'm': 7, 'init_date': '2021-12-01', 'end_date': '2022-03-29'},
                       'Woodland': {'test_set': 0, 'm': 7, 'init_date': '2021-12-01', 'end_date': '2022-03-31'},
                       'UCDavis': {'test_set': 0, 'm': 7, 'init_date': '2021-12-01', 'end_date': '2022-03-31'}}
        self.ylim ={'Davis':[0.0035, 400], 'Woodland':[0.003,280], 'UCDavis':[0.0022,400]}
        self.colors = ['green', 'red']


class mcmc_conv:
    def __init__(self, city, init_training, end_training, m):
        self.data_ww = pd.read_csv('data/data_ww_cases.csv')  # read data
        self.city = city
        self.init_training = init_training
        self.end_training = end_training
        self.size_window = 10  # for smothing ww data
        self.m = m             # shedding load distribution lenght
        self.city_data, self.cases, self.data = self.read_data()
        self.n = len(self.data) - self.m - 1
        self.a = 1
        self.n_c = len(self.cases)
        self.time = np.arange(self.m) #[::-1]
        self.n_sample = 3000 # predictive size
        #self.likelihood = 'Normal'
        #self.likelihood = 'Poisson'
        #self.likelihood = 'logNormal'
        self.likelihood = 'Bin_Neg_fp'  # Observational model
        #self.trim_aver = {'Davis':False, 'Woodland':True, 'UCDavis':True}
        self.hiper_pars = {'Bin Neg':1, 'Bin_Neg_fp':0.5, 'Poisson':1, 'logNormal':1} #0.005
        self.omega = 2.0      # overdispersion parameter to the negative binomial
        self.theta_bn = 0.05  # overdispersion parameter to the negative binomial
        self.shift = 3        # shift for negative binomial

        # hyper parameters
        # Sigma ~ gamma(ss_a, ss_b)
        self.ss_a = 1
        self.ss_b = self.ss_a / self.hiper_pars[self.likelihood]

        # M ~ gamma(M_a, M_b)
        self.M_a = 2
        self.M_b = self.M_a / 0.0001    # Ma/1e-5 (sludge)

        # b ~ gamma(b_a, b_b)
        self.b_a = 2
        self.b_b = self.b_a / 2     # scale parameter b = 2
        self.min_period = 3         # min period to apply a smooth function

        self.Alpha = {'Bin Neg':np.array([self.M_a, self.b_a, self.ss_a]),'Bin_Neg_fp':np.array([self.M_a, self.b_a]),
                      'Poisson':np.array([self.M_a, self.b_a]), 'Normal':np.array([self.M_a, self.b_a, self.ss_a]),
                      'logNormal':np.array([self.M_a, self.b_a, self.ss_a])}
        self.Beta = {'Bin Neg':np.array([self.M_b, self.b_b, self.ss_b]),'Bin_Neg_fp':np.array([self.M_b, self.b_b]),
                      'Poisson':np.array([self.M_b, self.b_b]),'Normal':np.array([self.M_b, self.b_b, self.ss_b]),
                      'logNormal':np.array([self.M_b, self.b_b, self.ss_b])}

        self.alpha = self.Alpha[self.likelihood]
        self.beta = self.Beta[self.likelihood]
        self.d = len(self.alpha)  # number of parameters to estimate

        self.burnin = 5000        # burnin size
        self.thini = 5            # integration autocorrelation time

    def trim_fun(self, x):
        """ Remove max and min in an array
            :param x:(array)
            :return x1: (array)
        """
        x = x.dropna()
        x1 = x.sort_values().ravel()
        return np.mean(x1[1:-1])

    def read_data(self):
        city_data = self.data_ww[self.data_ww['City'] == self.city]
        city_data = city_data.reset_index()
        city_data['positives_average'] = np.copy(city_data['positives'].rolling(window=7, center=False, min_periods=3).mean())

        city_data['NormalizedConc_trimmed'] = city_data['NormalizedConc_crude'].rolling(window=self.size_window, center=True, min_periods=3).apply(lambda x: self.trim_fun(x))
        city_data['NormalizedConc_average'] = city_data['NormalizedConc_crude'].rolling(window=self.size_window, center=True, min_periods=3).mean()
        Data_ana = city_data[(city_data['SampleDate'] >= self.init_training) & (city_data['SampleDate'] <= self.end_training)]

        if self.city=='Davis':
            data = Data_ana['NormalizedConc_trimmed'][self.m - 1:]  # to eliminate the first m data to compare with the deconvolution
        else:
            data = Data_ana['NormalizedConc_average'][self.m - 1:]  # to eliminate the first m data to compare with the deconvolution

        data.index = pd.DatetimeIndex(Data_ana['SampleDate'])[self.m-1:]
        cases = Data_ana.positives
        city_data.index = pd.DatetimeIndex(city_data['SampleDate'])
        return city_data, cases, data

    def deconv(self, theta, conc_filled):
        """
        Compute the deconvolution with Richardson–Lucy algorithm
            :params theta: (array) (M,b):(Normalization constant, parameter
                                to compute the weights for shedding load distribution (wj))
            :params conc_filled:(Dateframe): Wastewater data (N/PMMoV)
            :return:(dcv): Incidence (Dateframe)
        """
        b = theta[1]
        M = theta[0]
        weight = stats.gamma.sf(self.time, self.a, scale=b)
        weight = weight / np.sum(weight)
        dcv = deconvolve_series(conc_filled / M, weight)
        return dcv

    def loglikelihood(self, x):
        """
        Likelihood functions available
        """
        if self.likelihood == 'Poisson':
            return self.loglikelihood_Pois(x)
        elif self.likelihood == 'Bin Neg':
            return self.loglikelihood_BN(x)
        elif self.likelihood == 'Bin_Neg_fp':
            return self.loglikelihood_BN_fp(x)
        else:
            return self.loglikelihood_logNormal(x)

    def loglikelihood_logNormal(self, x):
        """
        log normal likelihood
            :params x: (array) x=(M,b,sig):(Normalization constant, parameter
                                to compute the weights for shedding load distribution (wj),
                                variance)
            :return: log_likelihood (array)
        """
        sig = x[2]
        ss = -self.n * np.log(sig)
        deconvolution = self.deconv(x, self.data)
        v1 = (np.log(self.cases.values) - deconvolution.values)/sig
        log_likelihood = ss - 0.5*np.dot(v1, v1)
        return log_likelihood

    def loglikelihood_Pois(self, x):
        """
        Poisson likelihood
            :params x: (array) x=(M,b):(Normalization constant, parameter
                                to compute the weights for shedding load distribution (wj))
            :return: log_likelihood (array)
        """
        dec = self.deconv(x, self.data)
        mu = dec.values
        log_likelihood = np.sum(stats.poisson.logpmf(self.cases.values, mu))
        return log_likelihood

    def loglikelihood_BN(self, x):
        """
        Binomial negative likelihood
            :params x: (array) x=(M,b,theta):(Normalization constant, parameter
                                to compute the weights for shedding load distribution (wj),
                                overdispesion parameter)
            :return: log_likelihood (array)
        """
        dec = self.deconv(x, self.data)
        mu = dec.values + self.shift
        # likelihood for infectious
        theta = x[2] # 4
        r = mu / (self.omega - 1.0 + theta * mu)
        q = 1.0 / (self.omega + theta * mu)
        log_likelihood = np.sum(stats.nbinom.logpmf(self.cases.values + self.shift, r, q))
        return log_likelihood

    def loglikelihood_BN_fp(self, x):
        """
        Binomial negative likelihood with overdispersion parameters fixed
            :params x: (array) x=(M,b):(Normalization constant, parameter
                                to compute the weights for shedding load distribution (wj))
            :return: log_likelihood (array)
        """
        dec = self.deconv(x, self.data)
        mu = dec.values + self.shift

        r = mu / (self.omega - 1.0 + self.theta_bn * mu)
        q = 1.0 / (self.omega + self.theta_bn * mu)
        log_likelihood = np.sum(stats.nbinom.logpmf(self.cases.values + self.shift, r, q))
        return log_likelihood

    def logprior(self, x):
        """
        Logarithm of a gamma distribution
        """
        log_p = (self.alpha - 1)*np.log(x) - self.beta*x
        return np.sum(log_p)

    def Energy(self, x):
        """
        -log of the posterior distribution
        """
        return -1*(self.loglikelihood(x) + self.logprior(x))

    def Supp(self, x):
        """
        Support of the parameters to be estimated
        """
        b = x[1]
        weight = stats.gamma.sf(self.time, self.a, scale=b)
        weight = weight / np.sum(weight)
        return all(weight>0) and all(x > 0.0)

    def LG_Init(self):
        """
        Initial condition
        """
        sim = gamma.rvs(self.alpha, scale=1/self.beta)
        return sim.ravel()

    def predictive(self, out_th, conc):
        """
        Predictive funcion
        """

        if self.likelihood == 'logNormal':
            Output_trace = np.log(self.deconv(out_th, conc)) + norm.rvs(loc=0, scale=out_th[2], size=1)
        elif self.likelihood == 'Normal':
            Output_trace = self.deconv(out_th, conc) + norm.rvs(loc=0, scale=out_th[2], size=1)
        elif self.likelihood == 'Bin_Neg':

            mu = self.deconv(out_th, conc)# + self.shift
            theta = out_th[2]
            r = mu / (self.omega - 1.0 + theta * mu)
            q = 1.0 / (self.omega + theta * mu)
            Output_trace = nbinom.rvs(r, q)

        elif self.likelihood == 'Bin_Neg_fp':
            mu = self.deconv(out_th, conc) #+ self.shift
            r = mu / (self.omega - 1.0 + self.theta_bn * mu)
            q = 1.0 / (self.omega + self.theta_bn * mu)
            Output_trace = nbinom.rvs(r, q)

        else:  # Poisson
            Output_trace = poisson.rvs(self.deconv(out_th, conc))
        return Output_trace


    def eval_predictive(self, conc, Output_pars):
        """
        To evaluate the predictive funcion on mcmc samples
        """
        Output_trace = np.zeros((self.n_sample, conc.shape[0] + self.m - 1))
        i = 0
        while i < self.n_sample:
            out_th = Output_pars[-(i + 1)]
            Output_trace[i] = self.predictive(out_th, conc)
            i = i + 1

        if self.likelihood == 'logNormal':
            trace_Q500 = np.exp(np.quantile(Output_trace, 0.5, axis=0))
            trace_Q025 = np.exp(np.quantile(Output_trace, 0.10, axis=0))
            trace_Q975 = np.exp(np.quantile(Output_trace, 0.90, axis=0))
        else:
            trace_Q500 = np.quantile(Output_trace, 0.5, axis=0)
            trace_Q025 = np.quantile(Output_trace, 0.025, axis=0)
            trace_Q975 = np.quantile(Output_trace, 0.975, axis=0)
        return trace_Q500,  trace_Q025, trace_Q975

    def summary(self, Output_all):
        """
        Summary of mcmc samples
        """
        Output = Output_all[self.burnin::self.thini, :]
        Output_pars = Output[:, :self.d]
        Energy = Output[self.burnin:, -1]
        i = np.argmax(-Output_all[:, -1])
        th_map_ = Output_all[i, :]; th_map = th_map_[:self.d]

        return Output, Output_pars, th_map


    def DIC(self, output):
        """
        Compute deviance information criteria
        """
        theta_mean = np.mean(output, 0)
        log_vero = np.apply_along_axis(self.loglikelihood, 1, output)
        DIC_all = -2*log_vero
    
        # Deviance information criterion
        DIC_mean = -2*self.loglikelihood(theta_mean)
        mean_DIC = np.mean(DIC_all)
        DIC_i = 2*mean_DIC - DIC_mean
        var_DIC = 0.5 * np.var(DIC_all)
        DIC_v = var_DIC + mean_DIC
        return DIC_i, DIC_v

    def linear_model(self, city_data_, init_training, end_training):
        """
        Compute regression model
            :params city_data_: (dataframe)
            :params init_training: (str)
            :params end_training: (str)

            :return: (array, array) ymean:mean, ystd:standard variance
        """
        Y = city_data_['positives_average']
        #Y = city_data_['positives_inter']
        Date_full = pd.to_datetime(city_data_['SampleDate'])
        X = city_data_['NormalizedConc_trimmed'].values.reshape(-1, 1)
            
        eps = 1e-10
        X = np.log(X + eps)
        Y = np.log(Y + eps)

        index = np.where((Date_full >= init_training) & (Date_full <= end_training))[0]
        x_train = X[index]
        y_train = Y[index]

        reg = BayesianRidge(tol=1e-10, fit_intercept=True, compute_score=True)
        reg.fit(x_train, y_train)
        ymean, ystd = reg.predict(X, return_std=True)

        return ymean, ystd






