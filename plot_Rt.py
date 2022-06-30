from run_mcmc import mcmc_conv
import matplotlib.pyplot as plt
from pytwalk import pytwalk
import numpy as np
import pandas as pd
from run_mcmc import Params
import matplotlib.dates as mdates
import matplotlib as mpl

import epyestim.covid19 as covid19
import matplotlib.pyplot as plt


fontsize=20
plt.rcParams['font.size'] = fontsize
Pars = Params()
#city = 'UCDavis'
#city='Davis'
city = 'Woodland'
workdir = "./"



def estimated_cases(city, method):
    test_set, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = Pars.training_date[city][test_set]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)

    output_mcmc = pd.read_csv(workdir + 'output/' + 'mcmc_' + city + '_' + 'loglik_' + str(mcmc.likelihood) + '_m_' + str(mcmc.m) + '_' + init_training + '-' + end_training, index_col=0)

    conc_filled = mcmc.city_data.NormalizedConc_trimmed
    conc_filled.index = pd.DatetimeIndex(mcmc.city_data.SampleDate)
    Output_all = output_mcmc.values
    Output, Output_theta, th_map, th_mean, Q025, Q500, Q975 = mcmc.summary(Output_all)
    conc_filled_ = conc_filled[init_date:end_date]
    city_data = mcmc.city_data[init_date:end_date]
    if city=='UCDavis':
        conc_filled_[conc_filled_==0]=1e-10

    if city == 'Woodland' or city == 'Winters':
        new_ix = pd.date_range(start=conc_filled_.index[0], end=conc_filled_.index[-1], freq='D')
        conc_filled_ = conc_filled_.reindex(new_ix, method='bfill')
        conc_filled_ = conc_filled_.interpolate()

    if method=='Linear':
        ymean, ystd, dates, cases_plot = mcmc.linear_model(city_data_=city_data, init_training=init_training, end_training=end_training)
        #ymean, ystd, dates, cases_plot = mcmc.linear_model(city_data_=city_data)
        est_cases = np.exp(ymean)
        date = city_data.index

    elif method=='Deconvolution':
        sample = 3000
        Output_trace = np.zeros((sample, conc_filled_.shape[0] + m - 1))
        i = 0
        while i < sample:
            out_th = Output_theta[-(i + 1)]
            Output_trace[i] = mcmc.predictive(out_th, conc_filled_)
            i = i + 1
        trace_Q500 = np.quantile(Output_trace, 0.5, axis=0)
        est_cases = trace_Q500[m-1:]
        date =city_data.index

    else:
        date = city_data.index
        est_cases = city_data.positives_crude.interpolate(method='linear', axis=0).ravel()
    return date, est_cases




def plot_RT(ax, cases,dates,label, color):
  davisdf_s = pd.Series(data=cases, index=dates)
  ch_time_varying_r = covid19.r_covid(davisdf_s)

  dates_ = ch_time_varying_r.index.strftime("%b-%d-%y")

  mpl.rcParams['axes.spines.right'] = False

  ax.plot(dates_[:-13],ch_time_varying_r['Q0.5'][:-13], color=color, label=label)
  ax.fill_between(dates_[:-13], ch_time_varying_r['Q0.025'][:-13],ch_time_varying_r['Q0.975'][:-13], color=color, alpha=0.15)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
  ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
  ax.set_xlabel('2022')
  ax.tick_params(which='major', axis='x')

  ax.set_xlim([dates_[0],dates_[-14]])

  ax.set_ylabel(r'$R_e(t)$ with 95% CI', fontsize=fontsize)

  ax.axhline(y=1, color="black")
  #ax.grid(linestyle='-')


def plot_comp_Rt(city, save):
    methods =['Deconvolution', 'Linear', 'Cases']
    colors=['green','blue', 'black']
    nn=len(methods)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), tight_layout=True)
    for i in range(nn):
        date, cases = estimated_cases(city, method=methods[i])
        plot_RT(ax, cases, date, color=colors[i], label=methods[i])
    ax.legend(loc='upper right', frameon=False)
    ax.grid(linestyle='--')
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/' + city + '_Rt' + '.png')

plot_comp_Rt(city, save=True)


