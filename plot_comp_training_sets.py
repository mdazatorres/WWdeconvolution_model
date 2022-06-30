from run_mcmc import mcmc_conv, Params
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from scipy.stats import gamma
import matplotlib.dates as mdates
import datetime


plt.rcParams['font.size'] = 20
font_xylabel = 24

city='Davis'
#city = 'UCDavis'
#city = 'Woodland'
workdir = "./"
Pars = Params()


def plot_deconv(training_date, city, cases, output_name, workdir, ax, label, color):
    """ Plot estimated cases with the deconvolution model
        :param training_date:(str)
        :param city: (str)
        :param cases: (array)
        :params output_name:(str)
        :params workdir:(dir)
        :params ax:(fig)
        :params label:(str)
        :params color:(str)
        :return:fig: (fig):
    """
    test_set, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = training_date
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    if output_name==None:
        output_mcmc = pd.read_csv(workdir + 'output/' + 'mcmc_' + city + '_' + 'loglik_' + str(mcmc.likelihood) + '_m_' + str(mcmc.m) + '_' + init_training + '-' + end_training, index_col=0)
    else:
        output_mcmc = pd.read_csv(workdir + 'output/' + output_name, index_col=0)
    conc_filled = mcmc.city_data['NormalizedConc_trimmed']
    conc_filled.index = pd.DatetimeIndex(mcmc.city_data.SampleDate)
    Output_all = output_mcmc.values
    _, Output_pars, th_map = mcmc.summary(Output_all)
    conc_filled_ = conc_filled[init_date:end_date]

    if city == 'Woodland':
      new_ix = pd.date_range(start=conc_filled_.index[0], end=conc_filled_.index[-1], freq='D')
      conc_filled_ = conc_filled_.reindex(new_ix, method='bfill')
      conc_filled_=conc_filled_.interpolate()
    if city=='UCDavis':
        conc_filled_[conc_filled_==0]=1e-10
    city_data = mcmc.city_data[init_date:end_date]
    trace_Q500, trace_Q025, trace_Q975 = mcmc.eval_predictive(conc_filled_, Output_pars)
    if cases:
      ax.plot(city_data.index, city_data.positives_crude, 'o', markersize=4, linewidth=2, color='k', alpha=0.6, label="Cases")
    ax.plot(city_data.index, trace_Q500[m - 1:], markersize=2, linewidth=2, color=color, label=label)
    ax.fill_between(city_data.index, trace_Q025[m-1:], trace_Q975[m-1:], color=color, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim(city_data.index[0] - datetime.timedelta(days=1), city_data.index[-1]+ datetime.timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")



def plot_linear_model(city, cases, label, test_set, color, ax):
    """ Plot estimated cases with the linear model
        :param city: (str)
        :param cases: (array)
        :params label:(str)
        :params test_set:(int)
        :params ax:(str)
        :return:fig: (fig):
    """

    test_set_best, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = Pars.training_date[city][test_set]
    if city=='UCDavis':
      k=0.5
    else:
      k=1
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    city_data = mcmc.city_data[init_date: end_date]

    ymean, ystd = mcmc.linear_model(city_data_=city_data, init_training=init_training, end_training=end_training)

    if cases:
      ax.plot(city_data.index, city_data.positives_crude, 'o', markersize=4, linewidth=2, color='k', alpha=0.6,label="Cases")

    ax.plot(city_data.index, np.exp(ymean), color=color, label=label, lw=2)
    ax.fill_between(city_data.index, np.exp(ymean - k*ystd), np.exp(ymean + k*ystd), color=color, alpha=0.3) #label="Predict std"

    ax.set_xlim(city_data.index[0] - datetime.timedelta(days=1), city_data.index[-1] + datetime.timedelta(days=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    ax.tick_params(which='major', axis='x')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))



def comparison_conv_Tsets(city, output_name_1, output_name_2, workdir, save):
    """ Plot estimated cases with the deconvolution model with the two selected training sets
        :param city: (str)
        :param output_name_1/2: (str) # output name for the mcmc samples to the dec model with the training set adequate/not adequate
        :params workdir:(dir)
        :params save:(bol)
        :return:fig: (fig):
    """
    fig, ax = subplots(num=1, figsize=(9, 5))
    plot_deconv(Pars.training_date[city][0], city=city, cases=True, output_name=output_name_1, workdir=workdir, ax=ax, label=r'$T_A$', color=Pars.colors[0])
    plot_deconv(Pars.training_date[city][1], city=city,  cases=False, output_name=output_name_2, workdir=workdir, ax=ax, label=r'$T_{NA}$', color=Pars.colors[1])
    ax.set_ylim(-10, Pars.ylim[city][1])
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/' + city + '_deconvolution_comparison' + '.png')


def comparison_linear_Tsets(city, save):
    """ Plot estimated cases with the linear model with the two selected training sets
        :param city: (str)
        :param save: (bool)
        :return:fig: (fig):
    """
    fig, ax = subplots(num=1, figsize=(9, 5))
    plot_linear_model(city=city, cases=True, label=r'$T_A$', color='blue', test_set=0, ax=ax)
    plot_linear_model(city=city, cases=False, label=r'$T_{NA}$',color='red', test_set=1, ax=ax)
    ax.axes.yaxis.set_ticklabels([])
    ax.set_ylim(-10, Pars.ylim[city][1])
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/' + city + '_linear_comparison' + '.png')


def plot_linear_vs_conv(city, test_set, output_name, workdir, save):
    """ Plot estimated cases with the linear and the deconvolution model with the  adequated training set
        :param city: (str)
        :param test_set: (int)
        :params output_name:(str)
        :params workdir:(dir)
        :params save:(bool)
        :return:fig: (fig):
    """
    fig, ax = subplots(num=1, figsize=(9, 5))
    plot_deconv(Pars.training_date[city][test_set], city=city, cases=True, output_name=output_name, workdir=workdir, ax=ax, label="Deconvolution", color='green')
    plot_linear_model(city=city, cases=False, label='Linear', color='blue', test_set=test_set, ax=ax)
    ax.set_ylim(-10, Pars.ylim[city][1])
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/'+ city + '_comp_conv_linear_model' + '.png')


#comparison_conv_Tsets(city, output_name_1=None, output_name_2=None, workdir=workdir, save=False)
#comparison_conv_Tsets(city, output_name=None, workdir=workdir,save=False)
#comparison_conv_linear(city, save=True)
#plot_linear_vs_conv(city=city, test_set=0, output_name=None, workdir=workdir, save=True)
#plot_linear_model(city=city, cases=False, label='Linear', ax=ax)










