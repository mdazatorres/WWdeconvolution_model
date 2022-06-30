from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from run_mcmc import mcmc_conv, Params
import pandas as pd
import numpy as np

plt.rcParams['font.size'] = 19
Pars = Params()
workdir = "./"

def trim_fun(x):
    """ Remove max and min in an array
        :param x:(array)
        :return x1: (array)
    """
    x = x.dropna()
    x1 = x.sort_values().ravel()
    return np.mean(x1[1:-1])


def plot_data(city, save, workdir):
    """ Plot data vs Normalized concentration interpolated and trimmed
        :param city: (str)
        :params save:(bool)
        :params workdir:(dir)
        :return:fig: (fig):
    """
    test_set, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = Pars.training_date[city][test_set]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    city_data = mcmc.city_data
    city_data = city_data[init_date: end_date]
    fig, ax = subplots(num=1, figsize=(8,4)) #(10, 6)
    ax_p = ax.twinx()
    p1,=ax.plot(city_data.index, city_data['NormalizedConc_trimmed'],  linewidth=3, color='brown', label='Smoothed N/PMMoV')

    p3,=ax_p.plot(city_data.index, city_data['positives_crude'], 'o', markersize=5, linewidth=3, color='k', alpha=0.6, label='Cases')
    if city =='Woodland':
        p2, = ax_p.plot(city_data.index[:-45], city_data['positives_average'][:-45], '-', linewidth=3, color='gray',label='Smoothed cases')
    else:
        p2, = ax_p.plot(city_data.index, city_data['positives_average'], '-', linewidth=3,color='gray', label='Smoothed cases')

    # for i in range(2):
    #     init_training, end_training = Pars.training_date[city][i]
    #     ax.axvspan(init_training, end_training, alpha=0.2, color=Pars.colors[i])

    ax.set_ylim(-1e-6,Pars.ylim[city][0])
    ax.legend(handles=[p3, p2, p1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_p.set_ylabel('Cases')
    ax.set_ylabel('N/PMMoV')
    #ax.axes.yaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/'+city + '_data.png')


def plot_data_test(city, save):
    """ Plot testing and cases data
        :param city: (str)
        :params save:(bool)
        :params workdir:(dir)
        :return:fig: (fig):
    """
    test_set, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = Pars.training_date[city][test_set]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    city_data = mcmc.city_data
    city_data = city_data[init_date: end_date]
    fig, ax = subplots(num=1, figsize=(12, 5))
    ax_p = ax.twinx()
    p1,=ax.plot(city_data.index, city_data['Testing'].rolling(window=7, center=True, min_periods=2).mean(),  linewidth=2, color='blue', label='Tests')
    p3,=ax_p.plot(city_data.index, city_data['positives_crude'], 'o', markersize=5, linewidth=3, color='k', alpha=0.6, label='Cases')
    p2, = ax_p.plot(city_data.index, city_data['positives_average'], '-', linewidth=3,color='gray', label='Smoothed cases')

    ax.legend(handles=[p3, p2, p1]) #loc=5 woodland
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_p.set_ylabel('Cases')
    ax.set_ylabel('Tests')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/'+city + '_data_tests_all.png')


def plot_smother_comp_conc(city, mov_ave, data_imput, save, workdir=workdir):

    """ Plot WW concentration data applying different smooth functions
        :param city: (str)
        :params mov_ave:(bool)
        :params data_imput:(bool)
        :params save:(bool)
        :params workdir:(dir)
        :return fig: (fig):
    """
    fig, ax = subplots(num=1, figsize=(8, 4)) #figsize=(10, 6)
    test_set, m, init_date, end_date = Pars.params[city].values()
    init_training, end_training = Pars.training_date[city][test_set]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    city_data = mcmc.city_data
    city_data = city_data[init_date: end_date]

    ax.plot(city_data.index, city_data['NormalizedConc_crude'], 'o', markersize=5, color='blue', label='N/PMMoV')
    if data_imput:
        ax.plot(city_data.index, city_data['NormalizedConc'], 'o',markersize=5, color='cyan', label=r'Imputed')
    ax.plot(city_data.index, city_data['NormalizedConc_crude'], 'o', markersize=5, color='blue')
    if mov_ave:
        ax.plot(city_data.index, city_data['NormalizedConc_average'], lw=3, color='brown', label='Smoothed N/PMMoV')
    else:
        ax.plot(city_data.index, city_data['NormalizedConc_trimmed'], lw=3, color='brown', label='Smoothed N/PMMoV')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_ylabel('N/PMMoV')
    ax.set_ylim(-1e-6,Pars.ylim[city][0])
    ax.legend(frameon=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    ax.tick_params(which='major', axis='x')
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    if save:
        fig.savefig(workdir + 'figures/'+city+'_imputed_data'+'.png')

   
    


city = 'Davis'
#city='Woodland'
plot_smother_comp_conc(city, mov_ave=False, data_imput=False, save=True, workdir=workdir)
#plot_smother_comp_conc(city, mov_ave=True, data_imput=False, save=True, workdir=workdir)
#plot_data(city,workdir=workdir, save=True)
#plot_data_test(city, save=True)

