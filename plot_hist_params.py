from run_mcmc import mcmc_conv, Params
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from scipy.stats import gamma



plt.rcParams['font.size'] = 20
font_xylabel = 24
city = 'Davis'
#city = 'Woodland'
workdir = "./"
Pars = Params()


def plot_params(city, par, save):
  """ Plot posterior distribution of a specific parameter.
    :param par (int) parameter's position to plot
    :param save (bolean)
    :return: figure (.png)
    """
  m = Pars.params[city]['m']
  plt.rcParams['font.size'] = 28
  pars = ['B', 'b', r'$\sigma$']
  fig, ax = subplots(num=1, figsize=(9, 5))
  nn = len(Pars.training_date[city])
  for i in range(nn):
    init_training, end_training = Pars.training_date[city][i]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    mcmc_data = pd.read_csv(workdir + 'output/' + 'mcmc_' + city + '_' + 'loglik_' + str(mcmc.likelihood) + '_m_' + str(mcmc.m) + '_' + init_training + '-' + end_training, index_col=0)
    Output_mcmc = mcmc_data.values
    Output, Output_pars, th_map= mcmc.summary(Output_mcmc)
    ax.hist(Output_pars[:, par], lw=2, density=True, histtype=u'step', label=r'$T_%s$'%(i+1), color=Pars.colors[i])

  #xpri = np.linspace(0, 1, 400)
  #ax.plot(xpri, gamma.pdf(xpri, mcmc.alpha[par], scale=1/mcmc.beta[par]), color='k')
  ax.set_ylabel('Density', fontsize=font_xylabel)
  ax.set_xlabel(pars[par], fontsize=font_xylabel)
  ax.legend()
  fig.tight_layout()
  if save:
    if par==2:
      fig.savefig(workdir + 'figures/' + city + '_hist_M'+'.png')
    else:
      fig.savefig(workdir + 'figures/' + city + '_hist_' + pars[par]+ '.png')



def plot_SLD(city, save):
  """
  Plot the estimated SLD for each training set.
    :param training_date (list) list of trainig sets.
    :param save (bolean)
    :return: figure (.png)
  """
  fig, ax = subplots(num=1, figsize=(9, 5))
  m = Pars.params[city]['m']
  nn = len(Pars.training_date[city])
  for i in range(nn):
    init_training, end_training = Pars.training_date[city][i]
    mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
    mcmc_data = pd.read_csv(workdir + 'output/' + 'mcmc_' + city + '_' + 'loglik_' + str(mcmc.likelihood) + '_m_' + str(mcmc.m) + '_' + init_training + '-' + end_training, index_col=0)

    Output_mcmc = mcmc_data.values
    Output, Output_pars, th_map = mcmc.summary(Output_mcmc)
    weight = gamma.sf(mcmc.time, mcmc.a, scale=1/th_map[1])
    weight_map = weight / np.sum(weight)
    ax.bar(mcmc.time, weight_map, lw=2, edgecolor=Pars.colors[i], label=r'$T_%s$'%(i+1), fill=False)
  ax.set_xlabel('Days', fontsize=font_xylabel)
  ax.set_ylabel('SLD', fontsize=font_xylabel)
  ax.legend()
  fig.tight_layout()
  if save:
    fig.savefig(workdir + 'figures/' + city + '_SLD'+  '.png')



#plot_SLD(city,  save=True)
#plot_params(city=city, par=1, save=True)













