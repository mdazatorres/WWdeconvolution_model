from run_mcmc import mcmc_conv, Params
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
import datetime


plt.rcParams['font.size'] = 25

workdir = "./"
Pars = Params()

def fig_training_set_WW(city, save):
    """ Plot Testing vs  cases and the training-sets.
      :param city (str)
      :param save (bolean)
      :return: figure (.png)
    """
    data_test = pd.read_excel('data/Testing_case_HYT.xlsx', sheet_name=city)
    data_test['m test'] = np.round(data_test['Total']/data_test['Total'].shift(), 2)
    data_test['m pos'] = np.round(data_test['Positive']/data_test['Positive'].shift(), 2)
    #fig, ax = subplots(num=1, figsize=(14, 7))
    fig, ax = subplots(num=1, figsize=(10, 5))
    data_test.index = data_test['End Date']
    init_date = Pars.params[city]['init_date']; end_date= Pars.params[city]['end_date']

    data_test_ = data_test[init_date:end_date]
    ax.semilogy(data_test_['End Date'], data_test_['Total'], '-o', markersize=3, linewidth=2, color='c', label='Tests')
    ax.semilogy(data_test_['End Date'], data_test_['Positive'], '-o', markersize=3, linewidth=2, color='k', label='Cases')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(which='major', axis='x')
    for i in range(1, len(data_test_['End Date'])):
        if data_test_['m test'][i] > data_test_['m pos'][i]:
            color1 = 'red'
            color2 = 'green'
        else:
            color1 = 'green'
            color2 = 'red'

        ax.text(data_test_['End Date'][i]-datetime.timedelta(days=4), data_test_['Positive'][i], data_test_['m pos'][i], horizontalalignment='left', size='small', color=color1)
        ax.text(data_test_['End Date'][i]-datetime.timedelta(days=4), data_test_['Total'][i], data_test_['m test'][i],horizontalalignment='left', size='small', color=color2)
    for i in range(2):
        init_training, end_training = Pars.training_date[city][i]
        ax.axvspan(init_training, end_training, alpha=0.2, color=Pars.colors[i])

    ax.text(data_test_.index[2]+ datetime.timedelta(days=2), 500, r'$T_{NA}$',horizontalalignment='right', size='small', color='black', weight='semibold')
    ax.text(data_test_.index[7] + datetime.timedelta(days=1), 500, r'$T_{A}$',horizontalalignment='right', size='small', color='black', weight='semibold')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
    ax.set_xlabel('2021' + '                              ' + '2022')
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig(workdir + 'figures/' + city + '_training_periods' + '.png')



def fig_training_set(city, save):
  """ Plot Testing vs  cases and the training-sets.
    :param city (str)
    :param save (bolean)
    :return: figure (.png)
  """
  test_set, m, init_date, end_date = Pars.params[city].values()
  init_training, end_training = Pars.training_date[city][test_set]
  mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
  city_data = mcmc.city_data
  city_data =  city_data[init_date: end_date]
  city_data_w = city_data.groupby(pd.Grouper(freq="W")).sum()

  #fig, ax = subplots(num=1, figsize=(18, 8))
  fig, ax = subplots(num=1, figsize=(10, 5))
  ax.semilogy(city_data_w.index, city_data_w['Testing'], '-o', markersize=3, linewidth=2, color='c', label='Tests')
  ax.semilogy(city_data_w.index, city_data_w['positives'], '-o', markersize=3, linewidth=2,  color='k', label='Cases')
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

  city_data_w['m test'] = np.round(city_data_w['Testing'] / city_data_w['Testing'].shift(), 2)
  city_data_w['m pos'] = np.round(city_data_w['positives'] / city_data_w['positives'].shift(), 2)

  for i in range(1, len(city_data_w.index)):
      if city_data_w['m test'][i]>city_data_w['m pos'][i]:
          color1 = 'red'
          color2='green'
      else:
          color1 = 'green'
          color2 = 'red'

      ax.text(city_data_w.index[i] - datetime.timedelta(days=1), city_data_w['positives'][i], city_data_w['m pos'][i],
              horizontalalignment='right', size='xx-small', color=color1)
      ax.text(city_data_w.index[i] - datetime.timedelta(days=1), city_data_w['Testing'][i], city_data_w['m test'][i],
              horizontalalignment='right', size='xx-small', color=color2)

  ax.text(city_data_w.index[7], 20, r'$T_A$',
          horizontalalignment='right', size='small', color='black', weight='semibold')
  ax.text(city_data_w.index[3] + datetime.timedelta(days=2), 20, r'$T_{NA}$',
          horizontalalignment='right', size='small', color='black', weight='semibold')

  ax.set_ylabel('Log-scale')

  for i in range(2):
   init_training, end_training = Pars.training_date[city][i]
   ax.axvspan(init_training, end_training, alpha=0.2, color=Pars.colors[i])
  ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
  ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
  ax.legend()
  #ax.legend(frameon=False)
  ax.tick_params(which='major', axis='x')
  ax.set_xlabel('2021'+ '                              '+'2022')
  plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
  if save:
    fig.savefig(workdir + 'figures/' + city + '_training_periods'+'.png')



def plot_selec_training_set(city, save):
    """ Plot Testing vs  cases and the training-sets.
      :param city (str)
      :param save (bolean)
      :return: figure (.png)
    """
    if city=='Woodland':
        fig_training_set_WW(city, save)
    else:
        fig_training_set(city, save)


#city = 'Davis (sludge)'
#city = 'Davis'
#city = 'UCDavis'
#ig_training_set_WW(city='Woodland', save=True)
#plot_selec_training_set(city='UCDavis', save=True)


