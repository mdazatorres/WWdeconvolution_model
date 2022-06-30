from run_mcmc import mcmc_conv, Params
from pytwalk import pytwalk
import pandas as pd


workdir = "./"
Pars = Params()


def save_output(city, m, ts, workdir, output_name):
  """ Generate mcmc output for a training set.
      :param city:(str)
      :param m: (int)
      :param ts: (int) test set
      :params workdir: (str)
      :params output_name:(str)
      :return:(Output_df): (Dateframe)
  """
  i=ts
  init_training, end_training = Pars.training_date[city][i]
  mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=m)
  LG_twalk = pytwalk(n=mcmc.d, U=mcmc.Energy, Supp=mcmc.Supp)
  LG_twalk.Run(T=50000, x0=mcmc.LG_Init(), xp0=mcmc.LG_Init())
  if mcmc.d == 2:
    Output_df = pd.DataFrame(LG_twalk.Output, columns=['M','b', 'Energy'])
  else:
    Output_df = pd.DataFrame(LG_twalk.Output, columns=['M', 'b', 'sigma','Energy'])
  if output_name==None:
    Output_df.to_csv(workdir+'output/'+'mcmc_'+city+'_'+'loglik_'+str(mcmc.likelihood)+'_m_'+str(mcmc.m)+'_'+init_training+'-'+end_training)
  else:
    Output_df.to_csv(workdir + 'output/' + output_name)


#city = 'Woodland'
#city='Davis'
city='Davis'
m = 7
#save_output(city, m, ts=0,workdir=workdir, output_name=None)
