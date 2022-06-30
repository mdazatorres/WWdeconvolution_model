from run_mcmc import mcmc_conv, Params
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['font.size'] = 20
font_xylabel = 24
#city = 'Davis'
city='Woodland'


workdir = "./"


Pars= Params()

mm = [6, 7, 8,9, 10]

def computeDIC(city, test_set):
    """ Compute DIC
        :param city: (str)
        :return: df_DIC: (dataframe):
    """
    init_training, end_training = Pars.training_date[city][test_set]
    DIC=[]
    for i in range(len(mm)):
        mcmc = mcmc_conv(city=city, init_training=init_training, end_training=end_training, m=mm[i])
        output_mcmc= pd.read_csv(workdir + 'output/' + 'mcmc_' + city + '_' + 'loglik_' + str(mcmc.likelihood) + '_m_' + str(mm[i]) + '_' + init_training + '-' + end_training, index_col=0)
        Output =output_mcmc.values[mcmc.burnin::mcmc.thini, :]
        Output_theta = Output[:, :mcmc.d]

        DIC.append(mcmc.DIC(Output_theta))
    df_DIC=pd.DataFrame(DIC, columns=['DIC i', 'DIC_v'], index=[ 'm6', 'm7', 'm8', 'm9','m10'])
    return df_DIC


A = computeDIC(city=city, test_set=0)





