 
#### Simulation codes for the paper:
### Model training periods impact estimation of COVID-19 incidence from wastewater viral loads

This module consists in the following files:

#### Data
data_ww_cases.csv
Testing_case_HYT.xls

#### Codes

#### Example_Estimation of COVID-19 incidence from ww data.ipynb

Notebook tutorial to estimate COVID-19 incidence from ww data

#### run_mcmc.py
Main code:
- Set all the parameters for the models
- Load and processed data
- Likelihood, priors for mcmc are defined
- DIC function 
- Linear model 

#### plot_training_set.py
We plotted number of tests administered and cases by week for Davis, UCDavis and Woodland, on a log-scale, from December 1, 2021 to March 31, 2022. Also the training periods  used for the analysis.

#### save_mcmc.py
This function is for running and saving the mcmc for each training period. We just save the output.

#### plot_data.py
To plot ww concentration and covid-19 cases (trimmed and moving average). 

#### plot_comp_training_sets.py

- plot_deconv(): Plot estimated cases with the deconvolution model
- plot_linear_model(): Plot estimated cases with the linear model
- comparison_conv_Tsets(): Plot estimated cases with the deconvolution model with the mcmc output for the two selected training sets.
- comparison_linear_Tsets(): Plot estimated cases with the linear model for the two selected training sets
- plot_linear_vs_conv() Plot estimated cases with the linear a deconvolution model

#### plot_hist_params.py

- Plot posterior distribution for estimated parameters


Auxiliary programs

- #### pytwalk.py

Library for the t-walk MCMC algorithm. For more details about this library see https://www.cimat.mx/~jac/twalk/

- #### deconvolution.py

A Python library Epyestim helps us with this task by calculating the effective reproduction rate for the reported COVID-19 cases. . For more details about this library see
https://github.com/lo-hfk/epyestim


