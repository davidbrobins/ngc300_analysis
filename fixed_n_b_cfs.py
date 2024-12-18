# Script to compare GH12, RAG24 net CFs at fixed densities

# Import statements
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import matplotlib.pyplot as plt # Matplotlib for plotting
import gh12_chf # f2py wrapper for GH12 CHF interpolation table approximation
import xgboost as xgb # XGBoost
from pickle import load # Function to read in pickle files

# Read in the XGBoost models for CF, HF                                                                                                                                   
CF_model = xgb.XGBRegressor()                                                                                                                                             
CF_model.load_model('/nfs/turbo/lsa-cavestru/dbrobins/ml_chf/models/gh12_rates/all_data/CF_Z_0.3/trained_model.txt')                                                      
HF_model = xgb.XGBRegressor()                                                                                                                                             
HF_model.load_model('/nfs/turbo/lsa-cavestru/dbrobins/ml_chf/models/gh12_rates/all_data/HF_Z_0.3/trained_model.txt')                                                      
# Function to compute the XGBoost cooling function                                                                                                                        
def get_chf_xgb(temp, den_bar):
    ''' Function to get the net cooling function (cooling - heating) given gas temperature, baryon number density
    Inputs:
    temp (float): gas temperature, in K
    den_bar (float): gas baryon number density, in cm^{-3}
    Outputs:
    cf (float): gas cooling function - heating function, in erg cm^{3} s^{-1}
    '''
    # Constant values of the photoionization rates (to be consistent with simulation)                                                                                     
    P_LW = 2e-11                                                                                                                                                          
    P_HI = 2e-17                                                                                                                                                          
    P_HeI = 3e-16                                                                                                                                                         
    P_CVI = 9e-18                                                                                                                                                         
    # Convert from baryon density to hydrogen number density                                                                                                              
    Z = 0.3                                                                                                                                                               
    w = (1 - 0.02 * Z) / 1.4                                                                                                                                              
    hden = w * den_bar                                                                                                                                                    
    # Scale the features                                                                                                                                                  
    t_feat = (np.log10(temp) - (1.000000)) / (9.000000 - (1.000000))                                                                                                      
    n_h_feat = (np.log10(hden) - (-6.000000)) / (6.000000 - (-6.000000))                                                                                                  
    q_lw_feat = (np.log10(P_LW / hden) - (-14.940437)) / (-2.822464 - (-14.940437))                                                                                       
    q_hi_feat = (np.log10(P_HI / P_LW) - (-6.911031)) / (0.481141 - (-6.911031))                                                                                          
    q_hei_feat = (np.log10(P_HeI / P_LW) - (-5.551412)) / (0.717863 - (-5.551412))                                                                                        
    q_cvi_feat = (np.log10(P_CVI / P_LW) - (-9.017760)) / (-1.062396 - (-9.017760))                                                                                       
    # Make the predictions                                                                                                                                                  
    log_cf = CF_model.predict([[t_feat, n_h_feat, q_lw_feat, q_hi_feat, q_hei_feat, q_cvi_feat]])[0]                                                                      
    log_hf = HF_model.predict([[t_feat, n_h_feat, q_lw_feat, q_hi_feat, q_hei_feat, q_cvi_feat]])[0]                                                                      
    # Convert from log(CF) to CF (erg cm^3/s), return cooling and heating functions                                                                                                                            
    return 10 ** log_cf, 10 ** log_hf

# Evaluate GH12 approximation
#initialize CF with values from cf_table.I2.dat
print('Just before initializing table...')
gh12_chf.frtinitcf(0,'cf_table.I2.dat')
print('Table initialized!')
# Function to get net cooling from GH12 interpolation table
def get_chf_gh12(temp, den_bar):
    # Constant values of the photoionization rates (to be consistent with simulation)                                                                                     
    P_LW = 2e-11
    P_HI = 2e-17
    P_HeI = 3e-16
    P_CVI = 9e-18
    # Convert from baryon density to hydrogen number density                                                                                                              
    Z = 0.3
    #print('P_LW: ', P_LW)
    # Evaluate the interpolation table
    (cfun, hfun, ierr) = gh12_chf.frtgetcf(temp, den_bar, Z, P_LW, P_HI, P_HeI, P_CVI)
    # Return cooling function and heating function
    return cfun, hfun

# Get array of temps
t_vals = np.logspace(1, 9, 81)
# Initialize empty arrays to hold CFs and HFs
gh12_cf_log_n_0 = np.zeros(len(t_vals))
gh12_hf_log_n_0 = np.zeros(len(t_vals))
xgb_cf_log_n_0 = np.zeros(len(t_vals))
xgb_hf_log_n_0 = np.zeros(len(t_vals))
# Get CFs at these temps for some densities, with both approximations
for temp_index in range(len(t_vals)):
    # log(n) = 0, GH12
    gh12_cf_log_n_0[temp_index], gh12_hf_log_n_0[temp_index] = get_chf_gh12(t_vals[temp_index], 1e0)
    # log(n) = 0, XGB
    xgb_cf_log_n_0[temp_index], xgb_hf_log_n_0[temp_index] = get_chf_xgb(t_vals[temp_index], 1e0)
    print('Did T=', t_vals[temp_index], ' K')

# Initiate the plot
fig, ax = plt.subplots(1, 1, figsize = (3.4, 2.4))
# Plot GH12 cooling function in solid blue
ax.plot(t_vals, gh12_cf_log_n_0, color = 'b', linestyle = 'solid', label = r'$\Lambda, \log(n_b/\mathrm{cm}^{-3})=0$, GH12')
# Plot GH12 heating function in solid red
ax.plot(t_vals, gh12_hf_log_n_0, color = 'r', linestyle = 'solid', label = r'$\Gamma, \log(n_b/\mathrm{cm}^{-3})=0$, GH12')
# Plot XGB cooling function in dashed blue
ax.plot(t_vals, xgb_cf_log_n_0, color = 'b', linestyle = 'dashed', label = r'$\Lambda, \log(n_b/\mathrm{cm}^{-3})=0$, XGB')
# Plot XGB heating function in dashed red
ax.plot(t_vals, xgb_hf_log_n_0, color = 'r', linestyle = 'dashed', label = r'$\Gamma, \log(n_b/\mathrm{cm}^{-3})=0$, XGB')
# Use log-scaled axes
ax.set_xscale('log')
ax.set_yscale('log')
# Label axes
ax.set_xlabel(r'$T \, [\mathrm{K}]$')
ax.set_ylabel(r'$\mathcal{F} \, [\mathrm{erg} \, \mathrm{cm}^{3} \, \mathrm{s}^{-1}]$')
ax.legend()
# Save figure
fig.tight_layout()
fig.savefig('fixed_density_cfs.pdf')
plt.close(fig)
