# Script to gneratet phase plots without RT variables (T vs. baryon number density)

# Import statements                                                                                                                    
import yt # yt to handle simulation data
from yt.units import gram, second, erg, K,  centimeter # needed units    
import derived_fields # yt field calculations                                                            
import matplotlib.pyplot as plt # Matplotlib for plotting
import numpy as np # Numpy for math

# Define a function to make residual phase plot after n timesteps of 1 Myr
def residual_phase_plot(n, plot = True, show_most_affected_gas = False, den_bins = 250, temp_bins = 175):
    '''
    Function to plot residual phase plot after n timesteps of 1 Myr
    Inputs:
    n (int): Number of 1 Myr timesteps after initial snapshot
    plot (bool): Whether or not to save a PDF of the residual phase plot (default: True)
    show_most_affected_gas (bool): Whether or not to save a PDF of the most affected gas (default: False)
    den_bins (int): Number of density bins (default: 250)
    temp_bins (int): Number of temperature bins (default: 175)
    Outputs:
    residual (ndarray) : 2D array of the residuals (and saves PDF of residual phase plot if plot = True)
    '''

    # Create string with snapshot number
    snap_num = str(300 + n)
    snap_num = snap_num.zfill(6)
    
    # Read in the data for the GH12 CHF run
    ds_gh12 = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-gh12/outputs/fiducial_' + snap_num + '.art')
    # Extract density and temperature fields, and cell gas masses
    gh12_data = ds_gh12.all_data()
    den_gh12 = gh12_data[('gas', 'baryon_number_density')]
    temp_gh12 = gh12_data[('gas', 'temperature')]
    cell_mass_gh12 = gh12_data[('gas', 'cell_mass')]
    # Compute the 2D phase diagram, weighted by gas mass
    hist_gh12 = np.histogram2d(np.log10(temp_gh12), np.log10(den_gh12), bins = [temp_bins, den_bins],
                               range = [[1, 8], [-7, 3]], weights = cell_mass_gh12, density = False)[0]
    # Do the same for the RAG24 CHF run
    ds_xgb = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-xgbchf-test/outputs/fiducial_' + snap_num + '.art')
    xgb_data = ds_xgb.all_data()
    den_xgb = xgb_data[('gas', 'baryon_number_density')]
    temp_xgb = xgb_data[('gas', 'temperature')]
    cell_mass_xgb = xgb_data[('gas', 'cell_mass')]
    hist_xgb = np.histogram2d(np.log10(temp_xgb), np.log10(den_xgb), bins = [temp_bins, den_bins],
                              range = [[1, 8], [-7, 3]], weights = cell_mass_xgb, density = False)[0]

    # Get normalized residual (set to 0 if both phase diagrams are 0)
    residual = np.array([[0 if hist_gh12[i][j] == 0 and hist_xgb[i][j] == 0
                          else (hist_gh12[i][j] - hist_xgb[i][j])/(hist_gh12[i][j] + hist_xgb[i][j])
                          for j in range(den_bins)]
                         for i in range(temp_bins)])
    # Plotting:
    if plot == True:
        # Create a plot
        fig, ax = plt.subplots(1, 1, figsize = (3.4, 2.4))
        # Plot the residual phase diagram, with axes oriented correctly
        plot_res = ax.imshow(residual, origin = 'lower', extent = [-7, 3, 1, 8], cmap = 'bwr')
        # Set up and label the axes
        ax.set_xlim(-7, 3)
        ax.set_xlabel(r"$\log{(n_b \, [\mathrm{cm}^{-3}])}$")
        ax.set_ylim(1, 8)
        ax.set_ylabel(r"$\log{(T \, [\mathrm{K}])}$")
        # Initiate the colorbar
        fig.colorbar(plot_res, ax = ax, label = r'$\frac{m_\mathrm{GH12} - m_\mathrm{XGB}}{m_\mathrm{GH12} + m_\mathrm{XGB}}$')
        # Imporve the layout
        fig.tight_layout()
        # Save the figure
        fig.savefig('residual_' + str(n) + '_Myr.pdf')
        # Close the figure
        plt.close(fig)

    # To limit to most affected gas:
    if show_most_affected_gas == True:
        # Create a plot
        fig, ax = plt.subplots(1, 1, figsize = (3.4, 2.4))
        # Plot bins where residual is above threshold
        thresh = 0.95
        res_above_thresh = np.where(np.abs(residual) > thresh, 1, 0)
        # Count bins with negative residual above the threshold
        print("Under -0.95: ", len(np.argwhere(np.where(residual < -0.95, 1, 0))))
        # Count total bins above the threshold
        print("Total: ", len(np.argwhere(res_above_thresh)))
        # Count bins with |residual|=1 (phase diagram for one run is 0)
        print("+1: ", len(np.argwhere(np.where(residual == 1, 1, 0))))
        print("-1: ", len(np.argwhere(np.where(residual == -1, 1, 0))))
        # Plot bins with residual above the threshold
        plot_mag = ax.imshow(res_above_thresh, origin = 'lower', extent = [-7, 3, 1, 8], cmap = 'Greys', interpolation = "none")
        # Set up axes (same as residual plot)
        ax.set_xlim(-7, 3)
        ax.set_xlabel(r"$\log{(n_b \, [\mathrm{cm}^{-3}])}$")
        ax.set_ylim(1, 8)
        ax.set_ylabel(r"$\log{(T \, [\mathrm{K}])}$")
        fig.colorbar(plot_mag, ax = ax, label = '|Residual| > ' + str(thresh))
        fig.tight_layout()
        fig.savefig('most_affected_gas_' + str(n) + '_Myr.pdf')
        plt.close(fig)

        # Save the array with 1s where residual is above 0.95
        np.save('most_affected_gas_' + str(n) + '_Myr.npy', res_above_thresh)

    return residual

# residual_1_Myr = residual_phase_plot(1)
# residual_2_Myr = residual_phase_plot(2)
# residual_3_Myr = residual_phase_plot(3)
# residual_4_Myr = residual_phase_plot(4)
# residual_5_Myr = residual_phase_plot(5)

# Define a function to get the residual squared
def squared_residual(n_1, n_2):
    '''
    Function to compute the residual between the residuals at timesteps n_1 and n_2
    Inputs:
    n_1 (int): Number of 1 Myr timesteps after initial configuration for first residual
    n_2 (int): Number of 1 Myr timesteps after initial configuration for second residual
    Outputs:
    None (but saves a plot of the squared residual)
    '''

    # Get residuals at timesteps n_1 and n_2, without plotting
    residual_1 = residual_phase_plot(n_1, plot = False)
    residual_2 = residual_phase_plot(n_2, plot = False)
    # Get the residual between n_1 and n_2 Myrs to check convergence (residual squared!)
    residual_squared = np.array([[0 if residual_2[i][j] == 0 and residual_1[i][j] == 0
                                  else (residual_2[i][j] - residual_1[i][j])/(np.abs(residual_2[i][j]) + np.abs(residual_1[i][j]))
                                  for j in range(250)]
                                 for i in range(175)])
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize = (3.4, 2.4))
    # Plot the squared residual
    plot_res = ax.imshow(residual_squared, origin = 'lower', extent = [-7, 3, 1, 8], cmap = 'bwr')
    # Set up and label axes
    ax.set_xlim(-7, 3)
    ax.set_xlabel(r"$\log{(n_b \, [\mathrm{cm}^{-3}])}$")
    ax.set_ylim(1, 8)
    ax.set_ylabel(r"$\log{(T \, [\mathrm{K}])}$")
    # Initiate colorbar
    fig.colorbar(plot_res, ax = ax, label = r'$\delta_{%i \, \mathrm{Myr},%i \, \mathrm{Myr}}$' % (n_1, n_2))
    # Improve layout
    fig.tight_layout()
    # Save the figure
    fig.savefig('residual_squared_' + str(n_1) + '_' + str(n_2) + '_Myr.pdf')
    # Close the figure
    plt.close(fig)

squared_residual(1,2)
#squared_residual(2,3)
#squared_residual(1,3)
#squared_residual(3,4)
#squared_residual(3,5)
squared_residual(4,5)

