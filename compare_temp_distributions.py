# Calculate temperature quantiles in density bins

# Import statements                                                                                                                      
import yt # yt for simulation data analysis
from yt.units import gram, second, erg, K,  centimeter # needed units    
import derived_fields # Calculated yt fields                                                             
import matplotlib.pyplot as plt # Matplotlib for plotting
import numpy as np # Numpy for math

# Set number of T and n bins
temp_bins = 175
den_bins = 200

# Read in the GH12 phase diagram
ds_gh12 = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-gh12/outputs/fiducial_000305.art')
# Extract density and temperature fields, and cell gas masses
gh12_data = ds_gh12.all_data()
den_gh12 = gh12_data[('gas', 'baryon_number_density')]
temp_gh12 = gh12_data[('gas', 'temperature')]
cell_mass_gh12 = gh12_data[('gas', 'cell_mass')]
# Compute the 2D phase diagram, weighted by gas mass
# Normalize with the total gas mass
hist_gh12 = np.histogram2d(np.log10(temp_gh12), np.log10(den_gh12), bins = [temp_bins, den_bins],
                           range = [[1, 8], [-7, 3]], weights = cell_mass_gh12, density = False)[0] / np.sum(cell_mass_gh12.value)
# Do the same for the RAG24 CHF run
ds_xgb = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-xgbchf-test/outputs/fiducial_000305.art')
xgb_data = ds_xgb.all_data()
den_xgb = xgb_data[('gas', 'baryon_number_density')]
temp_xgb = xgb_data[('gas', 'temperature')]
cell_mass_xgb = xgb_data[('gas', 'cell_mass')]
hist_xgb = np.histogram2d(np.log10(temp_xgb), np.log10(den_xgb), bins = [temp_bins, den_bins],
                          range = [[1, 8], [-7, 3]], weights = cell_mass_xgb, density = False)[0] / np.sum(cell_mass_xgb.value)

# Get array of temperature bins
temp_array = np.linspace(1, 8, num = temp_bins)
den_array = np.linspace(-7, 3, num = den_bins)
# Make arrays for median, 25th-75th, and 10th-90th percentiles for both runs
median_gh12 = np.zeros(den_bins)
p25_gh12 = np.zeros(den_bins)
p75_gh12 = np.zeros(den_bins)
p10_gh12 = np.zeros(den_bins)
p90_gh12 = np.zeros(den_bins)
median_xgb = np.zeros(den_bins)
p25_xgb = np.zeros(den_bins)
p75_xgb = np.zeros(den_bins)
p10_xgb = np.zeros(den_bins)
p90_xgb = np.zeros(den_bins)
# Loop through density bins
for den_index in range(den_bins):
    # For GH12
    # Get histogram at this density bin
    temp_hist_gh12 = hist_gh12[:, den_index]
    # Take cumulative sum
    unnormed_cdf_gh12 = np.cumsum(temp_hist_gh12)
    # Normalize it
    normed_cdf_gh12 = unnormed_cdf_gh12 / np.max(unnormed_cdf_gh12)
    # Calculate percentiles and place in appropriate arrays
    median_gh12[den_index] = temp_array[np.searchsorted(normed_cdf_gh12, 0.5)]
    p25_gh12[den_index] = temp_array[np.searchsorted(normed_cdf_gh12, 0.25)]
    p75_gh12[den_index] = temp_array[np.searchsorted(normed_cdf_gh12, 0.75)]
    p10_gh12[den_index] = temp_array[np.searchsorted(normed_cdf_gh12, 0.1)]
    p90_gh12[den_index] = temp_array[np.searchsorted(normed_cdf_gh12, 0.9)]
    
    # Do the same thing for XGB run
    temp_hist_xgb = hist_xgb[:, den_index]
    unnormed_cdf_xgb = np.cumsum(temp_hist_xgb)
    normed_cdf_xgb = unnormed_cdf_xgb / np.max(unnormed_cdf_xgb)
    median_xgb[den_index] = temp_array[np.searchsorted(normed_cdf_xgb, 0.5)]
    p25_xgb[den_index] = temp_array[np.searchsorted(normed_cdf_xgb, 0.25)]
    p75_xgb[den_index] = temp_array[np.searchsorted(normed_cdf_xgb, 0.75)]
    p10_xgb[den_index] = temp_array[np.searchsorted(normed_cdf_xgb, 0.1)]
    p90_xgb[den_index] = temp_array[np.searchsorted(normed_cdf_xgb, 0.9)]

# Make a plot
# Set up subplots, with the first 4 times taller
fig, ax = plt.subplots(2, 1, sharex = True, gridspec_kw = {'height_ratios': [4, 1], 'wspace':0, 'hspace':0}, figsize = (3.4, 3))
# GH12 plotting:
# Plot median temperature as a line
ax[0].plot(den_array, median_gh12, color = 'steelblue', linestyle = 'solid', label = 'GH12')
# Shade between 25th and 75th percentiles in a darker color
ax[0].fill_between(den_array, p25_gh12, p75_gh12, color = 'steelblue', alpha = 0.5)
# Shade between 10th and 90th percentiles in a lighter color
ax[0].fill_between(den_array, p10_gh12, p90_gh12, color = 'steelblue', alpha = 0.2)
# Plot the same quantities for XGB run
ax[0].plot(den_array, median_xgb, color = 'darkorange', linestyle = 'dashed', label = 'XGB')
ax[0].fill_between(den_array, p25_xgb, p75_xgb, color = 'darkorange', alpha = 0.5)
ax[0].fill_between(den_array, p10_xgb, p90_xgb, color = 'darkorange', alpha = 0.2)
# Label y axis
ax[0].set_ylabel(r"$\log{(T \, [\mathrm{K}])}$")
# Include a legend
ax[0].legend()
# Plot the difference in log(T_median) in a smaller panel
ax[1].plot(den_array, median_xgb - median_gh12, color = 'black')
# Add a reference horizontal dotted line at 0 (median temperatures equal)
ax[1].plot(den_array, np.zeros(len(den_array)), color = 'grey', linestyle = 'dotted')
# Label y axis, using that log(T_1) - log(T_2) = log(T_1/T_2)
ax[1].set_ylabel(r"$\log{(T_\mathrm{XGB}/T_\mathrm{GH12})}$")
# Label and configure x axis
ax[1].set_xlabel(r"$\log{(n_b \, [\mathrm{cm}^{-3}])}$")
ax[1].set_xlim([-5, 2.5])
fig.tight_layout()
# Save the plot
fig.savefig('temp_distributions.pdf')


