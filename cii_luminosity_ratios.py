# Script to experiment with plotting luminosity ratios
# at fixed density for 6 terms contributing to CII luminosity
# Metallicity is also fixed

# Import statement
import numpy as np # Numpy for math
import matplotlib.pyplot as plt # Matplotlib for plotting
import yt # yt for data handling
from yt.units import msun # Needed units
import derived_fields # Field calculations

# Read in the data for the GH12 CHF run
ds_gh12 = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-gh12/outputs/fiducial_000305.art')
ds_xgb = yt.load('/nfs/turbo/lsa-cavestru/dbrobins/NGC300/rt-Z0.3-xgbchf-test/outputs/fiducial_000305.art')
# Extract density and temperature fields, and cell gas masses
gh12_data = ds_gh12.all_data()
xgb_data = ds_xgb.all_data()

# Get profiles of each rate, weighted by cell mass (the default weight)
gh12_e_cooling = gh12_data.profile(fields = ('gas', 'e_cooling_rate'), bin_fields = ('gas', 'baryon_number_density'))
xgb_e_cooling = xgb_data.profile(fields = ('gas', 'e_cooling_rate'), bin_fields = ('gas', 'baryon_number_density'))
gh12_a_cooling = gh12_data.profile(fields = ('gas', 'a_cooling_rate'), bin_fields = ('gas', 'baryon_number_density'))
xgb_a_cooling = xgb_data.profile(fields = ('gas', 'a_cooling_rate'), bin_fields = ('gas', 'baryon_number_density'))
# Note: for CMB emission, rate is constant, so just profiling cell mass with no weighting
gh12_cmb_emission = gh12_data.profile(fields = ('gas', 'cell_mass'), bin_fields = ('gas', 'baryon_number_density'), weight_field = None)
xgb_cmb_emission = xgb_data.profile(fields = ('gas', 'cell_mass'), bin_fields = ('gas', 'baryon_number_density'), weight_field = None)
gh12_h2_para = gh12_data.profile(fields = ('gas', 'h2_para_rate'), bin_fields = ('gas', 'baryon_number_density'))
xgb_h2_para = xgb_data.profile(fields = ('gas', 'h2_para_rate'), bin_fields = ('gas', 'baryon_number_density'))
gh12_h2_ortho = gh12_data.profile(fields = ('gas', 'h2_ortho_rate'), bin_fields = ('gas', 'baryon_number_density'))
xgb_h2_ortho = xgb_data.profile(fields = ('gas', 'h2_ortho_rate'), bin_fields = ('gas', 'baryon_number_density'))

# Get density pdf from GH12 run (the two are nearly identical, so only plot GH12 here)
n_pdf = gh12_data.profile(fields = ('gas', 'cell_mass'), weight_field = None, bin_fields = ('gas', 'baryon_number_density'), fractional = True)

# Create subplots (sized to one column)
fig, ax = plt.subplots(2, 1, sharex = True, figsize = (3.4, 4.8), gridspec_kw = {'wspace':0, 'hspace':0})
# Get default color cycle
cmap = plt.get_cmap("tab10")
# Plot dE_j/dn ratios on the upper panel
ax[0].plot(gh12_e_cooling.x, xgb_e_cooling[('gas', 'e_cooling_rate')] / gh12_e_cooling[('gas', 'e_cooling_rate')], 
           linestyle = 'solid', color = cmap(0), label = 'e')
ax[0].plot(gh12_a_cooling.x, xgb_a_cooling[('gas', 'a_cooling_rate')] / gh12_a_cooling[('gas', 'a_cooling_rate')], 
           linestyle = 'dashed', color = cmap(1), label = 'H, He')
ax[0].plot(gh12_cmb_emission.x, xgb_cmb_emission[('gas', 'cell_mass')] / gh12_cmb_emission[('gas', 'cell_mass')], 
           linestyle = 'dashdot', color = cmap(2), label = 'CMB')
ax[0].plot(gh12_h2_para.x, xgb_h2_para[('gas', 'h2_para_rate')] / gh12_h2_para[('gas', 'h2_para_rate')], 
           linestyle = 'dotted', color = cmap(3), label = r'H$_2$ para')
ax[0].plot(gh12_h2_ortho.x, xgb_h2_ortho[('gas', 'h2_ortho_rate')] / gh12_h2_ortho[('gas', 'h2_ortho_rate')], 
           linestyle = 'solid', color = cmap(4), label = r'H$_2$ ortho')
ax[0].set_xscale('log')
ax[0].set_ylabel(r'Ratio $r_j$')
ax[0].legend()
# Plot density PDF on lower panel
ax[1].plot(n_pdf.x, n_pdf[('gas', 'cell_mass')], linestyle = 'solid', color = cmap(5))
# Label log-scale axes
ax[1].set_xscale('log')
ax[1].set_xlabel(r'Baryon number density $[\mathrm{cm}^{-3}]$')
ax[1].set_yscale('log')
ax[1].set_ylabel(r'PDF')
# Set layout
ax[0].set_xlim([1e-5, 1e3])
ax[0].set_xlim([1e-5, 1e3])
ax[1].set_ylim([1e-8, 1e-3])
fig.tight_layout()
# Save the figure
fig.savefig('luminosity_ratios.pdf')
         
