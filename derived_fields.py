# Module to store the needed derived fields

# Import statements
import yt # yt for handling the simulation data
from unyt import unyt_array # yt unit handling
from yt.units import gram, second, erg, K, centimeter # needed units
import numpy as np # Numpy for math

# Some useful constants
proton_mass = 1.67262192369e-24*gram #proton mass in grams
k_boltz=1.3807e-16*erg/K #Boltzmann's constant k in erg/K

# Get baryon number density from gas density/proton mass (neglecting electron mass, proton-neutron mass difference)  
def rho_to_n_b(field, data):
    return data[("gas", "density")]/proton_mass
yt.add_field(('gas', 'baryon_number_density'), function=rho_to_n_b, sampling_type = 'local', units='1/cm**3')
# Get T/mu from thermal energy density U = 1.5kT*rho/(mu*m_p) = 1.5kT * n_b/mu
def get_T_over_mu(field, data):
    return data[("gas", "thermal_energy_density")] * proton_mass / (1.5 * k_boltz * data[("gas", "density")])
yt.add_field(("gas", "T_per_mu"), function = get_T_over_mu, sampling_type = 'local', units = 'K')
# Use mean molecular mass field to convert T/mu to T:
def get_T(field, data):
    return data[("gas", "T_per_mu")] * data[("gas", "mean_molecular_weight")]
yt.add_field(("gas", "temperature"), function = get_T, sampling_type = 'local', units = 'K')

# CII emission rates (up to dependence on temperature and constants as will take ratio)
# Source: https://github.com/cavestruz/croc_CII/blob/master/tools/derived_field_CII.py#L142C1-L157C1
# Electron cooling rate
def e_cooling_rate(field, data):
    return np.exp(-91.2 * K / data[('gas', 'temperature')]) / np.sqrt(data[('gas', 'temperature')] / K) * erg * centimeter ** 3 / second
yt.add_field(("gas", "e_cooling_rate"), function = e_cooling_rate, sampling_type = 'local', units = 'erg*cm**3/s')
# Atomic cooling rate
def a_cooling_rate(field, data):
    x = np.max(16 + 0.344 * np.sqrt(data[('gas', 'temperature')]/ K) - 47.7 * K / data[('gas', 'temperature')], 0)
    return np.exp(-91.2 * K / data[('gas', 'temperature')]) * x * erg * centimeter ** 3 / second
yt.add_field(("gas", "a_cooling_rate"), function = a_cooling_rate, sampling_type = 'local', units = 'erg*cm**3/s')
# H2 para mode
def h2_para_rate(field, data):
    return unyt_array(pow(data[('gas', 'temperature')].to('K').to_ndarray() / (100), 0.124 - 0.018 * np.log(data[('gas', 'temperature')].to('K').to_ndarray() / (100)))) * erg * centimeter ** 3 / second
yt.add_field(('gas', 'h2_para_rate'), function = h2_para_rate, sampling_type = 'local', units = 'erg*cm**3/s')
# H2 ortho mode
def h2_ortho_rate(field, data):
    return unyt_array(pow(data[('gas', 'temperature')].to('K').to_ndarray() / (100), 0.095 + 0.023 * np.log(data[('gas', 'temperature')].to('K').to_ndarray() / (100)))) * erg * centimeter ** 3 / second
yt.add_field(('gas', 'h2_ortho_rate'), function = h2_ortho_rate, sampling_type = 'local', units = 'erg*cm**3/s')