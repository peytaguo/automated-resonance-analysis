import numpy as np
import matplotlib.pyplot as plt

from scipy.special import eval_legendre
from scipy.optimize import least_squares
from scipy.constants import physical_constants, hbar, electron_volt, c 


MU_C2 = (4*20)/(4+20) * physical_constants[
    "atomic mass constant energy equivalent in MeV"
][0] 
HBAR_C = HBAR_C = hbar * c / (1e6 * electron_volt * 1e-15)

def background(energy, m, c):
    return m * energy + c

def wavenumber(energy):
    return np.sqrt(
        2 * MU_C2 * energy 
    ) / (HBAR_C)

def phase(energy, a, b):
    return a * energy + b

#e^2ibeta - 1
def resonance_phase_factor(width, energy, resonance_energy):
    return ( width * 1j ) / (resonance_energy - energy - 1j * width/2)

"""
resonance disctionary:

resonance.keys() => angular_m, p_width_ratio, width, resonance_energy, a, b
"""

def resonance_contribution(
        theta,
        angular_m, 
        p_width_ratio, 
        energy, 
        width, 
        resonance_energy, 
        a, 
        b
    ):
    theta = np.deg2rad(float(theta))
    return (2*angular_m+1) * p_width_ratio * ( 
        resonance_phase_factor(width, energy, resonance_energy)) * (
        np.exp(2j * phase(energy, a, b))
        ) * eval_legendre(angular_m, np.cos(theta))


def model(
        theta, 
        energy,
        resonances,
        m,
        c
    ):

    total_resonance_contribution = 0j

    for resonance in resonances:
        total_resonance_contribution += resonance_contribution(
            theta,
            resonance["angular m"],
            resonance["p width ratio"],
            energy,
            resonance["width"],
            resonance["r energy"],
            resonance["a"],
            resonance["b"]
        )

    s_amplitude =  background(energy, m, c) + (1j / (2 * wavenumber( energy ))) * total_resonance_contribution

    return np.abs(s_amplitude) ** 2

def get_window_cs(all_cs, E_min, E_max):
    window_all_angles = {}

    for angle, data in all_cs.items():
        ke = data['ke']
        sigma = data['sigma']
        dsigma = data['dsigma']
        mask = (ke >= E_min) & (ke <= E_max)

        ke_window = ke[mask]
        sigma_window = sigma[mask]
        dsigma_window = dsigma[mask]

        window_all_angles[angle] = {
            "ke": ke_window,
            "sigma": sigma_window,
            "dsigma": dsigma_window
        }
    return window_all_angles 

#resonance.keys() => angular_m, p_width_ratio, width, resonance_energy, a, b

# energy, width, g, a, b <- per resonance  5 continuous parameters per resonance 
# m, c per angle - 2*16 = 32 background parameters 

"""
Order of parameters:
resonance energy, total width, width ratio, a, b
"""

def parameters_to_dictionary(parameters, angular_momenta):
    resonance_param_lst = []

    background_parameters = np.array(parameters[-32:])
    resonances_parameters = np.array(parameters[:-32])

    angular_index = 0
    current_index = 5 
    while current_index <= len(resonances_parameters):
        extract_from = resonances_parameters[:current_index]
        resonance = {
            "angular m": angular_momenta[angular_index],
            "p width ratio": extract_from[-3],
            "width": extract_from[-4],
            "r energy": extract_from[-5],
            "a": extract_from[-2],
            "b": extract_from[-1] 
        }

        resonance_param_lst.append(resonance)
        current_index += 5
        angular_index += 1

    return resonance_param_lst, background_parameters.reshape(16, 2)
    

def get_residual(parameters, angular_momenta, window):
    residuals = []

    resonance_param_lst, background_parameters = parameters_to_dictionary(parameters, angular_momenta)

    for (angle, data), (m,c) in zip(window.items(), background_parameters):
        model_pred = model(
            angle,
            data['ke'],
            resonance_param_lst,
            m,
            c
        )

        residual = data['sigma'] - model_pred

        mean_square_error = np.mean(residual ** 2)

        weighted_residual = np.sqrt(mean_square_error) * residual / np.sqrt(len(residual))
        
        residuals.append(weighted_residual) 

    residuals = np.concatenate(residuals)

    return residuals 


"""
Order of parameters:
resonance energy, total width, width ratio, a, b
"""

random_initial = np.random.default_rng(1)

def get_initial_parameters(
        cluster_energies, 
        r_widths,
        E_min, 
        E_max):

    initial_parameters = []

    cluster_energies = np.array(cluster_energies)
    r_widths = np.array(r_widths)

    mask = (cluster_energies >= E_min) & (cluster_energies <= E_max)

    energies_in_window = cluster_energies[mask]
    r_widths_in_window = r_widths[mask]

    for (energy, width) in zip(energies_in_window, r_widths_in_window):
        initial_parameters += [
            energy,
            width, 
            random_initial.uniform(0.1, 0.9), #width ratio
            random_initial.uniform(-1, 1), #a
            random_initial.uniform(0, 2*np.pi)] #b

    initial_parameters += [random_initial.uniform(-1, 1) for i in range(32)]

    return initial_parameters


# implement physical bounds on parameters
# E_min < resonance_energy < E_max 
# 0 < total width < E_max - E_min
# 0 < width ratio < 1
# -1 < a < 1 ??
# 0 < b < 2pi 

def get_bounds(E_min, E_max, cluster_energies):
    cluster_energies = np.array(cluster_energies)
    mask =  mask = (cluster_energies >= E_min) & (cluster_energies <= E_max)
    energy_window = cluster_energies[mask]
    

    upper_bound = []
    lower_bound = []

    background_lower_bounds = [-np.inf, -np.inf]*16
    background_upper_bounds = [np.inf, np.inf]*16


    r_upper_bound = [
        E_max, #resonance energy 
        1, #total width
        1, #width ratio
        1, 
        2*np.pi
    ]

    r_lower_bound = [
        E_min,
        0,
        0,
        -1,
        0
    ]

    for _ in energy_window:
        upper_bound += r_upper_bound
        lower_bound += r_lower_bound

    upper_bound += background_upper_bounds
    lower_bound += background_lower_bounds

    return lower_bound, upper_bound


def get_fit_visual(window, params, angular_momenta):
    fitted_resonances, fitted_backgrounds = parameters_to_dictionary(
        params,
        angular_momenta
    )

    for i, resonance in enumerate(fitted_resonances, start=1):
        print(
            f"Resonance {i}: "
            f"E_r = {resonance['r energy']:.6f} MeV, "
            f"l = {resonance['angular m']}, "
            f"Gamma = {resonance['width']:.6f} MeV"
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    for i, (ax, angle) in enumerate(zip(axes.flat, window.keys())):
        m, c = fitted_backgrounds[i]

        ke = window[angle]["ke"]
        ke_fit = np.linspace(ke.min(), ke.max(), 1000)

        sigma_fit = model(
            float(angle),
            ke_fit,
            fitted_resonances,
            m,
            c
        )

        ax.errorbar(
            ke,
            window[angle]["sigma"],
            yerr = window[angle]["dsigma"],
            fmt = "o",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color="blue",
            label="Experimental"
        )
        ax.plot(ke_fit, sigma_fit, color="red")

        ax.set_title(f"{angle}°")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross section")


    plt.show()