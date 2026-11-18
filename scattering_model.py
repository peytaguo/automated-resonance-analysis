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
    theta = np.deg2rad(theta)
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
        mask = (ke >= E_min) & (ke <= E_max)

        ke_window = ke[mask]
        sigma_window = sigma[mask]

        window_all_angles[angle] = {
            "ke": ke_window,
            "sigma": sigma_window
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

    background_parameters = parameters[-32:]
    resonances_parameters = parameters[:-32]

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
        residuals.append(residual)

    residuals = np.concatenate(residuals)

    return residuals 
