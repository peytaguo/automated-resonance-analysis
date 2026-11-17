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

    s_amplitude =  background( energy, m, c) + (1j / (2 * wavenumber( energy ))) * total_resonance_contribution

    return np.abs(s_amplitude) ** 2