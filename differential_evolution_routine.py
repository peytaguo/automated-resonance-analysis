import scattering_model

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from matplotlib import pyplot as plt

"""def de_background(m, c, E_ref, energy):
    return m * (energy - E_ref) + c"""

"""def de_model(
        theta, 
        energy,
        resonances,
        m,
        c,
        angle,
        E_ref
    ):

    total_resonance_contribution = 0j

    for resonance in resonances:
        total_resonance_contribution += scattering_model.resonance_contribution(
            theta,
            resonance["angular m"],
            resonance["p width ratio"],
            energy,
            resonance["width"],
            resonance["r energy"],
            resonance["a"][angle],
            resonance["b"][angle],
            E_ref
        )

    s_amplitude =  de_background(m, c, E_ref, energy) + (1j / (2 * scattering_model.wavenumber( energy ))) * total_resonance_contribution

    return np.abs(s_amplitude) ** 2"""


def get_de_residual(parameters, angular_momenta, window):

    angle = list(window.keys())[0]
    E_min = window[angle]["E_min"]
    E_max = window[angle]["E_max"]

    E_ref = (E_max + E_min)/2
    
    residuals = []

    resonance_param_lst, background_parameters = scattering_model.parameters_to_dictionary(parameters, angular_momenta)

    for angle_i, ((angle, data), (m,c)) in enumerate(zip(window.items(), background_parameters)):
        assert np.all(data["dsigma"] > 0), f"Invalid uncertainty at {angle}°"
        model_pred = scattering_model.model(
            angle,
            data['ke'],
            resonance_param_lst,
            m,
            c,
            angle_i,
            E_ref
        )

        weights = hann_weights(data["ke"], E_min, E_max)

        residual = (
            np.sqrt(weights)
            * (data["sigma"] - model_pred)
            / data["dsigma"]
        )

        residuals.append(residual)

    return np.concatenate(residuals)


def cost_function(
        parameters, 
        angular_momenta,
        window
):
    residuals = get_de_residual(parameters, angular_momenta, window)
    return np.sum(residuals**2)

def get_de_bounds(
        E_min, E_max, cluster_energies, m_cap, c_cap
):
    lower_bound, upper_bound = scattering_model.get_bounds(
        E_min,
        E_max,
        cluster_energies
    )

    lower_bound[-32:] = [-m_cap, -c_cap] * 16
    upper_bound[-32:] = [m_cap, c_cap] * 16

    return list(zip(lower_bound, upper_bound))

def hann_weights(energy, E_min, E_max, edge_weight=0.5):
    x = (energy - E_min) / (E_max - E_min)

    return edge_weight + (1 - edge_weight) * np.sin(np.pi*x)**2


def de_fitting_routine(window, cluster_energies, angular_momenta, m_cap, c_cap, maxiter=40):
    angle = list(window.keys())[0]
    E_min = float(window[angle]["E_min"])
    E_max = float(window[angle]["E_max"])

    cluster_energies = np.array(cluster_energies)

    mask = (cluster_energies >= E_min) & (cluster_energies <= E_max)
    window_energies = cluster_energies[mask]

    bounds = get_de_bounds(
        E_min, 
        E_max, 
        window_energies,
        m_cap,
        c_cap
    )

    result = differential_evolution(
        cost_function,
        bounds = bounds,
        args = (angular_momenta, window),
        strategy = "best1bin",
        maxiter = maxiter,
        popsize = 8,
        mutation = (0.5, 1),
        recombination = 0.7,
        polish = False,
        rng = np.random.default_rng(1),
        disp = True
    )

    return result, bounds

def get_de_fit_visual(window, parameters, angular_momenta):
    r_params, background = scattering_model.parameters_to_dictionary(
        parameters, angular_momenta
    ) 

    angle_1 = list(window.keys())[0]
    E_min = window[angle_1]["E_min"]
    E_max = window[angle_1]["E_max"]
    E_ref = (E_min + E_max) / 2

    fig, axes = plt.subplots(4,4, figsize = (16, 12))

    for i, (ax, angle) in enumerate(zip(axes.flat, window.keys())):
        m, c = background[i]

        data = window[angle]
        ke_fit = np.linspace(data["ke"].min(), data["ke"].max(), 1000)

        fit = scattering_model.model(
            angle,
            ke_fit, 
            r_params,
            m,
            c,
            i,
            E_ref 
        )

        ax.errorbar(
            data["ke"],
            data["sigma"],
            yerr=data["dsigma"],
            fmt="o",
            markersize=3,
            capsize=3,
            color="blue",
            label="Experimental"
        )

        ax.plot(ke_fit, fit, color="red", label="DE fit")

        ax.set_title(f"{angle}°")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross section")
        

    plt.tight_layout()
    plt.show()

 