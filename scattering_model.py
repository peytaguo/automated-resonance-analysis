import numpy as np
import matplotlib.pyplot as plt

from scipy.special import eval_legendre
from scipy.optimize import least_squares
from scipy.constants import physical_constants, hbar, electron_volt, c 

from random import randint
from itertools import product

from sklearn.cluster import AgglomerativeClustering


MU_C2 = (4*20)/(4+20) * physical_constants[
    "atomic mass constant energy equivalent in MeV"
][0] 
HBAR_C = HBAR_C = hbar * c / (1e6 * electron_volt * 1e-15)

def background(energy, m, c, E_ref):
    return m * (energy-E_ref) + c

def wavenumber(energy):
    return np.sqrt(
        2 * MU_C2 * energy 
    ) / (HBAR_C)

def phase(energy, a, b, E_ref):
    return a * (energy-E_ref) + b

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
        b, 
        E_ref
    ):
    theta = np.deg2rad(float(theta))
    return (2*angular_m+1) * p_width_ratio * ( 
        resonance_phase_factor(width, energy, resonance_energy)) * (
        np.exp(2j * phase(energy, a, b, E_ref))
        ) * eval_legendre(angular_m, np.cos(theta))


def model(
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
        total_resonance_contribution += resonance_contribution(
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

    s_amplitude =  background(energy, m, c, E_ref) + (1j / (2 * wavenumber( energy ))) * total_resonance_contribution

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
            "dsigma": dsigma_window,
            "E_min": E_min,
            "E_max": E_max
        }
    return window_all_angles 

#resonance.keys() => angular_m, p_width_ratio, width, resonance_energy, a, b

# energy, width, g, a, b <- per resonance  5 continuous parameters per resonance 
# m, c per angle - 2*16 = 32 background parameters 

"""
Order of parameters:
resonance energy, total width, width ratio, a_1, b_1,..., a_16, b_16, m_1, c_1, ..., m_16, c_16
"""

def parameters_to_dictionary(parameters, angular_momenta):
    resonance_param_lst = []

    background_parameters = np.array(parameters[-32:])
    resonances_parameters = np.array(parameters[:-32])


    angular_index = 0
    current_index = 35 
    while current_index <= len(resonances_parameters):
        extract_from = resonances_parameters[current_index- 35:current_index]
        phase_parameters = extract_from[3:].reshape(16,2)
        resonance = {
            "angular m": angular_momenta[angular_index],
            "p width ratio": extract_from[2],
            "width": extract_from[1],
            "r energy": extract_from[0],
            "a": phase_parameters[::,0],
            "b": phase_parameters[::,1] 
        }

        resonance_param_lst.append(resonance)
        current_index += 35
        angular_index += 1

    return resonance_param_lst, background_parameters.reshape(16, 2)
    

def get_residual(parameters, angular_momenta, window):

    angle = list(window.keys())[0]
    E_min = window[angle]["E_min"]
    E_max = window[angle]["E_max"]

    E_ref = (E_min + E_max) / 2

    
    residuals = []

    resonance_param_lst, background_parameters = parameters_to_dictionary(parameters, angular_momenta)

    for angle_i, ((angle, data), (m,c)) in enumerate(zip(window.items(), background_parameters)):
        assert np.all(data["dsigma"] > 0), f"Invalid uncertainty at {angle}°"
        model_pred = model(
            angle,
            data['ke'],
            resonance_param_lst,
            m,
            c,
            angle_i,
            E_ref
        )

        residual = (data["sigma"] - model_pred) / data["dsigma"]
        residuals.append(residual)

    residuals = np.concatenate(residuals)

    return residuals

    """    residual = data['sigma'] - model_pred

        mean_square_error = np.mean(residual ** 2)

        weighted_residual = np.sqrt(mean_square_error) * residual / np.sqrt(len(residual))
        
        residuals.append(weighted_residual) 

    residuals = np.concatenate(residuals)

    return residuals 
"""

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
            ] #b

        for i in range(16):
            initial_parameters += [
                random_initial.uniform(-1, 1),       # a
                random_initial.uniform(0, 2*np.pi)   # b
            ]
        

    initial_parameters += [random_initial.uniform(-1, 1) for i in range(32)]

    return initial_parameters


# implement physical bounds on parameters
# resonance_energy - 0.1 < resonance_energy < resonance_energy + 0.1
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

    for energy in energy_window:
        r_upper_bound = [
                energy+0.02, #resonance energy 
                1, #total width
                1, #width ratio
            ]

        r_lower_bound = [
                energy-0.02,
                0,
                0,
            ]
        
        r_upper_bound += [1, 2*np.pi] * 16
        r_lower_bound += [-1, 0] * 16
    
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

    angle = list(window.keys())[0]

    E_min = window[angle]["E_min"]
    E_max = window[angle]["E_max"]
    E_ref = (E_min + E_max)/2

    for i, resonance in enumerate(fitted_resonances, start=1):
        print(
            f"Resonance {i}: "
            f"E_r = {resonance['r energy']:.6f} MeV, "
            f"l = {resonance['angular m']}, "
            f"Gamma = {resonance['width']:.6f} MeV "
            f"ratio = {resonance["p width ratio"]:.6f}"
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    for i, (ax, angle) in enumerate(zip(axes.flat, window.keys())):
        m, c = fitted_backgrounds[i]

        E_min = window[angle]["ke"].min()
        E_max = window[angle]["ke"].max()

        ke_fit = np.linspace(E_min, E_max, 1000)

        sigma_fit = model(
            float(angle),
            ke_fit,
            fitted_resonances,
            m,
            c,
            i,
            E_ref
        )

        ax.errorbar(
        window[angle]["ke"],
        window[angle]["sigma"],
        yerr=window[angle]["dsigma"],
        fmt="o",
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


def get_window_clusters(cluster_energies):
    cluster_energies = np.array(cluster_energies).reshape(-1,1) #features along a row. 
    
    clustering = AgglomerativeClustering(
        n_clusters = None,
        distance_threshold=0.04,
        linkage = "single"
    )

    outcome = clustering.fit_predict(cluster_energies)

    clusters = [
        np.sort(cluster_energies[outcome == label, 0])
        for label in np.unique(outcome)
    ]

    clusters.sort(key=lambda cluster: cluster[0])

    return clusters

def merge_clusters(clusters):
    merged = []
    skip = False

    for i in range(len(clusters)-1):
        if skip:
            skip = False
            continue

        if len(clusters[i]) < 2:
            if len(clusters[i+1]) < 2:
                merged_cluster = list(clusters[i]) + list(clusters[i+1])
                merged.append(merged_cluster)
                skip = True
                continue

        merged.append(list(clusters[i]))

    if clusters and not skip:
        merged.append(list(clusters[-1]))

    return merged

"""
window_all_angles[angle] = {
            "ke": ke_window,
            "sigma": sigma_window,
            "dsigma": dsigma_window,
            "E_min": E_min,
            "E_max": E_max
        }
"""

def get_windows(all_cs, clusters):
    windows = []
    
    midpoints = [
        (max(clusters[i]) + min(clusters[i+1])) / 2
        for i in range(len(clusters) - 1)
    ]

    midpoints = [
    min(data["ke"].min() for data in all_cs.values())
] + midpoints + [
    max(data["ke"].max() for data in all_cs.values())
]

    E_mins = midpoints[:-1]
    E_maxs = midpoints[1:]

    for E_min, E_max in zip(E_mins, E_maxs):
        windows.append(get_window_cs(all_cs, E_min, E_max))

    return windows


def fitting_routine_1(window, cluster_energies, cluster_widths):
    cluster_energies = np.array(cluster_energies)
    cluster_widths = np.array(cluster_widths)

    angle = list(window.keys())[0]
    E_min = float(window[angle]["E_min"])
    E_max = float(window[angle]["E_max"])

    mask = (
    (cluster_energies >= E_min)
    & (cluster_energies <= E_max)
)

    window_energies = cluster_energies[mask]
    window_widths = cluster_widths[mask]
    
    initial_parameters = get_initial_parameters(
        window_energies, 
        window_widths,
        E_min,
        E_max
    )

    number_of_resonances = int((len(initial_parameters) - 32)/35)

    lower_bound, upper_bound = get_bounds(E_min, E_max, window_energies)

    print("Initial parameters:", np.shape(initial_parameters))
    print("Lower bounds:", np.shape(lower_bound))
    print("Upper bounds:", np.shape(upper_bound))

    fits = []

    combinations = list(product(range(6), repeat=number_of_resonances))
    total = len(combinations)

    for i, angular_momenta in enumerate(combinations, start=1):
        fit = least_squares(
        get_residual,
        x0 = initial_parameters,
        args = (angular_momenta, window),
        bounds = (lower_bound, upper_bound),
        max_nfev = 200,
        verbose = 1
    )
        fits.append((fit, angular_momenta))

        print(
        f"\rProgress: {100*i/total:.1f}% "
        f"({i}/{total})",
        end="",
        flush=True
        )
        
    fits.sort(key=lambda fit: fit[0].cost)

    return fits, lower_bound, upper_bound

def fitting_routine_2(fits, window, lower_bound, upper_bound):

    top_fits = fits[:10]

    refined_fits = []

    for i, (fit, angular_momenta) in enumerate(top_fits, start=1):

        fit = least_squares(
            get_residual,
            x0=fit.x,
            args=(angular_momenta, window),
            bounds=(lower_bound, upper_bound),
            max_nfev=3000,
            verbose=1
        )

        refined_fits.append(
            (fit.cost, angular_momenta, fit.x)
        )

        print(f"\r {100*i/len(top_fits):.1f}% complete",
            end="", flush=True)

    # Rank by refined cost
    refined_fits.sort(key=lambda fit: fit[0])

    return refined_fits

####################################################

def get_scattering_amplitude(
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
        total_resonance_contribution += resonance_contribution(
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

    s_amplitude =  background(energy, m, c, E_ref) + (1j / (2 * wavenumber( energy ))) * total_resonance_contribution

    return s_amplitude 


def get_dscattering_dm(window):
    angle = list(window.keys())[0]
    E_min = window[angle]["E_min"]
    E_max = window[angle]["E_max"]

    E_ref = (E_min + E_max)/2

    partial = []

    for angle in window.values():
        for energy in angle["ke"]:
            partial.append(
                energy - E_ref
            )

    partial = np.array(partial)

    return partial

def get_conjugate_scattering_amp(window, resonances, background_params, E_ref):
    conj_scattering_amplitude = np.array([])
    for i, ((angle, data), (m,c)) in enumerate(zip(window.items(), background_params)):
        scattering_amplitude = get_scattering_amplitude(
                    angle, 
                    data["ke"],
                    resonances,
                    m,
                    c,
                    i,
                    E_ref
                )
        
        conj_scattering_amplitude = np.append(
            conj_scattering_amplitude, np.conjugate(scattering_amplitude)
            )
    
    return conj_scattering_amplitude



def get_jacobian(parameters, angular_momenta, window):
    angle = list(window.keys())[0]
    E_min = window[angle]["E_min"]
    E_max = window[angle]["E_max"]
    E_ref = (E_min + E_max)/2

    resonances_params, background_params = parameters_to_dictionary(
        parameters, angular_momenta
    )

    
    n_parameters = len(parameters)
    n_data = 0
    for angle in window.values():
        n_data += len(angle["ke"])

    jacobian = np.zeros((n_data, n_parameters))

    #ordering of resonances:
    # Er, gamma, a_1, b_1, ..., a_16, b_16, 
    # ... (if more resonances)
    # m_1, c_1,..., m_16, c_16 

    dscattering_dm = get_dscattering_dm(window)

    conjugate_scattering_amp = get_conjugate_scattering_amp(
        window, 
        resonances_params,
        background_params,
        E_ref
    )

    start = 0

    for i, (angle, data) in enumerate(window.items()):
        end = start + len(data["ke"])

        rows = slice(start, end)
        m_col = n_parameters - 32 + 2*i
        c_col = m_col + 1

        partials_m = -2 * np.real(
            conjugate_scattering_amp[rows] * dscattering_dm[rows]
        ) / data["dsigma"]

        partials_c = -2 * np.real(
            conjugate_scattering_amp[rows]
        ) / data["dsigma"]

        jacobian[rows, m_col] = partials_m 
        jacobian[rows, c_col] = partials_c

        start = end

        

    return jacobian 

"""
 theta, 
        energy,
        resonances,
        m,
        c,
        angle,
        E_ref
"""




