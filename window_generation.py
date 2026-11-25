import scattering_model

import numpy as np

def group_resonance_energies(cluster_energies, cluster_widths, threshold):
    cluster_energies = np.array(cluster_energies)
    cluster_widths = np.array(cluster_widths)

    sorted_index = np.argsort(cluster_energies)
    cluster_energies = cluster_energies[sorted_index]
    cluster_widths = cluster_widths[sorted_index]

    separation = np.diff(cluster_energies)
    widths_start = cluster_widths[:-1]
    widths_end = cluster_widths[1:]

    size_of_separation = 2 * separation / (widths_start + widths_end)

    make_split = np.where(size_of_separation > threshold)[0] + 1

    energies_grouped = np.split(cluster_energies, make_split)
    widths_grouped = np.split(cluster_widths, make_split)

    #make boundaries at midpoint between the ends of the groups

    boundaries = (cluster_energies[make_split - 1] + cluster_energies[make_split])/2

    return list(zip(energies_grouped, widths_grouped)), boundaries 

