import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks_cwt, find_peaks

# each energy starts as its own cluster
# compute distances between clusters
# merge clusters with the min distance subject to
# - the difference is sufficiently small
# - each cluster contains at most one element from each array
# repeat until no more merges are possible

def get_peaks_cwt(interp_parts):
    peak_energies_all = []
    peak_heights_all = []

    for ke_part, sigma_part in interp_parts:
        peak_index = find_peaks_cwt(
            sigma_part,
            widths = np.arange(1,50),
            min_length = 3,
            min_snr = 1
        )

        peak_index = np.unique(peak_index) #sometimes cwt gives the exact same index, yielding np.diff to have zeroes
        peak_energies_all.append(ke_part[peak_index])
        peak_heights_all.append(sigma_part[peak_index])
    return np.concatenate(peak_energies_all), np.concatenate(peak_heights_all)


def visualise_peaks_cwt(interp_peaks, angle):
    plt.figure(figsize=(15, 6))
    for ke_part, sigma_part in interp_peaks:

        peak_index = find_peaks_cwt(
            sigma_part,
            widths = np.arange(1,10),
            min_length = 3,
            min_snr = 1
        )

        plt.plot(ke_part, sigma_part)

        plt.scatter(
            ke_part[peak_index],
            sigma_part[peak_index],
            marker="x",
            s=60
        )

    plt.xlabel("Energy")
    plt.ylabel(r"Cross section ({}$^\circ$)".format(angle))
    plt.show()
    

def get_threshold(all_peaks):
    min_dist_per_angle = []

    for angle in all_peaks:
        energies = np.sort(all_peaks[angle]["energy"])

        if len(energies) > 1:
            spacings = np.diff(energies)
            min_dist_per_angle.append(np.min(spacings))
            #print(min_dist_per_angle)

    median_min_dist = np.median(min_dist_per_angle)
    #print(median_min_dist)
    return median_min_dist

def get_distance_matrix(clusters, threshold):
    n = len(clusters)

    distance_matrix = np.full(
        (n, n),
        np.inf
    )

    for i in range(n):
        for j in range(i+1, n):
            angles_i = {angle for _, angle, _ in clusters[i][1]}
            angles_j = {angle for _, angle, _ in clusters[j][1]}

            all = clusters[i][1] + clusters[j][1]
            energies = [energy for energy, _, _ in all]
            spread = np.max(energies) - np.min(energies)
            

            disjoint = angles_i.isdisjoint(angles_j)
            #distance = abs(clusters[i][0] - clusters[j][0])

            if disjoint and spread < threshold:
                distance_matrix[i, j] = abs(
                    clusters[i][0] - clusters[j][0]
                )
            else:
                #dont compute distance if merging isnt valid
                distance_matrix[i,j] = np.inf 
    
    c1, c2 = np.unravel_index(
        np.argmin(distance_matrix), distance_matrix.shape
    )

    terminate = np.all(np.isinf(distance_matrix))

    return c1, c2, terminate #returns the index of the clusters to be merged 

def merge_clusters(index_1, index_2, clusters):
    cluster_1 = clusters[index_1]
    cluster_2 = clusters[index_2]

    all_clusters = cluster_1[1] + cluster_2[1]

    energies = np.array([
        energy for energy, _, _ in all_clusters
    ])

    heights = np.array([
        height for _, _, height in all_clusters
    ])

    energy_representative = np.average(
        energies,
        weights=heights
    )
    new_cluster = [energy_representative, all_clusters]

    for index in sorted([index_1, index_2], reverse=True):
        clusters.pop(index)

    clusters.append(new_cluster)

    return clusters

def get_clusters(all_peaks):
     #each cluster in clusters come in the form of [representative, [(energy, angle)]]
    threshold = get_threshold(all_peaks)
    clusters = []
     #initialise each element as its own cluster 

    for angle in all_peaks.keys():
        energies = all_peaks[angle]["energy"]
        heights = all_peaks[angle]["height"]

        for energy, height in zip(energies, heights):
            clusters.append([
                energy,
                [(energy, angle, height)]
            ])
    
    terminate = False
    
    while not terminate:
        c1, c2, terminate = get_distance_matrix(clusters, threshold)
       
        if terminate:
            break

        clusters = merge_clusters(c1, c2, clusters)

    return clusters

def clusters_remove_noise(clusters, min_cluster_items):
    final_clusters = [cluster for cluster in clusters 
                          if len(cluster[1]) >= min_cluster_items]
    return final_clusters



def plot_clusters(clusters, trial_l):

    for cluster in clusters:
        representative_energy = cluster[0]

        members = cluster[1]

        angles = [float(angle) for _, angle, _ in members]
        cos_theta = np.cos(np.radians(angles))
        
        heights = np.array([height for _, _, height in members])
        heights = heights/np.max(heights)

        if trial_l != 0:

            zeroes, _ = roots_legendre(trial_l)

            p = legendre(trial_l)**2
            dp = p.deriv(1)   
            ddp = p.deriv(2)

            stationary = dp.roots
            #print(stationary)
            local_maxima = stationary[ddp(stationary) < 0]
            #print(ddp(stationary))

            #green: zeroes
            #red: maximas
        
            for zero in zeroes:
                plt.axvline(zero, linestyle='--', color = 'green')

            for maximum in local_maxima:
                plt.axvline(maximum, linestyle='--', color = 'red')

            plt.axvline(-1, linestyle='--', color = 'red')
            plt.axvline(1, linestyle='--', color = 'red')

        plt.scatter(
                cos_theta, 
                heights, 
                s=100,
                marker='x',
        )

        #x = np.linspace(-1,1,500)

        #legendre_polynomial = eval_legendre(trial_l, x)**2
        #legendre_polynomial = legendre_polynomial/np.max(legendre_polynomial)

        """plt.plot(
            x,
            legendre_polynomial,
        )"""

        plt.xlabel(r"$\cos\theta$")
        plt.ylabel("Normalised peak height")
        plt.title(r"{}".format(representative_energy))
        plt.legend()
        plt.show()