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



###################################################################################################

def get_midpoints(energies_lst):
    energies_lst = np.array(energies_lst)
    diff = np.diff(energies_lst)

    midpoints = energies_lst[:-1] + diff/2
    return midpoints 
            



def get_clusters_midpoints(cross_sections):
    peaks = get_all_peaks_lst(cross_sections)
    peaks.sort(
        key = lambda angle_lst: len(angle_lst[1]),
        reverse = True
    )

    angle_energies = []
    for lst in peaks:
        angle_energies.append([lst[0], lst[1]])

    current_partition = get_midpoints(np.sort(angle_energies[0][-1]))

    i = 0
   

    while i < len(cross_sections)-1:
        peaks_in_bins = [[] for i in range(len(current_partition)+1)]
        i += 1
        current_energies = angle_energies[i][-1]
        for energy in current_energies:
            index = np.searchsorted(current_partition, energy, side="right")

            peaks_in_bins[index].append(energy)

        for j in range(len(peaks_in_bins)):
            bin = peaks_in_bins[j]
            if len(bin) > 1:
                bin_midpoints = get_midpoints(np.sort(bin))
                current_partition = np.concatenate((current_partition, bin_midpoints))
                current_partition = np.sort(current_partition)

        clusters = [[] for i in range(len(current_partition)+1)]

        for angle, energies, heights in peaks:
            for energy, height in zip(energies, heights):

                index = np.searchsorted(
                    current_partition, energy, side="right"
                )

                clusters[index].append([ angle, energy, height])

        final = []
        for cluster in clusters:
            rep_energy = np.mean(np.array(cluster)[::,1])
            final.append([rep_energy, cluster])


    return final, current_partition


####################################################################################################




def get_clusters_sort(cross_sections):

    all_peaks_flattened = []

    all_peaks_lst = get_all_peaks_lst(
        cross_sections
    )

    for angle_lst in all_peaks_lst:
        for i in range(len(angle_lst[1])):
            all_peaks_flattened += [
            [angle_lst[0], angle_lst[1][i], angle_lst[2][i]]
        ]

    all_peaks_sorted = sorted(
        all_peaks_flattened, 
        key = lambda peak : peak[1]
    )

    current_cluster = []
    cluster_angles = []
    cluster_energies = []

    all_clusters = []
    clustered_energies = []

    i = 0
    while i < len(all_peaks_sorted):
        current_peak = all_peaks_sorted[i]
        current_angle = current_peak[0]

        if current_angle not in cluster_angles:
            current_cluster.append(current_peak)
            cluster_angles.append(current_angle)
            cluster_energies.append(float(current_peak[1]))
        else:
            clustered_energies.append(cluster_energies)
            all_clusters.append([np.mean(cluster_energies), current_cluster])
            current_cluster = [current_peak]
            cluster_angles = [current_angle]
            cluster_energies = [current_peak[1]]

        i += 1

    all_clusters.append([np.mean(cluster_energies), current_cluster])
    clustered_energies.append(cluster_energies)

    return all_clusters, clustered_energies



def get_clusters_hdbscan(cross_sections):
    all_peaks_flattened = []

    all_peaks_lst = get_all_peaks_lst(
        cross_sections
    )

    for angle_lst in all_peaks_lst:
        for i in range(len(angle_lst[1])):
            all_peaks_flattened += [
            [angle_lst[0], angle_lst[1][i], angle_lst[2][i]]
        ]
            
    if len(all_peaks_flattened) < 2:
        return [], all_peaks_flattened

    peak_energies = np.array([[peak[1]] for peak in all_peaks_flattened])

    model = HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.02
    )

    pred = model.fit_predict(peak_energies)

    clusters = []
    noise = []

    for label in np.unique(pred):
        cluster = [
            peak for peak, y in zip(all_peaks_flattened, pred) if y == label
        ]

        if label == -1:
            noise.extend(cluster)
        else:
            rep_energy = np.mean([p[1] for p in cluster])
            clusters.append([rep_energy, cluster])

    clusters.sort(
        key = lambda cluster: cluster[0]
    )

    return clusters, noise

###############################################################




def get_peaks_w_widths(interp_parts):
    peak_energies_all = []
    peak_heights_all = []
    peak_widths_all =[]

    for ke_part, sigma_part in interp_parts:
        peak_index, prop = find_peaks(
            sigma_part,
            prominence = 0.005,
            width = (None, None),
        )
        peak_index = np.unique(peak_index) #sometimes cwt gives the exact same index, yielding np.diff to have zeroes
        peak_energies_all.append(ke_part[peak_index])
        peak_heights_all.append(sigma_part[peak_index])

        indices = np.arange(len(ke_part))

        start = np.interp(prop["left_ips"], indices, ke_part)
        end = np.interp(prop["right_ips"], indices, ke_part)

        peak_widths_all.append(abs(start - end))
        
    return np.concatenate(peak_energies_all), np.concatenate(peak_heights_all), np.concatenate(peak_widths_all)

def get_all_peaks_lst_w_widths(cross_sections):
    angle_peaks = []

    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )
        peak_energies, peak_heights, peak_widths = get_peaks_w_widths(interp_parts)

        angle_peaks.append(
            [float(angle), peak_energies, peak_heights, peak_widths]
        )
        
    return angle_peaks

def peak_metric(peak_1, peak_2):
    return 2 * abs(peak_1[0] - peak_2[0]) / (peak_1[-1] + peak_2[-1])

def get_clusters_linkage(cross_sections):
    all_peaks_flattened = []

    all_peaks_lst = get_all_peaks_lst_w_widths(
        cross_sections
    )

    for angle_lst in all_peaks_lst:
        for i in range(len(angle_lst[1])):
            all_peaks_flattened += [
            [angle_lst[0], angle_lst[1][i], angle_lst[2][i], angle_lst[3][i]]
        ]

    peaks_w_widths = np.array([
        [peak[1], peak[3]] for peak in all_peaks_flattened
    ])

    distances = pdist(peaks_w_widths, metric = peak_metric)

    l = linkage(distances, method="complete")

    labels = fcluster(l, t=2.5, criterion="distance")

    clusters = []
    for label in np.unique(labels):
        cluster = [
                peak[:-1] for peak, y in zip(all_peaks_flattened, labels) if y == label
            ]
        rep_energy = np.mean([p[1] for p in cluster])
        clusters.append([rep_energy, cluster])

    clusters.sort(
            key = lambda cluster: cluster[0]
        )
        
    return clusters

##############################################################################################################




#compare the innerproducts of heights of peaks to the lengendre polnomials squared 
# after normalisation. Compare the scores. 

def get_angular_momentum(cluster, trial_lst):
    all_scores = []
    elements_info = np.array(cluster[-1])

    
    angles = np.radians(elements_info[::,1].astype(float))
    cos_angle = np.cos(angles)


    #normalise so that magnitudes do not influence comparison

    heights = elements_info[::,-1].astype(float)

    for trial_l in trial_lst:
        l_poly = eval_legendre(trial_l, cos_angle)**2

        scale = np.dot(heights, l_poly) / np.dot(l_poly, l_poly)
        l_poly_scaled = scale * l_poly

        err = np.mean(
            ((heights - l_poly_scaled)/np.mean(heights) )** 2
        )
        all_scores.append([trial_l, round(err, 3)])

        all_scores.sort(
            key = lambda score: score[1]
        )
    
    return all_scores, all_scores[:5]

def get_all_angular_momentum(clusters, trial_lst):
    predictions = []
    for cluster in clusters:
        _, max = get_angular_momentum(cluster, trial_lst)
        predictions.append((float(round(cluster[0], 5)), max))

    return predictions

################# completely different clustering approach