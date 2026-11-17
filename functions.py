from scipy.signal import find_peaks_cwt, find_peaks
from matplotlib import pyplot as plt
from scipy.special import eval_legendre
from scipy.special import roots_legendre
from scipy.special import legendre

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.cluster import HDBSCAN

import numpy as np 

def cross_sections(angle, theta_arr, ke_arr, sigma_arr, dsigma_arr, e_offset):
    angle_mask = theta_arr == angle

    ke_w_mask = ke_arr[angle_mask] 
    sigma_w_mask = sigma_arr[angle_mask]
    dsigma_w_mask = dsigma_arr[angle_mask]

    return ke_w_mask+ e_offset, sigma_w_mask, dsigma_w_mask

def get_all_cs(theta_arr, ke_arr, sigma_arr, dsigma_arr, e_offset = 0):
    unique_angles = np.unique(theta_arr)

    cs_by_angle = {}

    for angle in unique_angles:
        ke, sigma, dsigma = cross_sections(
            angle,
            theta_arr, 
            ke_arr,
            sigma_arr,
            dsigma_arr,
            e_offset
        )

        cs_by_angle[f"{angle}"] = {
            "ke" : ke,
            "sigma" : sigma,
            "dsigma" : dsigma,
        }

    return cs_by_angle

# interpolation function that takes the cross section data and 
# avoids interpolating gaps in the data
# requires some decision on what the maximum separation is between 
# data points for the separation to be considered a break/gap

def get_interpolation(ke_arr, sigma_arr, gap_size = 0.1):
    ke_arr = np.array(ke_arr)
    sigma_arr = np.array(sigma_arr)

    sort = np.argsort(ke_arr)
    
    ke_sorted = ke_arr[sort]
    ke_diff = np.diff(ke_sorted)

    sigma_sorted = sigma_arr[sort]

    no_interp = np.where(ke_diff > gap_size)[0] 
    
    ke_parts = np.split(ke_sorted, no_interp+1)
    sigma_parts = np.split(sigma_sorted, no_interp+1)

    interp_parts = []

    ke_stitched = []
    sigma_stitched = []

    #print(len(ke_parts[0]))

    for i in range(len(ke_parts)):
        if len(ke_parts[i]) < 2:
            continue # nothing to interpolate for one segment consisting of 1 data point.
        else:
            #want the spacing to be approximately the resolution of the data 

            spacing = np.median(np.diff(ke_parts[i])) 
            ke_interp = np.arange(
                        ke_parts[i][0],
                        ke_parts[i][-1] + spacing/5,
                        spacing
                    )

            sigma_interp = np.interp(
                        ke_interp,
                        ke_parts[i], 
                        sigma_parts[i]
                    )

            interp_parts.append(
                [ke_interp, sigma_interp]
            )
        ke_stitched.extend(ke_parts[i])
        sigma_stitched.extend(sigma_parts[i])

        #stiched only for visualisng purposes
        ke_stitched.append(np.nan) 
        sigma_stitched.append(np.nan)

    ke_stitched = np.array(ke_stitched)
    sigma_stitched = np.array(sigma_stitched)

    return interp_parts, ke_stitched, sigma_stitched


def get_peaks_lm(interp_parts):
    peak_energies_all = []
    peak_heights_all = []

    for ke_part, sigma_part in interp_parts:
        peak_index, _ = find_peaks(
            sigma_part,
            prominence = 0.005
        )
        peak_index = np.unique(peak_index) #sometimes cwt gives the exact same index, yielding np.diff to have zeroes
        peak_energies_all.append(ke_part[peak_index])
        peak_heights_all.append(sigma_part[peak_index])
    return np.concatenate(peak_energies_all), np.concatenate(peak_heights_all)

def visualise_peaks_lm(interp_peaks, angle):
    plt.figure(figsize=(15, 6))
    for ke_part, sigma_part in interp_peaks:

        peak_index, _ = find_peaks(
            sigma_part,
            prominence = 0.005
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
    plt.yscale("log")
    plt.show()
    

def get_all_peaks(cross_sections):
    angle_peaks = {}

    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )

        peak_energies, peak_heights = get_peaks_lm(interp_parts)

        angle_peaks[angle] = {
            "energy": peak_energies,
            "height": peak_heights
        }
        
    return angle_peaks

def visualise_all_peaks(cross_sections):
    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )

        visualise_peaks_lm(interp_parts, angle)

###################################################################################################

def get_midpoints(energies_lst):
    energies_lst = np.array(energies_lst)
    diff = np.diff(energies_lst)

    midpoints = energies_lst[:-1] + diff/2
    return midpoints 
            

def get_all_peaks_lst(cross_sections):
    angle_peaks = []

    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )
        peak_energies, peak_heights = get_peaks_lm(interp_parts)

        angle_peaks.append(
            [float(angle), peak_energies, peak_heights]
        )
        
    return angle_peaks

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


def visualise_clusters(clusters, trial_l):

    for cluster in clusters:
        rep_energy = cluster[0]
        cluster = np.array(cluster[-1])

        angles = cluster[::,0]
        cos_theta = np.cos(np.radians(angles))
        
        heights = cluster[::,-1]

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
        plt.ylabel("Peak height")
        plt.yscale("log")
        plt.title(r"{}".format(rep_energy))
        plt.show()




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

def cost_function(cluster, repeat_angle_p, new_cluster_p):

    cluster = np.array(cluster)
    cluster_energies = cluster[::,1]
    mean_energy = np.mean(cluster_energies)
    spread_cost = np.sum((cluster_energies - mean_energy)**2)

    cluster_angles = cluster[::,0]
    n_unique_angles = len(np.unique_counts(cluster_angles))
    repeating_angles = len(cluster_angles) - n_unique_angles 

    repeating_angle_cost = repeat_angle_p * repeating_angles

    total_cost = spread_cost + repeating_angle_cost + new_cluster_p

    return total_cost

def get_clusters_dp(cross_sections, repeat_angle_p, new_cluster_p):

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

    index = [0 for i in range(len(all_peaks_sorted)+1)]

    costs = [np.inf for i in range(len(all_peaks_sorted)+1)]
    costs[0] = 0

    for i in range(1, len(all_peaks_sorted) + 1):

        for j in range(i):

            last_cluster = all_peaks_sorted[j:i]

            last_cluster_cost = cost_function(
                last_cluster,
                repeat_angle_p,
                new_cluster_p
            )

            total_cost = costs[j] + last_cluster_cost

            if total_cost < costs[i]:
                costs[i] = total_cost
                index[i] = j

    clusters = []
    i = len(all_peaks_sorted)
    while i != 0:
        l_cluster_start_index = index[i]
        last_cluster = all_peaks_sorted[l_cluster_start_index:i]
        rep_energy = np.mean(np.array(last_cluster)[::,1])
        i = l_cluster_start_index
        clusters.append([rep_energy, last_cluster])

    clusters.reverse()
    return clusters

#fix energy but vary angle 

def get_height_at_energy(ke_arr, sigma_arr, target_energy, gap_size=0.1):
    ke_arr = np.array(ke_arr)
    sigma_arr = np.array(sigma_arr)


    splits = np.where(np.diff(ke_arr) > gap_size)[0] + 1

    ke_parts = np.split(ke_arr, splits)
    sigma_parts = np.split(sigma_arr, splits)

    for ke, sigma in zip(ke_parts, sigma_parts):
        if len(ke) >= 2 and ke[0] <= target_energy <= ke[-1]:
            return np.interp(target_energy, ke, sigma)

    return np.nan
        
def get_all_height(all_cs, target_energy):
    height_across_angle = []
    for angle in all_cs.keys():
        ke_arr = all_cs[angle]['ke']
        sigma_arr = all_cs[angle]['sigma']

        height_across_angle.append(
            [float(angle), get_height_at_energy(ke_arr, sigma_arr, target_energy)]
        )
    return height_across_angle

def visualise_height_fix_energy(all_cs, target_energies):
    for target_energy in target_energies:
        heights_lst = np.array(get_all_height(all_cs, target_energy))
        angles = heights_lst[::,0]
        heights = heights_lst[::,-1]

        plt.figure(figsize=(15, 6))
        plt.scatter(angles, heights)

        plt.xlabel("angle")
        plt.ylabel("cross section")
        plt.title(target_energy)
        plt.yscale("log")
    plt.show()
    
def get_dp_clusters_info(clusters):
    number_of_clusters = len(clusters)
    length_of_clusters = np.array([
        len(cluster[-1]) for cluster in clusters
    ])
    cluster_sizes, how_many = np.unique(
        length_of_clusters,
        return_counts = True
    )

    print("number of clusters: {}".format(number_of_clusters))
    #cluster_sizes.tolist()
    #how_many.tolist()
    for cluster_size, n in zip(cluster_sizes, how_many):
        print("{} clusters of size {}".format(n, cluster_size))
    
    n_repeated_clusters = sum(
    len(set(peak[0] for peak in cluster[1])) < len(cluster[1])
    for cluster in clusters
)

    print("Clusters with repeated angles:", n_repeated_clusters)

    print("===========================")
