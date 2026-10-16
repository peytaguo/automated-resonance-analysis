from scipy.signal import find_peaks_cwt
from matplotlib import pyplot as plt
from scipy.special import eval_legendre

import numpy as np 
import heapq

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

            spacing = np.mean(np.diff(ke_parts[i])) 
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


def get_peaks(interp_parts):
    peak_energies_all = []
    peak_heights_all = []

    for ke_part, sigma_part in interp_parts:
        peak_index = find_peaks_cwt(
            sigma_part,
            widths = np.arange(1, 50),
            min_length = 3,
            min_snr = 1
        )
        peak_energies_all.append(ke_part[peak_index])
        peak_heights_all.append(sigma_part[peak_index])
    return np.concatenate(peak_energies_all), np.concatenate(peak_heights_all)

def visualise_peaks(interp_peaks, angle):
    for ke_part, sigma_part in interp_peaks:

        peak_index = find_peaks_cwt(
            sigma_part,
            np.arange(1, 50),
            min_length=3,
            min_snr=1
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
    

def get_all_peaks(cross_sections):
    angle_peaks = {}

    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )

        peak_energies, peak_heights = get_peaks(interp_parts)

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
        visualise_peaks(interp_parts, angle)

# each energy starts as its own cluster
# compute distances between clusters
# merge clusters with the min distance subject to
# - the difference is sufficiently small
# - each cluster contains at most one element from each array
# repeat until no more merges are possible

def get_threshold(all_peaks):
    min_dist_per_angle = []

    for angle in all_peaks:
        energies = np.sort(all_peaks[angle]["energy"])

        if len(energies) > 1:
            spacings = np.diff(energies)
            min_dist_per_angle.append(np.min(spacings))

    median_min_dist = np.median(min_dist_per_angle)
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

            disjoint = angles_i.isdisjoint(angles_j)
            distance = abs(clusters[i][0] - clusters[j][0])

            if disjoint and distance < threshold:
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

    energy_representative = np.mean([energy for energy, _ , _ in all_clusters])

    new_cluster = [energy_representative, all_clusters]

    for index in sorted([index_1, index_2], reverse=True):
        clusters.pop(index)

    clusters.append(new_cluster)

    return clusters

def get_clusters(all_peaks, min_cluster_items):
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

    final_clusters = [cluster for cluster in clusters 
                      if len(cluster[1]) >= min_cluster_items]
    return final_clusters


def plot_clusters(clusters):

    for cluster in clusters:
        representative_energy = cluster[0]

        members = cluster[1]

        angles = [angle for _, angle, _ in members]
        heights = np.array([height for _, _, height in members])

        sizes = 150 * heights / np.max(heights)

        plt.scatter(
            [representative_energy] * len(angles),
            angles,
            s=sizes,
            alpha=0.7
        )

    plt.xlabel("Representative energy")
    plt.ylabel("Angle (degrees)")
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

        """top_3 = heapq.nlargest(
            3, 
            all_scores, 
            key = lambda score: score[1]
        )"""
    
    return all_scores, all_scores[:5]

def get_all_angular_momentum(clusters, trial_lst):
    predictions = []
    for cluster in clusters:
        _, max = get_angular_momentum(cluster, trial_lst)
        predictions.append((float(round(cluster[0], 5)), max))

    return predictions