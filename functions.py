from scipy.signal import find_peaks_cwt
from matplotlib import pyplot as plt
import numpy as np 

def cross_sections(angle, theta_arr, ke_arr, sigma_arr, dsigma_arr):
    angle_mask = theta_arr == angle

    ke_w_mask = ke_arr[angle_mask]
    sigma_w_mask = sigma_arr[angle_mask]
    dsigma_w_mask = dsigma_arr[angle_mask]

    return ke_w_mask, sigma_w_mask, dsigma_w_mask

def get_all_cs(theta_arr, ke_arr, sigma_arr, dsigma_arr):
    unique_angles = np.unique(theta_arr)

    cs_by_angle = {}

    for angle in unique_angles:
        ke, sigma, dsigma = cross_sections(
            angle,
            theta_arr, 
            ke_arr,
            sigma_arr,
            dsigma_arr
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
    for ke_part, sigma_part in interp_parts:
        peak_index = find_peaks_cwt(
            sigma_part,
            widths = np.arange(1, 50),
            min_length = 3,
            min_snr = 1
        )
        peak_energies_all.append(ke_part[peak_index])
    return np.concatenate(peak_energies_all)

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
        angle_peaks[f"{angle}"] = get_peaks(interp_parts)

    return angle_peaks

def visualise_all_peaks(cross_sections):
    for angle in cross_sections.keys():
        #print(angle)
        interp_parts, _, _ = get_interpolation(
            cross_sections[angle]["ke"],
            cross_sections[angle]["sigma"]
        )
        visualise_peaks(interp_parts, angle)