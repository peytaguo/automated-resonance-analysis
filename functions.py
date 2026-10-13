from pathlib import Path
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

def get_interpolation(ke_arr, sigma_arr, gap_size):
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

    print(len(ke_parts[0]))

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




