# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
from scipy import ndimage
from skimage import feature
from skimage.util import invert

#defining the main function 
def first_guess(elecData, ionData, config, all_axes, sa):

    # normalizing data nd accounting for feducials if necesary 
    def data_processing(data, config,wave_type):
        # If data is timed resolved, remove feducials
        if config["other"]["extraoptions"]["spectype"] == "temporal":
                #ROI without feducials for IAW & EPW
            if wave_type == "IAW":
                a,b = 150,850
            elif wave_type == "EPW":
                a, b = 200, 900
        else:
            #ROI cuting omitting known noise for EPW imaging data
            if wave_type =="EPW":
                a= 150
            elif wave_type == "IAW":
                a= 0
            b = 1023
        data = data[a:b,0:1023]
        min_val = data.min()
        max_val = data.max()
        data_normalized = ((data - min_val)/(max_val - min_val))* 255.0
        img = data_normalized.astype(np.uint8)

        return img

    # assuming the notch filter is located at 528 +-12 pixels divide epw image into blue and red regions
    def notch_filter(img):

        # blur_epw = cv.GaussianBlur(img,(21,21),0)

        notch_fliter_start = 516
        notch_filter_end = 540 

        if config["other"]["extraoptions"]["spectype"] == "temporal":

            notch_fliter_start = notch_fliter_start - 200
            notch_filter_end = notch_filter_end - 200
        else: 
            notch_fliter_start = notch_fliter_start - 150
            notch_filter_end = notch_filter_end - 150
        
        epw_blue_box = img[:notch_fliter_start,:]
        epw_red_box = img[notch_filter_end:,:]

        return epw_red_box, epw_blue_box
    
    def morphological_opening(img):
        #morphological opening to reduce noice and enhance features
        # plt.imshow(img, cmap = 'gray')
        # plt.show()
        # plt.title("Before morphological opening")
        kernel = np.ones((3, 3), np.uint8)
        erode = cv.erode(img,kernel,iterations = 1) 
        dilate = cv.dilate(erode,kernel, iterations = 1)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv.erode(dilate,kernel,iterations = 1)
        dilate = cv.dilate(erode,kernel, iterations = 1)

        kernel = np.ones((7, 7), np.uint8)
        erode = cv.erode(dilate,kernel,iterations = 1) 
        img = cv.dilate(erode,kernel, iterations = 1)
        # plt.title("After morphological opening")
        # plt.imshow(img)
        # plt.show()

        return img
    
    def alternate_procesing(data, config,wave_type):
          # If data is timed resolved, remove feducials
        if config["other"]["extraoptions"]["spectype"] == "temporal":
                #ROI without feducials for IAW & EPW
            if wave_type == "IAW":
                a,b = 150,850
            elif wave_type == "EPW":
                a, b = 200, 900
        else:
            #ROI cuting omitting known noise for EPW imaging data
            if wave_type =="EPW":
                a= 150
            elif wave_type == "IAW":
                a= 0
            b = 1023
        data = data[a:b,0:1023]
        p2, p98 = np.percentile(data, (2, 98))
        data = np.clip((data - p2) / (p98 - p2), 0, 1) * 255
        data = data.astype(np.uint8)
        # plt.imshow(data, cmap = 'turbo_r', origin = 'lower')
        # plt.title("Normalized data")
        # plt.show()
        data = morphological_opening(data)
        data = np.clip(data, 0, 255)
        data = np.rint(data).astype(np.uint8)
        plt.imshow(data, origin = 'upper', cmap = 'turbo_r')
        plt.title("jupyter process, morphological opening and clip")
        plt.show()

        return data        
    
    #corner detection and filtration
    def data_analysis(img):
        #find corners in eroded image
        corners = cv.goodFeaturesToTrack(img, 100, 0.25, 10)
        corners = np.intp(corners).reshape(-1, 2)
        # for corner in corners:
        #     x, y = corner.ravel()
        #     cv.circle(copy, (x, y), 5 , 255, 2)
        # cv.imshow('Filtered Corners', copy)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #filter found corners, only keep corners that have at least one neighboor within the max distance 
        filtered_corners = []
        max_distance = 100
        for i, corner in enumerate(corners):
            has_neighboor = False
            for j, other_corner in enumerate(corners):
                if i == j : 
                    continue # skip the same corner 
                # Euclaidian distance between current corner and other corners
                distance = np.linalg.norm(corner-other_corner)
                if distance <= max_distance:
                    has_neighboor = True
                    break
            if has_neighboor:
                filtered_corners.append(corner)
        filtered_corners = np.array(filtered_corners)
        #  uncomment the next few lines to display located corners
        copy2 = img.copy()
        # for corner in filtered_corners:
        #     x, y = corner.ravel()
        #     cv.circle(copy2, (x, y), 5 , 255, 2)
        # cv.imshow('Filtered Corners', copy2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #find the max and min x and y coordinates
        min_x = filtered_corners[:, 0].min()
        max_x =  filtered_corners[:, 0].max()
        min_y = filtered_corners[:, 1].min()
        max_y = filtered_corners[:, 1].max()

        # add buffer to the min and max values
        x_start = config["feature_detector"]["buffer"]["lineout_start"]
        x_end = config["feature_detector"]["buffer"]["lineout_end"]
        y_start = config["feature_detector"]["buffer"]["spectral_start"]
        y_end = config["feature_detector"]["buffer"]["spectral_end"]
        min_x -= x_start
        max_x += x_end
        min_y -= y_start
        max_y += y_end


        return min_x, max_x, min_y, max_y
    
    def edge_detection(img,min_x, max_x, min_y, max_y):

        # Output the results
        print("min x coordinate:", min_x)
        print("max x coordinate:", max_x)
        print("min y coordinate:", min_y)
        print("max y coordinate:", max_y)

        #roi
        roi_box = img[min_y:max_y, min_x:max_x]
        plt.imshow(roi_box)
        plt.show()
        # min_y = np.min(roi_box[:,1])
        # max_y = np.max(roi_box[:,1])
        # max_y = roi_box[:,1].max()

        edges = feature.canny(roi_box, sigma =4)
        plt.imshow(edges)
        plt.axis('on')
        plt.show()
        indices = np.where(edges != [0]) 
        coordinates = list(zip(indices[1],indices[0])) #make a list of coordinates with the found points
        coordinates.sort(key=lambda point: point[1]) #sort by y value 

        return coordinates
    
    def midpoints(roi_box,coordinates,min_y, max_y):

        # iaw specific since it does upper and lower ? 
        #center = ((max_y - min_y) / 2) + min_y
        # for x, y in coordinates.items():
        #     miny = min(y)
        #     maxy = max(y)
        # miny = coordinates[:, 1].min()
        # maxy = coordinates[:, 1].max()
        y_vals = [y for x, y in coordinates]
        miny = min(y_vals)
        maxy = max(y_vals)
        center = ((maxy - miny)/2) +miny
        lower_groups = {}
        upper_groups = {}
        # Separate and group in a single pass
        for x, y in coordinates:
            if y <= center:
                if x not in lower_groups:
                    lower_groups[x] = []
                lower_groups[x].append(y)
            else:
                if x not in upper_groups:
                    upper_groups[x] = []
                upper_groups[x].append(y)
        #return lower_groups, upper_groups
        def compute_stats(groups):
            stats = {}
            for x, y_values in groups.items():
                y_min = min(y_values)
                y_max = max(y_values)
                width = y_max - y_min
                midpoint = (y_min + y_max) / 2
                stats[x] = (y_min, y_max, width, midpoint)
            return stats
        

        lower_stats = compute_stats(lower_groups)
        upper_stats = compute_stats(upper_groups)


        # Compute midpoint distances for common x-values
        common_x = set(lower_groups.keys()) & set(upper_groups.keys())
        midpoint_distances = {
            x: abs(upper_stats[x][3] - lower_stats[x][3])
            for x in common_x
        }

        incident = {
            x: (abs(upper_stats[x][3] - lower_stats[x][3]) / 2) + lower_stats[x][3]
            for x in common_x
        }


        result = {
                'groups_lower': lower_groups,
                'groups_upper': upper_groups,
                'lower_stats': lower_stats,
                'upper_stats': upper_stats,
                'midpoint_distances': midpoint_distances,
                'incident': incident
            }
        lower_midpoints = {x: stats[3] for x, stats in result['lower_stats'].items()}
        upper_midpoints = {x: stats[3] for x, stats in result['upper_stats'].items()}

        # Find common x-values (where both lower and upper midpoints exist)
        common_x = sorted(set(lower_midpoints.keys()) & set(upper_midpoints.keys()))

        # Ensure we only plot x-values that exist in BOTH groups
        if not common_x:
            print("No common x-values found between lower and upper groups.")
        else:
            
            # Plot
            #plt.figure(figsize=(10, 5))
            plt.imshow(roi_box)
            plt.scatter(common_x, [lower_midpoints[x] for x in common_x], marker='o', color='blue', s=3,label='Lower Midpoints (y ≤ c)')
            plt.scatter(common_x, [upper_midpoints[x] for x in common_x], marker='o', color='red', s=3, label='Upper Midpoints (y > c)')
            plt.show()



        # plt.imshow(roi_box)
        # for x, y_values in upper_stats.items():
        #     plt.plot(x, y_values['midpoint'], marker='o', color='red', markersize=1)

        # # Plot midpoints for the lower half
        # for x, y_values in lower_stats.items():
        #     plt.plot(x, y_values['midpoint'], marker='o', color='blue', markersize=1)

        # plt.title("Image with Midpoints")
        # plt.xlabel("X-coordinate")
        # plt.ylabel("Y-coordinate")

        # plt.show()      

        # Assuming 'result' is the output from `process_coordinates_efficient()`
        midpoint_distances = result['midpoint_distances']
        # Extract x and distances
        # x_values = list(midpoint_distances.keys())
        # distances = list(midpoint_distances.values())

        # plt.imshow(roi_box)
        # plt.scatter(x_values, distances, color='red', label='Midpoint Distance')
        # plt.plot(x_values, distances, linestyle='--', alpha=0.5)  # Optional: Connect points
        # plt.title('Midpoint Distances for Each X Value')
        # plt.show()

        incident = result['incident']
        x_values = list(incident.keys())
        incident_values = list(incident.values())
        # plt.imshow(roi_box)
        # plt.scatter(x_values, incident_values, color='green', s = 1, label='Incident')
        # # plt.plot(x_values, incident_values, linestyle='--', alpha=0.5)  # Optional: Connect points
        # plt.show()

        
        # plt.imshow(roi_box)
        # for x, incident in result.items():
        #     plt.plot(x, incident['incident'], marker='o', color='green', markersize=3)
        # plt.title("Image with incident frequency ?")
        # plt.xlabel("X-coordinate")
        # plt.ylabel("Y-coordinate")

        # plt.show()
        # x = 500
        # if x in result['midpoint_distances']:
        #     peak_separation = result['midpoint_distances'][x]
        # else: 
        #     print("peak separation not found")
        # if x in result['upper_stats']:
        #     upper_midpoint = result['upper_stats'][x][3]
        #     upper_width = result['upper_stats'][x][2]
        # else:
        #     upper_midpoint = None

        # if x in result['lower_stats']:
        #     lower_midpoint = result['lower_stats'][x][3]
        #     lower_width = result['lower_stats'][x][2]
        # else:
        #     lower_midpoint = None
        
        # if upper_midpoint is not None and lower_midpoint is not None:
        #     midpoint_separation = upper_midpoint - lower_midpoint
        # else:
        #     midpoint_separation = None
        # print( "peak separation", peak_separation)
        # print("midpoint_separation", midpoint_separation)
        # # conversion factor pixels to nm/pixel
        # epw_conversion_factor = all_axes["epw_y"][1] - all_axes["epw_y"][0]
        # iaw_conversion_factor = all_axes["iaw_y"][1] - all_axes["iaw_y"][0]

        # # nm/ pixel to cm / picel 
        # epw_cm_per_pixel = epw_conversion_factor / 1e-7
        # iaw_cm_per_pixel = iaw_conversion_factor / 1e-7

        # # speed of light in cm/s
        # c = 2.99792458e10
        # # wpe frequency in ras /s using the cm conversion
        # wpe_cm = (peak_separation / epw_cm_per_pixel ) * c * 2 * np.pi
        # me_cm = 510.9896 / c**2  # electron mass keV/C^2
        # re = 2.8179e-13  # classical electron radius cm
        # Esq = me_cm * c **2 * re  # sq of the electron charge keV cm
        # qe = Esq ** 2
        # ne_cm = (me_cm * wpe_cm) / (4 * np.pi * qe **2)

        # # speed of light in nm/s
        # c = 2.99792458e17
        # # wpe frequency in ras /s using the nm coinversion
        # me_nm = 0.91e-27 # electron mass in g
        # qe_c = 1.6e-19 # electron charge in C
        # qe_esu = 4.8e-10 # electron charge in esu
        # wpe_nm = (peak_separation / epw_conversion_factor ) * c * 2 * np.pi
        # ne_nm = (me_nm * wpe_nm) / (4 * np.pi * qe_esu **2)


        # print("electron density in cm^-3", ne_cm)
        # print("electron density in nm^-3",ne_nm)

        
        #wavelenght in nm to

        return result

    def scattering_angle(sa):
        # use the dictionary sa containing scattering angles and their corresponding weights to get a weighted average 
        print("scattering angle", sa)
        multiplied = [sa + k for sa, k in zip(sa["sa"], sa["weights"])]
        weights = sa["weights"]
        weighted_avg = sum(x * w for x,w in zip(multiplied, weights)) / sum(weights)
        print("weighted average", weighted_avg)
        return weighted_avg

    def iaw_feature_detector():
        #ion_img = data_processing(ionData, config,wave_type="IAW")

        #ion_img = morphological_opening(ion_img)

        ion_img = alternate_procesing(ionData, config,wave_type="IAW")

        #find and filter corners
        ion_min_x, ion_max_x, ion_min_y, ion_max_y = data_analysis(ion_img)

        #ion_img = morphological_opening(ion_img)

        coordinates = edge_detection(ion_img, ion_min_x, ion_max_x, ion_min_y, ion_max_y)
        if config["other"]["extraoptions"]["spectype"] == "temporal":
            a = 150
        else: 
            a = 0
        mapped_coordinates = [ (x + ion_min_x,y + a + ion_min_y) for x, y in coordinates]

        ion_data = ionData

        result = midpoints(ion_data, mapped_coordinates, ion_min_y, ion_max_y)
        x = 500
        if x in result['midpoint_distances']:
            peak_separation = result['midpoint_distances'][x]
        else: 
            print("peak separation not found")
        if x in result['upper_stats']:
            upper_midpoint = result['upper_stats'][x][3]
            upper_width = result['upper_stats'][x][2]
        else:
            upper_midpoint = None

        if x in result['lower_stats']:
            lower_midpoint = result['lower_stats'][x][3]
            lower_width = result['lower_stats'][x][2]
        else:
            lower_midpoint = None
        
        if upper_midpoint is not None and lower_midpoint is not None:
            midpoint_separation = upper_midpoint - lower_midpoint
            incident = (upper_midpoint - lower_midpoint) / 2 + lower_midpoint
        else:
            midpoint_separation = None
        print( "peak separation", peak_separation)
        print("midpoint_separation", midpoint_separation)

        scat = scattering_angle(sa)
        sang = scat / 180 * np.pi
        
        c = 2.99792458e10 # c in cm       
        # convert iaw y axis from nm to cm 
        iaw_y_axis_cm = (np.array(all_axes["iaw_y"])) / (1e7)
        #iaw_axis_frequency = (2 * np.pi * c) / (np.array(all_axes["iaw_y"]))
        iaw_frequency_axis = (2 * np.pi * c) / iaw_y_axis_cm

        incident_laser = 526.5 / (1e7) # in cm 
        incidentL = iaw_frequency_axis[int(incident)]
        #laser frquency in rads /s from nm -> cm -> hz -> rads/s
        omegaL = (2 * np.pi * c) / incident_laser
        kwi = np.sqrt((incidentL**2)/ (c **2))
        kiaw = 2 * kwi * np.sin(sang/2)

        #shifted frequencies 
        red_frequency_shift= iaw_frequency_axis[int(upper_midpoint)] - incidentL
        blue_frequency_shift = iaw_frequency_axis[int(lower_midpoint)] - incidentL

        wia =abs(blue_frequency_shift)
        
        # te 
        me_cm = 510.9896 / c**2  # electron mass keV/C^2
        mp = me_cm * 1837
        mn_mev = 939.57 # mss of a neutron in MeV / C^2
        mn = 939570 # mass of a neutron in KeV/C^2
        A = config["parameters"]["ion-1"]["A"]["val"]
        # for 1 amu = 931.5 MeV/c^2 ~931.494
        amu_g = 1.66054e-24 # g/amu
        amu_kev = (amu_g / c) / (6.242e15)
        amu = 931494 # kev/c^2
       # mi = (A * mn) + (A * mp) # ion mass
        mi = A * mp
        #mi = A * amu_kev
        z = config["parameters"]["ion-1"]["Z"]["val"]
        Te = (wia **2 ) * mi / ((kiaw ** 2) * z) # KeV
        Te_red = (red_frequency_shift **2 ) * mi / ((kiaw ** 2) * z) # KeV
        



        # speed of light in cm/s



        c_nm = 2.99792458e17 # speed of light in nm/s
        #incident frequency for 2 omega 532nm  
        wi = (2 * np.pi * c_nm)/ 532
        ki = np.sqrt((wi**2)/ (c_nm **2))
        kia = 2 * ki * np.sin(scat/2)


        lineout_start = ion_min_x
        lineout_end = ion_max_x

        # Calculate iaw_cf, iaw_max, and iaw_min

        if config["other"]["extraoptions"]["spectype"] == "temporal":
            a = 150
            b= 173
            iaw_max = ion_max_y + a
            iaw_min = ion_min_y + a
        else:
            iaw_max = ion_max_y
            iaw_min = ion_min_y
        iaw_cf = (iaw_max - iaw_min) * 0.2
        midpoint = (iaw_max + iaw_min)/2
        iaw_cf_min = midpoint - iaw_cf
        iaw_cf_max = midpoint + iaw_cf

        print("this are the min and maxs for IAW")
        print(f"IAW min x: {lineout_start}, IAW max x: {lineout_end}")
        print(f"IAW max y : {iaw_max}, IAW min y: {iaw_min}")
        print(f"IAW cf min: {iaw_cf_min}, IAW cf max: {iaw_cf_max}")

        roi = ion_img[ion_min_y:ion_max_y, ion_min_x:ion_max_x]
        # plt.imshow(roi)
        # plt.show()

        return lineout_end,lineout_start,iaw_cf_min,iaw_cf_max,iaw_max,iaw_min
    

    def epw_feature_detector():
        #elec_img = data_processing(elecData, config, wave_type="EPW")
        elec_img = alternate_procesing(elecData, config, wave_type="EPW")

        epw_red, epw_blue= notch_filter(elec_img)
        red_min_x, red_max_x, red_min_y, red_max_y = data_analysis(epw_red)
        blue_min_x, blue_max_x, blue_min_y, blue_max_y = data_analysis(epw_blue)
        
        
        print("this are the min and maxs for red and blue shifted EPW")
        print(f"Blue min x: {blue_min_x}, Blue max x: {blue_max_x}")
        print(f"Blue min y: {blue_min_y}, Blue max y: {blue_max_y}")
        print(f"Red min x: {red_min_x}, Red max x: {red_max_x}")
        print(f"Red min y: {red_min_y}, Red max y: {red_max_y}")     

        blue_coordinates = edge_detection(epw_blue, blue_min_x, blue_max_x, blue_min_y, blue_max_y)
        red_coordinates = edge_detection(epw_red, red_min_x, red_max_x, red_min_y, red_max_y)

        if config["other"]["extraoptions"]["spectype"] == "temporal":
            b = 200
            r = 540
        else: 
            b = 150
            r = 540
        mapped_blue_coordinates = [ (x + blue_min_x,y + b + blue_min_y) for x, y in blue_coordinates]
        mapped_red_coordinates = [ (x + red_min_x,y + 390 + b + red_min_y) for x, y in red_coordinates]
        b_min_y = blue_min_y 
        r_max_y = red_max_y 


        combined_coordinates = []
        combined_coordinates.extend(mapped_blue_coordinates)
        combined_coordinates.extend(mapped_red_coordinates)

        result = midpoints(elec_img, combined_coordinates, b_min_y, r_max_y)

        x = 500
        if x in result['midpoint_distances']:
            peak_separation = result['midpoint_distances'][x]
        else: 
            print("peak separation not found")
        if x in result['upper_stats']:
            upper_midpoint = result['upper_stats'][x][3]
            upper_width = result['upper_stats'][x][2]
        else:
            upper_midpoint = None

        if x in result['lower_stats']:
            lower_midpoint = result['lower_stats'][x][3]
            lower_width = result['lower_stats'][x][2]
        else:
            lower_midpoint = None
        
        if upper_midpoint is not None and lower_midpoint is not None:
            midpoint_separation = upper_midpoint - lower_midpoint
        else:
            midpoint_separation = None
        print( "peak separation", peak_separation)
        print("midpoint_separation", midpoint_separation)
        # conversion factor pixels to nm/pixel
        
        c_nm = 2.99792458e17 # speed of light in nm/s
        c = 2.99792458e10 # speed of light in cm/s
        me_cm = 510.9896 / c**2  # electron mass keV/C^2
        re = 2.8179e-13  # classical electron radius cm
        Esq = me_cm * c **2 * re  # sq of the electron charge keV cm
        constants = 4 * np.pi * Esq / me_cm
        axis_cm = (np.array(all_axes["epw_y"])) / (1e7)
        axis_frequecy = (2 * np.pi * c) / axis_cm
        epw_y_axis_array_frequency_cm = (2 * np.pi * c) / (np.array(all_axes["epw_y"])/ (1e7))
        # get omegal from lam val in the input deck
        incident_laser_cm = 526.5 / (1e7) # in cm
        omegaLcm = (2 * np.pi * c) / incident_laser_cm
        blue_frewuency_shift_cm = epw_y_axis_array_frequency_cm[int(lower_midpoint)] - omegaLcm
        ne_cm = ((blue_frewuency_shift_cm  **2)/ 5.65e4)
        print("electron density in cm^-3", ne_cm)    
        ne0 = blue_frewuency_shift_cm ** 2 / constants        
        print("ne using constants ", ne0)
        #  frequencies in rads / s using c in nm
        epw_y_axis_array_frequency = (2 * np.pi * c_nm) / (np.array(all_axes["epw_y"]))

        


        incident_laser = 532
        omegaL = (2 * np.pi * c_nm) / incident_laser

        red_frequency_shift = epw_y_axis_array_frequency[int(upper_midpoint)] - omegaL # should be negativve? 
        blue_frewuency_shift = epw_y_axis_array_frequency[int(lower_midpoint)] - omegaL

        ne = (blue_frewuency_shift/5.65e4)**2
        #ne_cm = (me_cm * (blue_frewuency_shift ** 2)) / ( * qe **2)

        print("electron density in cm^-3, with c and lambda in nm", ne)




        # epw_conversion_factor = all_axes["epw_y"][1] - all_axes["epw_y"][0]
        # iaw_conversion_factor = all_axes["iaw_y"][1] - all_axes["iaw_y"][0]

        # nm/ pixel to cm / picel 
        # epw_cm_per_pixel = epw_conversion_factor / 1e-7
        # iaw_cm_per_pixel = iaw_conversion_factor / 1e-7


        # wpe frequency in ras /s using the cm conversion
        #wpe_cm = (peak_separation / epw_cm_per_pixel ) * c * 2 * np.pi

        

        # wpe frequency in ras /s using the nm coinversion
        # me_nm = 0.91e-27 # electron mass in g
        # qe_c = 1.6e-19 # electron charge in C
        # qe_esu = 4.8e-10 # electron charge in esu
        # wpe_nm = (peak_separation / epw_conversion_factor ) * c * 2 * np.pi
        # ne_nm = (me_nm * wpe_nm) / (4 * np.pi * qe_esu **2)
        # ne_cm_2 = (me_cm * wpe_cm) / (4 * np.pi * qe **2)
        # print("electron density in nm^-3",ne_nm)





        # preprocessed_ewp_red = morphological_opening(epw_red)
        # preprocessed_ewp_blue = morphological_opening(epw_blue)
        # red_min_x, red_max_x, red_min_y, red_max_y = data_analysis(preprocessed_ewp_red)
        # blue_min_x, blue_max_x, blue_min_y, blue_max_y = data_analysis(preprocessed_ewp_blue)

        if config["other"]["extraoptions"]["spectype"] == "temporal":
            a = 200
            b = 540
        else: 
            a = 150
            b = 540
        red_max_y = red_max_y + a
        red_min_y = red_min_y + a
        blue_max_y = blue_max_y + b
        blue_min_y = blue_min_y + b

        print("this are the min and maxs for red and blue shifted EPW")
        print(f"Blue min x: {blue_min_x}, Blue max x: {blue_max_x}")
        print(f"Blue min y: {blue_min_y}, Blue max y: {blue_max_y}")
        print(f"Red min x: {red_min_x}, Red max x: {red_max_x}")
        print(f"Red min y: {red_min_y}, Red max y: {red_max_y}")        

        blue_max = blue_max_y
        blue_min = blue_min_y
        lineout_start = red_min_x
        lineout_end = red_max_x
        red_min =  red_min_y
        red_max = red_max_y 

        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max
    

    if config["feature_detector"]["estimate_lineouts_iaw"] and not config["feature_detector"]["estimate_lineouts_epw"]:
        lineout_end,lineout_start,iaw_cf_min, iaw_cf_max,iaw_max,iaw_min = iaw_feature_detector()
        #Sanity checks
        if lineout_end < lineout_start or iaw_min > iaw_max:
            raise ValueError("Lineout end is less than lineout start or IAW min is greater than max. Detector failed")
        else:
            pass
        return lineout_end, lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min

    if config["feature_detector"]["estimate_lineouts_epw"] and not config["feature_detector"]["estimate_lineouts_iaw"]:    

        lineout_end, lineout_start, blue_min, blue_max, red_min, red_max = epw_feature_detector()
        #Sanity checks
        if lineout_end < lineout_start or blue_min > blue_max or red_min > red_max:
            raise ValueError("Lineout end is less than lineout start or  blue/red min is greater than max. Detector failed")
        else:
            pass
        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max

    if config["feature_detector"]["estimate_lineouts_epw"] and config["feature_detector"]["estimate_lineouts_iaw"]:

        iaw_lineout_end, iaw_lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min = iaw_feature_detector()

        epw_lineout_end, epw_lineout_start, blue_min, blue_max, red_min, red_max = epw_feature_detector()
        
        #Sanity checks
        if epw_lineout_end < epw_lineout_start or blue_min > blue_max or red_min > red_max:
            raise ValueError("Lineout end is less than lineout start or  blue/red min is greater than max. Detector failed")
        else:
            pass

        if iaw_lineout_start > iaw_lineout_end or iaw_min > iaw_max:
            raise ValueError("Lineout end is less than lineout start or IAW min is greater than max. Detector failed")
        else:
            pass
        # Calculate ion t0 shift
        if iaw_lineout_start == epw_lineout_start:
            ion_t0_shift = 0
        else:
            ion_t0_shift =  epw_lineout_start - iaw_lineout_start
            #ion_t0_shift = iaw_lineout_start - epw_lineout_start

        return epw_lineout_end, epw_lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min, ion_t0_shift, blue_min, blue_max, red_min, red_max
