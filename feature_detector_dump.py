# this will be the feature detector for the first run of the code 
# importing necesary modelues
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.colors as colors
import tempfile, os
from skimage import feature

#defining the main function 
def first_guess(elecData, ionData, all_axes, config):

    #if config["other"]["extraoptions"]["load_ion_spec"]: # ploting the iaw spec w/out lineouts
    def data_processing(data, config):
        if config["data"]["background"]["type"] == "pixel":
                #ROI without feducials for IAW
            if config["data"]["estimate_lineouts_iaw"]:
                a,b = 150,850
                # ROI without feducials for EPW 
            elif config["data"]["estimate_lineouts_epw"]:
                a, b = 170, 900
        
            feducials = data[a:b,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials.min()
            max_val = feducials.max()

            # Perform normalization
            feducials_normalized = ((feducials - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_normalized.astype(np.uint8)

        else:

            min_val = data.min()
            max_val = data.max()
            data_normalized = ((data - min_val)/(max_val - min_val))* 255.0
            img = data_normalized.astype(np.uint8)
        return img 
    
    def find_filter_corners(img):
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(img, kernel, iterations=2)

        # Show the eroded image
        plt.imshow(eroded)
        plt.title("Eroded Image")
        plt.axis('off')
        plt.show()

        corners = cv.goodFeaturesToTrack(eroded, 50, 0.01, 10)
        corners = np.int0(corners).reshape(-1, 2)

        for corner in filtered_corners:
            x, y = corner.ravel()
            cv.circle(eroded, (x, y), 5 , 255, 2)

        # Display the result
        cv.imshow('Filtered Corners', eroded)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Filter corners: only keep corners that have at least one other corner within the max distance
        filtered_corners = []
        max_distance = 50
        for i, corner in enumerate(corners):
            has_neighbor = False
            for j, other_corner in enumerate(corners):
                if i == j:
                    continue  # skip same corner
                # Euclidean distance between current corner and other corners
                distance = np.linalg.norm(corner - other_corner)
                if distance <= max_distance:
                    has_neighbor = True
                    break
            if has_neighbor:
                filtered_corners.append(corner)
        return np.array(filtered_corners)

    # Function to calculate min and max coordinates
    def calculate_min_max(corners):
        if corners.size > 0:
            min_x = np.min(corners[:, 0])  # Minimum x-coordinate
            max_x = np.max(corners[:, 0])  # Maximum x-coordinate
            min_y = np.min(corners[:, 1])  # Minimum y-coordinate
            max_y = np.max(corners[:, 1])  # Maximum y-coordinate

            print(f"Min x: {min_x}, Max x: {max_x}")
            print(f"Min y: {min_y}, Max y: {max_y}")

            # Add a margin of error
            min_x -= 20
            max_x += 20
            min_y -= 20
            max_y += 20

            return min_x, max_x, min_y, max_y
        else:
            print("No filtered corners found.")
            return None, None, None, None
        
    if config ["data"]["estimate_lineouts_iaw"]:
        # if the image is time resolved, select ROI without feducials
        #process data
        ion_img = data_processing(ionData, config)
        #find and filter corners
        ion_corners = find_filter_corners(ion_img)
        #calculate min and max coordinates
        ion_min_x, ion_max_x, ion_min_y, ion_max_y = calculate_min_max(ion_corners)

        if ion_min_x is not None:
            #lineout start and end for ionData
            lineout_start = ion_min_x
            lineout_end = ion_max_x

             # Calculate iaw_cf, iaw_max, and iaw_min
            if config["data"]["background"]["type"] == "pixel":
                a = 150
                b = 173
                iaw_max = ion_max_y + a
                iaw_min = ion_min_y - b
            else:
                iaw_max = ion_max_y
                iaw_min = ion_min_y
            iaw_cf = (iaw_max + iaw_min) / 2

        
        if config["data"]["background"]["type"] == "pixel":
            #ROI without feducials
            a = 150
            b= 173
            feducials_iaw = ionData[150:850,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials_iaw.min()
            max_val = feducials_iaw.max()

            # Perform normalization
            feducials_iaw_normalized = ((feducials_iaw - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_iaw_normalized.astype(np.uint8)

        else:

            min_val = ionData.min()
            max_val = ionData.max()
            ionData_normalized = ((ionData - min_val)/(max_val - min_val))* 255.0
            img = ionData_normalized.astype(np.uint8)
        

            # Ensure the selected region is in uint8 format
            #feducials_iaw = (feducials_iaw*255).astype(np.uint8)
            #img= feducials_iaw.astype(np.float32)

                            # Plot both ionData and feducials_iaw side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot ionData
            axes[0].imshow(ionData, cmap='inferno')
            axes[0].set_title("ionData")
            axes[0].axis('off')  # Hide axes for cleaner view

            # Plot feducials_iaw
            axes[1].imshow(feducials_iaw_normalized, cmap='inferno')
            axes[1].set_title("feducials_iaw")
            axes[1].axis('off')  # Hide axes for cleaner view

            # Plot img which is the uint8 version of feducials_iaw_normalized
            axes[1].imshow(img, cmap='inferno')
            axes[1].set_title("feducials_iaw")
            axes[1].axis('off')  # Hide axes for cleaner view

            plt.tight_layout()  # Adjust layout to avoid overlap
            plt.show()
             # Find corners in the eroded image
            # Perform erosion on the binary image

            kernel = np.ones((2, 2), np.uint8)
            eroded = cv.erode(img, kernel, iterations=2)

            # Show the eroded image
            plt.imshow(eroded)
            plt.title("Eroded Image")
            plt.axis('off')
            plt.show()

            corners = cv.goodFeaturesToTrack(eroded, 50, 0.01, 10)
            corners = np.int0(corners).reshape(-1,2)

            #filter conrners: only keep corners that have at least one other corner within the max distance
            filtered_corners = []
            max_distance = 50
            for i, corner in enumerate(corners):
                has_neighboor = False
                for j, other_corner in enumerate(corners):
                    if i == j:
                        continue # skip same corner
                    #Euclidean distance between current corner and other corners
                    distance = np.linalg.norm(corner-other_corner)
                    if distance <= max_distance:
                        has_neighboor = True
                        break
                if has_neighboor:
                    filtered_corners.append(corner)
            # Convert filtered corners back to the required format
            filtered_corners = np.array(filtered_corners)
            # Find the max and min x and y coordinates
            if filtered_corners.size > 0:
                min_x = np.min(filtered_corners[:, 0])  # Minimum x-coordinate
                max_x = np.max(filtered_corners[:, 0])  # Maximum x-coordinate
                min_y = np.min(filtered_corners[:, 1])  # Minimum y-coordinate
                max_y = np.max(filtered_corners[:, 1])  # Maximum y-coordinate

                print(f"Min x: {min_x}, Max x: {max_x}")
                print(f"Min y: {min_y}, Max y: {max_y}")
            else:
                print("No filtered corners found.")
            
            
            # Add a margin of erron

            min_x -= 20
            max_x += 20
            min_y -= 20
            max_y += 20

            lineout_start = min_x
            lineout_end = max_x

            if config["data"]["background"]["type"] == "pixel":

                # mapping found points to the original image
                iaw_max = max_y + a
                iaw_min =  min_y - b

            else:
                iaw_max = max_y
                iaw_min =  min_y
            

            iaw_cf = (iaw_max+iaw_min)/2

        return lineout_end,lineout_start,iaw_cf,iaw_max,iaw_min

    if config ["data"]["estimate_lineouts_epw"]:
        #process data
        elec_img = data_processing(elecData, config)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #plot electData
        axes[0].imshow(elecData, cmap='inferno')
        axes[0].set_title("elecData")
        axes[0].axis('off')
        #plot feducials_epw
        axes[1].imshow(feducials_epw_normalized, cmap='inferno')   
        axes[1].set_title("feducials_epw")
        axes[1].axis('off')
        # Plot img which is the uint8 version of feducials_iaw_normalized
        axes[1].imshow(img, cmap='inferno')
        axes[1].set_title("feducials_iaw")
        axes[1].axis('off')  # Hide axes for cleaner view
        plt.tight_layout()
        plt.show()

        #find and filter corners
        elec_corners = find_filter_corners(elec_img)
        #calculate min and max coordinates
        elec_min_x, elec_max_x, elec_min_y, elec_max_y = calculate_min_max(elec_corners)
        # if the image is time resolved, select ROI without feducials
        print( elec_min_x, elec_max_x, elec_min_y, elec_max_y)
        if config["data"]["background"]["type"] == "pixel":
            #ROI without feducials
            a = 150
            b= 123
            feducials_epw = elecData[150:900,0:1023]

            # Normalize the values to the range [0, 255]
            min_val = feducials_epw.min()
            max_val = feducials_epw.max()

            # Perform normalization
            feducials_epw_normalized = ((feducials_epw - min_val) / (max_val - min_val)) * 255.0

            # Convert the image to uint8
            img = feducials_epw_normalized.astype(np.uint8)

        else:

            min_val = elecData.min()
            max_val = elecData.max()
            elecData_normalized = ((elecData - min_val)/(max_val - min_val))* 255.0
            img = elecData_normalized.astype(np.uint8)
        

        # Plot ionData
        axes[0].imshow(elecData, cmap='inferno')
        axes[0].set_title("elecData")
        axes[0].axis('off')  # Hide axes for cleaner view

        # Plot feducials_iaw
        axes[1].imshow(feducials_epw_normalized, cmap='inferno')
        axes[1].set_title("feducials_epw")
        axes[1].axis('off')  # Hide axes for cleaner view

        # Plot img which is the uint8 version of feducials_iaw_normalized
        axes[1].imshow(img, cmap='inferno')
        axes[1].set_title("feducials_epw_img")
        axes[1].axis('off')  # Hide axes for cleaner view

        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()

        # Draw filtered corners on the image
        for corner in filtered_corners:
            x, y = corner.ravel()
            cv.circle(eroded, (x, y), 5 , 255, 2)

        # Display the result
        cv.imshow('Filtered Corners', eroded)
        cv.waitKey(0)
        cv.destroyAllWindows()


        # List to store the corner coordinates
        corner_list = []
        for i in corners:
            x, y = i.ravel()
            cv.circle(eroded, (x, y), 5,255, 2)  # Mark the corners on the image
            corner_list.append((x, y))

        # Show the image with marked corners
        plt.imshow(eroded, cmap='gray')
        plt.title("Image with Corners")
        plt.axis('off')
        plt.show()

        # Find the smallest and largest x and y values from the corner coordinates
        corner_list.sort()  # Sort by x-coordinate
        min_x = corner_list[0][0]
        max_x = corner_list[-1][0]

        corner_list.sort(key=lambda point: point[1])  # Sort by y-coordinate
        min_y = corner_list[0][1]
        max_y = corner_list[-1][1]

        # Calculate the average y-coordinate (iaw_cf)
        iaw_cf = (min_y + max_y) / 2

        # Output the results
        print("min x coordinate:", min_x)
        print("max x coordinate:", max_x)
        print("min y coordinate:", min_y)
        print("max y coordinate:", max_y)
        print("Average y-coordinate (iaw_cf):", iaw_cf)



        if iaw_cf == 0:
            # Ensure the selected region is in uint8 format
            img = feducials_iaw.astype(np.uint8)
            X, Y = np.meshgrid(np.arange(feducials_iaw.shape[1]), np.arange(feducials_iaw.shape[0]))

            # Create a figure to plot
            fig, ax = plt.subplots()

            # Plotting using pcolormesh
            mesh = ax.pcolormesh(X, Y, feducials_iaw, cmap="inferno", 
                                norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                        vmin=np.amin(feducials_iaw), vmax=np.amax(feducials_iaw)),
                                shading="auto")  # Use "flat" or "auto" for shading
            #fig.colorbar(mesh, label="Intensity")
            #plt.show()


            # Apply adaptive thresholding to get a binary image
            bw_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            # Show the binary (black & white) image
            plt.imshow(bw_img, cmap='gray')
            plt.title("Binary Image")
            plt.axis('off')
            plt.show()

            # Perform erosion on the binary image
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv.erode(bw_img, kernel, iterations=2)

            # Show the eroded image
            plt.imshow(eroded)
            plt.title("Eroded Image")
            plt.axis('off')
            plt.show()
            # Find corners in the eroded image
            corners = cv.goodFeaturesToTrack(eroded, 10, 0.01, 10)
            corners = np.int0(corners)

            # List to store the corner coordinates
            corner_list = []
            for i in corners:
                x, y = i.ravel()
                cv.circle(img, (x, y), 3, 255, -1)  # Mark the corners on the image
                corner_list.append((x, y))

            # Show the image with marked corners
            plt.imshow(img, cmap='gray')
            plt.title("Image with Corners")
            plt.axis('off')
            plt.show()

            # Find the smallest and largest x and y values from the corner coordinates
            corner_list.sort()  # Sort by x-coordinate
            min_x = corner_list[0][0]
            max_x = corner_list[-1][0]

            corner_list.sort(key=lambda point: point[1])  # Sort by y-coordinate
            min_y = corner_list[0][1]
            max_y = corner_list[-1][1]

            # Calculate the average y-coordinate (iaw_cf)
            iaw_cf = (min_y + max_y) / 2

            # Output the results
            print("min x coordinate:", min_x)
            print("max x coordinate:", max_x)
            print("min y coordinate:", min_y)
            print("max y coordinate:", max_y)
            print("Average y-coordinate (iaw_cf):", iaw_cf)

            # Create a figure to plot
            
        #if the data is not time resolved, image size remains 1024x1024
        #else:
            img = ionData.astype(np.uint8)
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            fig, ax = plt.subplots()
            ax.axis('on')
            cb = ax.pcolormesh(
            X,
            Y,
            ionData,
            cmap="turbo_r",
            #norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03, vmin =0, vmax= np.amax(ionData)),
            norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
            shading= "auto",

            )               
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            ionData,
            fig1, ax1 = plt.subplots()
            ax1.axis('on')
            ax1.pcolormesh(
                X,
                Y,
                ionData,
                cmap="turbo_r",
                norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
                #vmin=0,
                #vmax=0.05*np.amax(ionData),
                #shading = "auto",
            )
        
            """plt.figure(figsize=(10, 8))

            # Displaying the extracted portion using imshow
            plt.imshow(feducials_iaw, cmap="turbo_r", origin="lower", aspect="auto", 
        norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                vmin=np.amin(feducials_iaw), vmax=np.amax(feducials_iaw)))
            #fig1.show()"""
            # Add color bar to the plot
            #fig1.colorbar(mesh, ax=ax1)
            #plt.imshow(img_iaw)
            plt.show()
            fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
            fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
            #openinn iaw and geting roi
            iaw1 = cv.imread(fig1_path)
            #fig1.show()
            #plt.plot(ionData)
            #plt.show()




            # assuming these are the right coordinates for feducials
            a = 70
            b = 390
            c = 10
            d = 487
            
            img_iaw = ionData[a:b,c:d]# to account for time feducials
    else:

        img_iaw = ionData.shape[:2] # if there are no feducials
    img_iaw = ionData.shape[:2]
    # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
    #gray_iaw =cv.cvtColor(img_iaw,cv.COLOR_BGR2GRAY)
    max_gray_value = np.max(img_iaw) #find the highest value from the gray scale image 
    threshold_value = 0.5 * max_gray_value #set the threshold to 50% of the maximum value 
    ret,bw_iaw = cv.threshold(img_iaw,threshold_value, 255, cv.THRESH_BINARY)  # apply thresholdin0g
    kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

    eroded = cv.erode(bw_iaw, kernel, iterations = 2) #bw_iaw holds the eroded image
    eroded = eroded.astype(np.uint8) 
    corners_iaw = cv.goodFeaturesToTrack(eroded,50,0.01,10) # eroded is now single channel
    corners_iaw = np.int0(corners_iaw)

    #list for corners coordinates
    corner_list_iaw = []
    for i in corners_iaw:
        x,y = i.ravel()
        cv.circle(img_iaw,(x,y),2,255,-1)
        corner_list_iaw.append((x,y))

    #Finding the min and max x and y points of the feature
    corner_list_iaw.sort()  # Sort in-place by x-coordinate (default behavior)
    x_min = corner_list_iaw[0][0]  # Extract only the x-value
    x_max = corner_list_iaw[-1][0]
    corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
    y_min = corner_list_iaw[0][1]  # Extract only the y-value
    y_max = corner_list_iaw[-1][1] # Extract only the y-value

    #mapoping found points to the original image
    OGX_min = x_min + c
    OGx_max = x_max + d
    OGy_min = y_min + b
    OGy_max = y_max + a

    #maping findings to their corresponding parameters
    lineout_start = OGX_min
    lineout_end = OGx_max
    iaw_max = OGy_max
    iaw_min = OGy_min


    #argin of error
    x_min -= 20
    x_max += 20
    y_min -= 20
    y_max += 20

    iaw_cf = (y_min + y_max)/2 

    return lineout_end,lineout_start,iaw_cf,x_min,x_max,y_max,y_min




    def crop_by_parallel_edges(img):

        # Apply edge detection
        #edges = cv.Canny(img, 50, 150)
        #edges = feature.canny(img, sigma=1)

        #plt.imshow(edges, cmap='gray')
        #plt.show()
        image = img
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(img, kernel, iterations=2)
        plt.imshow(eroded)
        plt.show()

        blur = cv.GaussianBlur(image, (7,7),0)
        plt.imshow(blur)
        plt.show()

        inverted = invert(blur)
        plt.imshow(inverted)
        plt.show()


   

        
        # Detect lines using Hough Line Transform
        linesP = cv.HoughLinesP(inverted, rho=1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=10)

        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines, thickness=2

        plt.imshow(image)
        plt.show()


        lines = cv.HoughLines(inverted, 1, np.pi/180, 50, None, 50,10)


        if lines is None:
            print("No lines detected.")
            return
        
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(edges, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        plt.imshow(edges)
        plt.show()

        plt.imshow(cdst)
        plt.show()
        plt.imshow(cdstP)
        plt.show()

        #cv.imshow("Source", img)
        #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        #cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        # Filter and find parallel lines (based on their angles and positions)
        parallel_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = None if x2 == x1 else (y2 - y1) / (x2 - x1)
            parallel_lines.append((x1, y1, x2, y2, slope))

        # Sort by y-coordinates to find the topmost and bottommost parallel lines
        parallel_lines.sort(key=lambda line: (line[1] + line[3]) / 2)  # Sort by vertical center

        # Check if at least two parallel lines exist
        if len(parallel_lines) < 2:
            print("Not enough parallel lines detected.")
            return

        # Use the topmost and bottommost lines to crop the rectangle
        top_line = parallel_lines[0]
        bottom_line = parallel_lines[-1]

        # Calculate cropping bounds
        top_y = min(top_line[1], top_line[3])
        bottom_y = max(bottom_line[1], bottom_line[3])
        height, width = img.shape[:2]


        # Draw the rectangle on the original image
        color = (0, 255, 0)  # Green color for the rectangle
        thickness = 2  # Thickness of the rectangle border
        cv.rectangle(img, (0, top_y), (width, bottom_y), color, thickness)

        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap='gray')
        plt.title("Cropped Rectangle")
        plt.axis('off')
        plt.show()


        # Crop the image
        cropped_image = img[top_y:bottom_y, 0:width]

        plt.imshow(cropped_image, cmap='gray')
        plt.show()

        return cropped_image
    

    
    def find_central_rectangle(img):
        """Find the most prominent rectangular region near the image center"""

        H, W = img.shape[:2]
        upper = (2*H) // 5
        lower = (3 * H) // 5

        cropped_img = img[upper:lower, 0:W]

        plt.imshow(cropped_img, cmap='gray')
        plt.show()


        # Create copy for visualization
        vis_img = cv.cvtColor(cropped_img, cv.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()


        # Detect edges using Canny
        edges = cv.Canny(cropped_img, 50, 150)

        plt.imshow(edges, cmap='gray')
        plt.show()
        
        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours near the center
        center_y, center_x =cropped_img.shape[0]//2, cropped_img.shape[1]//2
        best_rect = None
        best_score = 0
        
        for cnt in contours:
            # Approximate contour as polygon
            approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
            
            # Look for quadrilaterals
            if len(approx) == 4:
                # Calculate contour properties
                x, y, w, h = cv.boundingRect(approx)
                # Skip invalid rectangles
                if h == 0 or w == 0:
                    continue
                aspect_ratio = float(w)/h
                 # Add epsilon to prevent division by zero
                aspect_term = 1 / (abs(aspect_ratio - 1) + 1e-6)  # Prevent ZeroDivisionError
                # Score based on proximity to center and aspect ratio
                distance_from_center = abs(y + h/2 - center_y)
                distance_term = 1 / (1 + distance_from_center)
            
                score = distance_term * aspect_term
                #score = (1 / (1 + distance_from_center)) * (1 / abs(aspect_ratio - 1))
                
                if score > best_score:
                    best_score = score
                    best_rect = (x, y, w, h)
                    
        if best_rect:
            x, y, w, h = best_rect
            # Draw rectangle on visualization image
            cv.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add center line
            center_y = y + h//2
            cv.line(vis_img, (0, center_y), (vis_img.shape[1], center_y), (255, 0, 0), 1)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.imshow(vis_img)
        plt.title("Detected Central Rectangle")
        plt.axis('off')
        plt.show()
        
        return best_rect  # (x, y, width, height)

    def analyze_epw(data, config):
        img = data_processing(elecData, config)
        rect = find_central_rectangle(img)
        
        # Create visualization image for ROIs
        roi_vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        
        if rect:
            x, y, w, h = rect
            split_y1 = y
            split_y2 = y + h
            blue_roi = img[split_y2+10:, :]  # Below the rectangle
            red_roi = img[:split_y1-10, :]    # Above the rectangle

            # Draw ROIs
            cv.rectangle(roi_vis, (0, split_y2+10), (img.shape[1], img.shape[0]), (255, 0, 0), 2)  # Blue ROI
            cv.rectangle(roi_vis, (0, 0), (img.shape[1], split_y1-10), (0, 0, 255), 2)  # Red ROI
            cv.putText(roi_vis, "Blue Shift", (10, split_y2+30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(roi_vis, "Red Shift", (10, split_y1-20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
        else:
            # Fallback split visualization
            mid_y = img.shape[0] // 2
            cv.rectangle(roi_vis, (0, mid_y), (img.shape[1], img.shape[0]), (255, 0, 0), 2)
            cv.rectangle(roi_vis, (0, 0), (img.shape[1], mid_y), (0, 0, 255), 2)
            cv.putText(roi_vis, "Blue Shift (Fallback)", (10, mid_y+30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(roi_vis, "Red Shift (Fallback)", (10, mid_y-20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Plot ROI visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(roi_vis)
        plt.title("EPW ROIs Visualization")
        plt.axis('off')
        plt.show()

 
        return lineout_end, lineout_start, blue_min, blue_max, red_min, red_max

    # Main logic
    results = {}


    # use normalized img to find corners 
    def find_filter_corners(img):
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv.erode(img, kernel, iterations=1)

        # Show the eroded image
        plt.imshow(eroded)
        plt.title("Eroded Image")
        plt.axis('off')
        plt.show()

        corners = cv.goodFeaturesToTrack(eroded, 100, 0.01, 10)
        corners = np.int0(corners).reshape(-1, 2)

        # Filter corners: only keep corners that have at least one other corner within the max distance
        filtered_corners = []
        max_distance = 100
        for i, corner in enumerate(corners):
            has_neighbor = False
            for j, other_corner in enumerate(corners):
                if i == j:
                    continue  # skip same corner
                # Euclidean distance between current corner and other corners
                distance = np.linalg.norm(corner - other_corner)
                if distance <= max_distance:
                    has_neighbor = True
                    break
            if has_neighbor:
                filtered_corners.append(corner)
            


        return np.array(filtered_corners)
    # Function to calculate min and max coordinates
    def calculate_min_max(corners):
        if corners.size > 0:
            min_x = np.min(corners[:, 0])  # Minimum x-coordinate
            max_x = np.max(corners[:, 0])  # Maximum x-coordinate
            min_y = np.min(corners[:, 1])  # Minimum y-coordinate
            max_y = np.max(corners[:, 1])  # Maximum y-coordinate

            print(f"Min x: {min_x}, Max x: {max_x}")
            print(f"Min y: {min_y}, Max y: {max_y}")

            # Add a margin of error
            min_x -= 20
            max_x += 20
            min_y -= 20
            max_y += 20

            return min_x, max_x, min_y, max_y
        else:
            print("No filtered corners found.")
            return None, None, None, None
        
    # object detection with canny 

    def detect_objects_edges(img):
        """
        Find objects using Canny edge detection with integrated visualization
        Returns list of bounding boxes (x, y, w, h)
        """
        # Set default parameters since config doesn't have edge_params
        gaussian_kernel = (5, 5)  # default blur kernel
        canny_thresholds = (50, 150)  # default Canny thresholds
        min_area = 400  # default minimum contour area

        # Create visualization image
        if len(img.shape) == 2:  # Grayscale
            vis_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:  # Color
            vis_img = img.copy()

        # Edge detection pipeline
        blurred = cv.GaussianBlur(img, gaussian_kernel, 0)
        edges = cv.Canny(blurred, *canny_thresholds)
        
        # Find and draw contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h))
                # Draw bounding box in green
                cv.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Overlay Canny edges in red
        vis_img[edges > 0] = [0, 0, 255]  # Set edge pixels to red

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.imshow(cv.cvtColor(vis_img, cv.COLOR_BGR2RGB))  # Convert for matplotlib
        plt.title("Canny Edge Detection Results\n(Red: Edges, Green: Bounding Boxes)")
        plt.axis('off')
        plt.show()

        return bounding_boxes
    def canny_edges(img):

        edges =feature.canny(img, sigma=3)
        plt.figure(figsize=(10, 6))
        plt.imshow(edges, cmap='gray')
        plt.show()
        return edges
    
    def lapplaciam(img):
        laplacian_gaussian = ndimage.gaussian_laplace(img, sigma=1)
        gaussian_gradient = ndimage.gaussian_gradient_magnitude(img, sigma=1)
        fouriesgaussian = ndimage.fourier_gaussian(img, sigma=1)

        x_prewitt = ndimage.prewitt(img, axis=0)
        y_prewitt = ndimage.prewitt(img, axis=1)

        img_canny = np.hypot(x_prewitt, y_prewitt)
        img_canny *= 255.0 / np.max(img_canny)




        fig, axes = plt.subplots(nrows=1,ncols=5, figsize=(30, 30))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original Image")
        axes[1].imshow(laplacian_gaussian, cmap='gray')
        axes[1].set_title("Laplacian of Gaussian")
        axes[2].imshow(gaussian_gradient, cmap='gray')
        axes[2].set_title("Gaussian Gradient")
        axes[3].imshow(fouriesgaussian, cmap='gray')
        axes[3].set_title("Fourier Gaussian")
        axes[4].imshow(img_canny, cmap='prism')
        axes[4].set_title("Canny prism")
        for ax in axes:
            ax.axis('off')
        plt.show()
        return img_canny



    if config["other"]["extraoptions"]["load_ele_spec"]: #plotting the epw spec wi/out lineouts
        X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.pcolormesh(
            X,
            Y,
            elecData,
            cmap="hot",
            vmin=0,
            vmax=0.15*np.amax(elecData)
        )
        """fig2_path = os.path.join(tdir, 'temp_epw.png') #path for epw spectra 
        fig2.savefig(fig2_path, bbox_inches='tight') #saving fig in temp dir
        #openinn iaw and geting roi
        epw1 = cv.imread(fig2_path)"""
        img_epw = elecData[70:390,10:497]

        # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
        gray_epw =cv.cvtColor(img_epw,cv.COLOR_BGR2GRAY)
        ret,bw_epw = cv.threshold(gray_epw, 127, 255, cv.THRESH_BINARY)  # unpack the result from Adaptative threshold into two variables
        kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

        eroded = cv.erode(bw_epw, kernel, iterations = 2) #bw_iaw holds the eroded image
        corners_epw = cv.goodFeaturesToTrack(eroded,20,0.01,20) # eroded is now single channel
        corners_epw = np.int0(corners_epw)

        #list for corners coordinates
        corner_list_epw = []
        for i in corners_epw:
            x,y = i.ravel()
            cv.circle(img_epw,(x,y),3,255,-1)
            corner_list_epw.append((x,y))

        #Finding the min and max x and y points of the feature, and mapping the to the original image
        corner_list_epw.sort()  # Sort in-place by x-coordinate (default behavior)
        x_min = corner_list_epw[0][0]  # Extract only the x-value
        OGX_min = x_min + 0
        x_max = corner_list_epw[-1][0]  # Extract only the x-value
        OGx_max = x_max + 10
        corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
        y_min = corner_list_epw[0][1]  # Extract only the y-value
        OGy_min = y_min + 60
        y_max = corner_list_epw[-1][1] # Extract only the y-value
        OGy_max = y_max + 70

        #maping findings to their corresponding parameters
        lineout_start = float(OGX_min)
        lineout_end = float(OGx_max)
        red_max = float(OGy_max)
        blue_min = float(OGy_min)

    if config["data"]["estimate_lineouts_iaw"]:
        [ lineout_end,lineout_start,iaw_cf,iaw_max,iaw_min] = first_guess(elecData, ionData,config)
        config["data"]["lineouts"]["start"] = lineout_start
        config["data"]["lineouts"]["end"] = lineout_end
        config["data"]["fit_rng"]["iaw_min"] = iaw_min
        config["data"]["fit_rng"]["iaw_max"] = iaw_max
        config["data"]["fit_rng"]["iaw_cf_min"] = iaw_cf
    if config["data"]["estimate_lineouts_epw"]:
        [ lineout_end,lineout_start] =first_guess(elecData, ionData, config)
        config["data"]["lineouts"]["start"] = lineout_start
        config["data"]["lineouts"]["end"] = lineout_end
    
    print(config["data"]["lineouts"]["start"])
    print(config["data"]["lineouts"]["end"])
    print(config["data"]["lineouts"]["skip"])

    return red_max,blue_min,lineout_end,lineout_start,iaw_max,iaw_min, iaw_cf,x_min,x_max,y_max,y_min

#make the function return the outputs, theese will be used later by another function 

    def notch_filter(img):
        # assuming the notch filter is located at 528 +-12 pixels 

        notch_fliter_start = 516
        notch_filter_end = 540 

        if config["data"]["background"]["type"] == "pixel":
            notch_fliter_start = notch_fliter_start - 200
            notch_filter_end = notch_filter_end - 200
        else: 
            notch_fliter_start = notch_fliter_start
            notch_filter_end = notch_filter_end
        
        epw_red_box = img[:notch_fliter_start,:]
        epw_blue_box = img[notch_filter_end:,:]

        return epw_red_box, epw_blue_box




        gray_blue = np.dot(epw_blue_box[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        plt.imshow(gray_blue)
        plt.show()    

        kernel = np.ones((2, 2), np.uint8)
        eroded_blue = cv.erode(gray_blue, kernel, iterations=1)
        plt.imshow(eroded_blue)
        plt.show()




        #plt.imshow(epw_red_box, cmap='gray')
        #plt.show()

        plt.imshow(epw_blue_box, cmap='gray')
        plt.show()

        red_corners= cv.goodFeaturesToTrack(epw_red_box, 50, 0.01, 10)
        red_corners = np.int0(red_corners).reshape(-1, 2)
        #plt.imshow(edges_red, cmap='gray')
        #plt.show()

        contast_blue = cv.convertScaleAbs(epw_blue_box, alpha = 1.5 )
        plt.imshow(contast_blue, cmap='gray')
        plt.show()

        blur_blue = cv.GaussianBlur(contast_blue,(5,5),0)
        plt.imshow(blur_blue, cmap='gray')
        plt.show()

        blue_cany = feature.canny(blur_blue,sigma = 2)
        plt.imshow(blue_cany, cmap='gray')                                                 
        plt.show()

        invert_contrast = invert(contast_blue)
        plt.imshow(invert_contrast, cmap='gray')
        plt.show()

        canny_invert = feature.canny(invert_contrast,sigma = 2)
        plt.imshow(canny_invert, cmap='gray')
        plt.show()


        inver_blue = invert(contast_blue)
        plt.imshow(inver_blue, cmap='gray')
        plt.show()

        kernel = np.ones((2, 2), np.uint8)
        erode_blue = cv.erode(inver_blue, kernel, iterations=1)
        plt.imshow(erode_blue, cmap='gray')
        plt.show()

        blur_eroded = cv.GaussianBlur(erode_blue,(5,5),0)
        plt.imshow(blur_eroded, cmap='gray')
        plt.show()




        edges_blue = cv.goodFeaturesToTrack(blur_eroded, 50, 0.01, 10)
        edges_blue = np.int0(edges_blue).reshape(-1, 2)
        for coreners in edges_blue:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #plt.imshow(edges_blue, cmap='gray')
        #plt.show()

        for corners in red_corners: 
            x,y = corners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return epw_red_box, epw_blue_box
    
       # return epw_red_box, epw_blue_box

        #erode the image 

        #gaussian blur
        
        blur_blue = cv.GaussianBlur(epw_blue_box,(21,21),0)
        blur_red = cv.GaussianBlur(epw_red_box,(21,21),0)

        plt.imshow(blur_blue, cmap='gray')
        plt.show()

        plt.imshow(blur_red, cmap='gray')
        plt.show()

        #try finding corners here 

    
        edges_blue = cv.goodFeaturesToTrack(blur_blue, 30, 0.5, 10)
        edges_blue = np.int0(edges_blue).reshape(-1, 2)
        for coreners in edges_blue:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners 0.5",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #plt.imshow(edges_blue, cmap='gray')
        #plt.show()

        edges_red = cv.goodFeaturesToTrack(blur_red, 30, 0.5, 10)
        edges_red = np.int0(edges_red).reshape(-1, 2)
        for corners in edges_red: 
            x,y = corners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners 0.5",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        invert_blue = invert(blur_blue)
        invert_red = invert(blur_red)

        plt.imshow(invert_blue, cmap='gray')
        plt.show()
        plt.imshow(invert_red, cmap='gray')
        plt.show()
        # try finding corners here 


        edges_blue2 = cv.goodFeaturesToTrack(invert_blue, 30, 0.35, 10)
        edges_blue2 = np.int0(edges_blue2).reshape(-1, 2)
        for coreners in edges_blue2:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #plt.imshow(edges_blue, cmap='gray')
        #plt.show()

        edges_red2 = cv.goodFeaturesToTrack(invert_red, 30, 0.35, 10)
        edges_red2 = np.int0(edges_red2).reshape(-1, 2)
        for corners in edges_red2: 
            x,y = corners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()

        canny_blur_blue = feature.canny(blur_blue,sigma = 1)
        canny_blur_red = feature.canny(blur_red,sigma = 1)
        plt.imshow(canny_blur_blue, cmap='gray')
        plt.show()
        plt.imshow(canny_blur_red, cmap='gray')
        plt.show()


        threshold_blue = cv.threshold(blur_blue, 127, 255, cv.THRESH_BINARY )
        threshold_red = cv.threshold(blur_red, 127, 255, cv.THRESH_BINARY )

        plt.imshow(threshold_blue[1], cmap='gray')
        plt.show()
        plt.imshow(threshold_red[1], cmap='gray')
        plt.show()

        kernel = np.ones((2, 2), np.uint8)
        contrast_blue = cv.convertScaleAbs(blur_blue, alpha = 1.5 )
        contrast_red = cv.convertScaleAbs(blur_red, alpha = 1.5 )

        plt.imshow(contrast_blue, cmap='gray')
        plt.show()
        plt.imshow(contrast_red, cmap='gray')
        plt.show()
        invert_blue = invert(contrast_blue)
        invert_red = invert(contrast_red)

        plt.imshow(invert_blue, cmap='gray')
        plt.show()
        plt.imshow(invert_red, cmap='gray')
        plt.show()
        # try finding corners here 


        corners_blue = cv.goodFeaturesToTrack(invert_blue, 20, 0.5, 10)
        corners_blue = np.int0(corners_blue).reshape(-1, 2)
        for coreners in corners_blue:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #plt.imshow(edges_blue, cmap='gray')
        #plt.show()

        corners_red = cv.goodFeaturesToTrack(invert_red, 20, 0.5, 10)
        corners_red = np.int0(corners_red).reshape(-1, 2)
        for corners in corners_red: 
            x,y = corners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()


       #lets try to binarize the image 

        biniary_blue = cv.adaptiveThreshold(invert_blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        binary_red = cv.adaptiveThreshold(invert_red, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        plt.imshow(biniary_blue, cmap='gray')
        plt.show()
        plt.imshow(binary_red, cmap='gray')
        plt.show()

        #try finding corners here 

        binary_corners_blue = cv.goodFeaturesToTrack(biniary_blue, 20, 0.1, 10)
        binary_corners_blue = np.int0(binary_corners_blue).reshape(-1, 2)
        for coreners in binary_corners_blue:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()

        binary_corners_red = cv.goodFeaturesToTrack(binary_red, 20, 0.1, 10)
        binary_corners_red = np.int0(binary_corners_red).reshape(-1, 2) 
        for conerners in binary_corners_red:
            x,y = conerners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        

        canny_binary_blue = feature.canny(biniary_blue,sigma = 3)
        canny_binary_red = feature.canny(binary_red,sigma = 3)
        plt.imshow(canny_binary_blue, cmap='gray')
        plt.show()
        plt.imshow(canny_binary_red, cmap='gray')
        plt.show()











        gray_blue = np.dot(epw_blue_box[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        plt.imshow(gray_blue)
        plt.show()    

        kernel = np.ones((2, 2), np.uint8)
        eroded_blue = cv.erode(gray_blue, kernel, iterations=1)
        plt.imshow(eroded_blue)
        plt.show()




        #plt.imshow(epw_red_box, cmap='gray')
        #plt.show()

        plt.imshow(epw_blue_box, cmap='gray')
        plt.show()

        red_corners= cv.goodFeaturesToTrack(epw_red_box, 50, 0.01, 10)
        red_corners = np.int0(red_corners).reshape(-1, 2)
        #plt.imshow(edges_red, cmap='gray')
        #plt.show()

        contast_blue = cv.convertScaleAbs(epw_blue_box, alpha = 1.5 )
        plt.imshow(contast_blue, cmap='gray')
        plt.show()

        blur_blue = cv.GaussianBlur(contast_blue,(5,5),0)
        plt.imshow(blur_blue, cmap='gray')
        plt.show()

        blue_cany = feature.canny(blur_blue,sigma = 2)
        plt.imshow(blue_cany, cmap='gray')                                                 
        plt.show()

        invert_contrast = invert(contast_blue)
        plt.imshow(invert_contrast, cmap='gray')
        plt.show()

        canny_invert = feature.canny(invert_contrast,sigma = 2)
        plt.imshow(canny_invert, cmap='gray')
        plt.show()


        inver_blue = invert(contast_blue)
        plt.imshow(inver_blue, cmap='gray')
        plt.show()

        kernel = np.ones((2, 2), np.uint8)
        erode_blue = cv.erode(inver_blue, kernel, iterations=1)
        plt.imshow(erode_blue, cmap='gray')
        plt.show()

        blur_eroded = cv.GaussianBlur(erode_blue,(5,5),0)
        plt.imshow(blur_eroded, cmap='gray')
        plt.show()




        edges_blue = cv.goodFeaturesToTrack(blur_eroded, 50, 0.01, 10)
        edges_blue = np.int0(edges_blue).reshape(-1, 2)
        for coreners in edges_blue:
            x,y = coreners.ravel()
            cv.circle(epw_blue_box,(x,y),2,255,-1)
        cv.imshow("blue corners",epw_blue_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        #plt.imshow(edges_blue, cmap='gray')
        #plt.show()

        for corners in red_corners: 
            x,y = corners.ravel()
            cv.circle(epw_red_box,(x,y),2,255,-1)
        cv.imshow("red corners",epw_red_box)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return epw_red_box, epw_blue_box





def get_lambda(config,img_iaw,img_epw,OGy_min,OGy_max,x_min,x_max,y_max,y_min):
 

#if the get boolean for calculating the lambda fro iaw is true do the following 

    if config["data"]["estimate_lambda_iaw"]:
        iaw_box = img_iaw[y_min:y_max,x_min:x_max] # use min and max x.y values found with detector to selelct the ROI
        edges = feature.canny(iaw_box,sigma=3) #canny edge detector
        indices = np.where(edges != [0]) 
        coordinates = list(zip(indices[1],indices[0])) #make a list of coordinates with the found points
        coordinates.sort(key=lambda point: point[1]) #sort by y value 
        edge_y_min = coordinates[0][1]
        edge_y_max = coordinates[-1][1] 
        iaw_cf = (edge_y_max+ edge_y_min)/2
        #dictyionaries for storing upper and lower half coordinates 
        upper_half = {} 
        lower_half ={}

        for x, y in coordinates:
            if y < iaw_cf:
                if x not in upper_half:
                    upper_half[x] = []
                upper_half[x].append(y)
            else:
                if x not in lower_half:
                    lower_half[x]= []
                lower_half[x].append(y)

        results_upper = {}
        upper_midpoint = {}

        for x, y_values in upper_half.items():
            min_y = min(y_values)
            max_y = max(y_values)
            mid_point = (min_y + max_y) / 2
            upper_midpoint[x] = mid_point
            results_upper[x] = {"min_y": min_y, "max_y": max_y, "mid_point": mid_point}
        
        results_lower = {}
        lower_midpoint ={}
        # Find min and max y-values for each x-group in the lower half and calculate midpoints
        for x, y_values in lower_half.items():
            min_y = min(y_values)
            max_y = max(y_values)
            mid_point = (min_y + max_y) / 2
            lower_midpoint[x] = mid_point
            results_lower[x] = {"min_y": min_y, "max_y": max_y, "mid_point": mid_point}

        






        X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
        fig1, ax1 = plt.subplots()
        ax1.axis('off')
        ax1.pcolormesh(
            X,
            Y,
            ionData,
            cmap="gray",
            vmin=0,
            vmax=0.05*np.amax(ionData),
        )
        #fig1_path = os.path.join(tdir,'temp_iaw.png') #path for iaw plot 
        #fig1.savefig(fig1_path , bbox_inches='tight') #save in temp dir
        #openinn iaw and geting roi
        #iaw1 = cv.imread(fig1_path)

        # using fig without a temp directory 
        
        img_iaw = ionData[70:390,10:497]# to account for time feducials

        #img_iaw = ionData.shape[:2] # if there are no feducials

        # convert the img to grayscale (migth delete later if the image is ploter in gray scale)
        #gray_iaw =cv.cvtColor(img_iaw,cv.COLOR_BGR2GRAY)
        max_gray_value = np.max(img_iaw) #find the highest value from the gray scale image 
        threshold_value = 0.5 * max_gray_value #set the threshold to 50% of the maximum value 
        ret,bw_iaw = cv.threshold(img_iaw,threshold_value, 255, cv.THRESH_BINARY)  # apply thresholdin0g
        kernel = np.ones((2,2),np.uint8 ) #kernel for eroding

        eroded = cv.erode(bw_iaw, kernel, iterations = 2) #bw_iaw holds the eroded image
        corners_iaw = cv.goodFeaturesToTrack(eroded,20,0.01,20) # eroded is now single channel
        corners_iaw = np.int0(corners_iaw)

        #list for corners coordinates
        corner_list_iaw = []
        for i in corners_iaw:
            x,y = i.ravel()
            cv.circle(img_iaw,(x,y),3,255,-1)
            corner_list_iaw.append((x,y))

        #Finding the min and max x and y points of the feature, and mapping the to the original image
        corner_list_iaw.sort()  # Sort in-place by x-coordinate (default behavior)
        x_min = corner_list_iaw[0][0]  # Extract only the x-value
        OGX_min = x_min + 0
        x_max = corner_list_iaw[-1][0]  # Extract only the x-value
        OGx_max = x_max + 10
        corner_list_iaw.sort(key=lambda point: point[1])  # Sort in-place by y-coordinate
        y_min = corner_list_iaw[0][1]  # Extract only the y-value
        OGy_min = y_min + 60
        y_max = corner_list_iaw[-1][1] # Extract only the y-value
        OGy_max = y_max + 70

        #maping findings to their corresponding parameters
        lineout_start = OGX_min
        lineout_end = OGx_max
        iaw_max = OGy_max
        iaw_min = OGy_min
        iaw_cf = (OGy_min + OGy_max)/2






    x_start = config["data"]["lineouts"]["start"]
    x_end = config["data"]["lineouts"]["end"]
    skips = config["data"]["lineouts"]["skip"]
    delta_x = x2 - x1 
    #lineout_size = delta_x / config["data"]["lineouts"]["skip"]
    #for lineout_size in range(delta_x):
    
    chuncks = [i for i in range (x_start,x_end,skips)]
    first_chuck = chuncks[0],chuncks[1]
    first_chunk_epw = img_epw[chuncks[0]:chuncks[1],OGy_min:OGy_max]
    first_chunck_iaw = img_iaw[OGy_max:OGy_min,chuncks[0],chuncks[1]]

    if config["data"]["estimate_lambda"]:
        inside_box = ionData
        for i in range(chuncks):
            chunk_iaw = img_iaw[OGy_max:OGy_min,chuncks[i],chuncks[i+1]]










"""  FIRST ATTEMPT AT MAKING AND TEMP SAVING THE PLOTS
    with tempfile.TemporaryDirectory() as tdir:
        #polting figure like data_visualizer.py but without the lineouts 
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
            fig, ax1 = plt.subplots()
            ax1.axis('off')
            cb = ax.pcolormesh(
                X,
                Y,
                ionData,
                cmap="gray",
                vmin=0,
                vmax=0.05*np.amax(ionData),
            )
            fig.savefig(tdir + '/temp_iaw.png',bbox_inches='tight')
            plt.close(fig)
        if config["other"]["extraoptions"]["load_ele_spec"]:
                X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

                fig, ax = plt.subplots()
                jc= ax.pcolormesh(
                    X,
                    Y,
                    elecData,
                    cmap="hot",
                    vmin=0,
                    vmax=0.15*np.amax(elecData)
                )
                fig.savefig(tdir + '/temp_epw.png', bbox_inches='tight')

   

        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "plots"), exist_ok=True)
            # until this can be made interactive this plots all the data regions
            if config["other"]["extraoptions"]["load_ion_spec"]:
                X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

                fig, ax = plt.subplots()
                cb = ax.pcolormesh(
                    X,
                    Y,
                    ionData,
                    cmap="gray",
                    #vmin=np.amin(ionData),
                    vmin=0,
                    vmax=0.05*np.amax(ionData),
                )
                fig.savefig(os.path.join(td, "ion1.png"), bbox_inches="tight")

            if config["other"]["extraoptions"]["load_ele_spec"]:
                X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

                fig, ax = plt.subplots()
                jc= ax.pcolormesh(
                    X,
                    Y,
                    elecData,
                    #norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=0, vmax= np.amax(elecData)),
                    cmap="hot",
                    shading= "auto",
                    #vmin=np.amin(elecData),
                    vmin=0,
                    vmax=0.15*np.amax(elecData)
                )
                fig.savefig(os.path.join(td, "plots", "electron_fit_ranges.png"), bbox_inches="tight")

            mlflow.log_artifacts(td)
    #plotting the raw data without lineouts using functions from data visualizer 
#def plot_raw_data(): """



