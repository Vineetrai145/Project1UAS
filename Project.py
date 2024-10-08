import os
import cv2 as cv
import numpy as np

def detect_triangles(image): #funtion to detect and count no. of triangles
    blur = cv.GaussianBlur(image, (3, 3), 3)
    filter = cv.bilateralFilter(blur, 15, 120, 75)
    canny = cv.Canny(filter, 50, 150)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    triangle_count = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 100 :
            continue
        arclength = cv.arcLength(contour, True)
        accuracy = 0.02 * arclength
        approx = cv.approxPolyDP(contour, accuracy, True)
        is_closed = cv.isContourConvex(approx)
        if len(approx) == 3 and is_closed:
            triangle_count += 1
    return triangle_count


folder_path = r'C:\Users\Vineet Rai\OneDrive\Desktop\project UAS\uas takimages'

# Initialize an empty list to store house results
n_houses = []
n_priority_houses=[]
n_priority_ratio=[]
img_dic ={}


# Loop through the folder and process each image
for i, image_file in enumerate(os.listdir(folder_path)):
    # Only process image files (you can add more extensions if needed)
    if image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
        # Load each image
        image_path = os.path.join(folder_path, image_file)
        image = cv.imread(image_path)
        
        # Convert the image from BGR to HSV
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        #lower and upper bounds for green , brown , blue and red color
        lower_green = np.array([30, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)

        lower_brown = np.array([0, 30, 10], dtype=np.uint8)
        upper_brown = np.array([35, 255, 250], dtype=np.uint8)

        lower_red = np.array([0, 50, 50], dtype=np.uint8)
        upper_red = np.array([10, 255, 255], dtype=np.uint8)

        lower_blue = np.array([90, 50, 50], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)

        # creating masks
        mask = cv.inRange(hsv_image, lower_green, upper_green)
        mask1 = cv.inRange(hsv_image, lower_brown, upper_brown)
        mask2 = cv.inRange(hsv_image, lower_red, upper_red)
        mask3 = cv.inRange(hsv_image, lower_blue, upper_blue)

        # Filling color on each masked part (yellow, blue, red, dark blue)
        fill_Color = np.zeros_like(image)
        fill_Color[:] = (107, 224, 250)
        yellow = cv.bitwise_and(fill_Color, fill_Color, mask=mask)

        fill_Color[:] = (223, 230, 46)
        blue = cv.bitwise_and(fill_Color, fill_Color, mask=mask1)

        fill_Color[:] = (0, 0, 255)
        red = cv.bitwise_and(fill_Color, fill_Color, mask=mask2)

        fill_Color[:] = (255, 0, 0)
        Dark_blue = cv.bitwise_and(fill_Color, fill_Color, mask=mask3)

        # Combining the colored images
        result1 = cv.bitwise_or(red, yellow, None)
        result2 = cv.bitwise_or(Dark_blue, blue, None)
        result = cv.bitwise_or(result1, result2, None)
        cv.imshow('result',result)

        # for creating  triangles mask in only green region 
        g_blur = cv.GaussianBlur(yellow, (3, 3), 3)
        g_filter = cv.bilateralFilter(g_blur, 15, 100, 75)
        g_canny = cv.Canny(g_filter, 50, 150)
        g_contours, _ = cv.findContours(g_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in g_contours:
            area = cv.contourArea(contour)
            if area < 100 :
                continue
            arclength = cv.arcLength(contour, False)
            accuracy = 0.02 * arclength
            approx = cv.approxPolyDP(contour, accuracy, True)
            is_closed = cv.isContourConvex(approx)
            if len(approx) == 3 and is_closed:
                cv.drawContours(g_canny, [contour], 0, (255, 255, 255), thickness=cv.FILLED)
                
        # for creating  triangles mask in only brown region 
        b_blur = cv.GaussianBlur(blue, (3, 3), 3)
        b_filter = cv.bilateralFilter(b_blur, 15, 120, 75)
        b_canny = cv.Canny(b_filter, 50, 250)
        b_contours, _ = cv.findContours(b_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in b_contours:
            area = cv.contourArea(contour)
            if area < 100 :
                continue
            arclength = cv.arcLength(contour, True)
            accuracy = 0.02 * arclength
            approx = cv.approxPolyDP(contour, accuracy, True)
            is_closed = cv.isContourConvex(approx)
            if len(approx) == 3 and is_closed:
                cv.drawContours(b_canny, [contour], 0, (255, 255, 255), thickness=cv.FILLED)

        #using greena and brown triangel mask
        gr_area = cv.bitwise_and(result, result, mask=g_canny)
        br_area = cv.bitwise_and(result, result, mask=b_canny)

        mask4 = cv.inRange(cv.cvtColor(gr_area, cv.COLOR_BGR2HSV), lower_red, upper_red)
        mask5 = cv.inRange(cv.cvtColor(gr_area, cv.COLOR_BGR2HSV), lower_blue, upper_blue)
        mask6 = cv.inRange(cv.cvtColor(br_area, cv.COLOR_BGR2HSV), lower_red, upper_red)
        mask7 = cv.inRange(cv.cvtColor(br_area, cv.COLOR_BGR2HSV), lower_blue, upper_blue)

        gr_ar_Tcount = detect_triangles(mask) 
        br_ar_Tcount = detect_triangles(mask1) 

        houses = [br_ar_Tcount,gr_ar_Tcount]
        n_houses.append(houses)    
        
        # counting red and blue color triangle on each region
        gr_area_red =detect_triangles(mask4)
        gr_area_blue=detect_triangles(mask5)
        br_area_red = detect_triangles(mask6)
        br_area_blue =detect_triangles(mask7)

        gr_area_houses=np.array([gr_area_red,gr_area_blue])
        br_area_houses=np.array([br_area_red,br_area_blue])

        #calculating priority of each color region
        priority_value = np.array([[1], [2]])
        gr_house_priority = np.dot(gr_area_houses, priority_value)
        br_house_priority = np.dot(br_area_houses, priority_value)

        priority = np.zeros(2,dtype='int32')
        priority[0] = br_house_priority.item()
        priority[1] = gr_house_priority.item()
        n_priority_houses.append(priority)
        
        priority_ratio =priority[0]/priority[1] # calulating priority ratio for each image
        n_priority_ratio.append(priority_ratio)
        
        img_dic[image_file]=priority_ratio
        cv.waitKey(0)
        
print(n_houses)
n_priority_houses_list =[arr.tolist() for arr in n_priority_houses] # converting array to list
print(n_priority_houses_list)
n_priority_ratio_list =[arr.tolist() for arr in n_priority_ratio]
print(n_priority_ratio_list)


image_by_rescue_ratio = dict(sorted(img_dic.items(), key=lambda item: item[1], reverse=True))
print(image_by_rescue_ratio.keys())


cv.waitKey(0)
cv.destroyAllWindows()

