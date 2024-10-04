import cv2 as cv
import numpy as np

# Load an image
image = cv.imread('uas takimages\\1.png')

# Convert the image from BGR to HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#lower and upper bounds for green , brown , blue and red color
lower_green = np.array([30, 40, 40],dtype='uint8')     
upper_green = np.array([90, 255, 255],dtype='uint8')

lower_brown = np.array([[0, 30, 10]])
upper_brown = np.array([[35, 255, 250]])

lower_red = np.array([[0, 50, 50]])
upper_red = np.array([[10, 255, 255]])

lower_blue = np.array([[90, 50, 50]])
upper_blue = np.array([[130, 255, 255]])

# Create a mask with the green range
mask = cv.inRange(hsv_image, lower_green, upper_green)

# Create a mask with the brown  range
mask1 = cv.inRange(hsv_image,lower_brown,upper_brown) 

# Create a mask with the red range
mask2 = cv.inRange(hsv_image,lower_red,upper_red)

# Create a mask with the blue range
mask3= cv.inRange(hsv_image,lower_blue,upper_blue)

# filling colour on each masked part
fill_Color = np.zeros_like(image)
fill_Color[:] = (107,224,250)
yellow = cv.bitwise_and(fill_Color,fill_Color,mask=mask)

fill_Color= np.zeros_like(image)
fill_Color[:] = (223,230,46)
blue= cv.bitwise_and(fill_Color,fill_Color,mask=mask1)

fill_Color = np.zeros_like(image)
fill_Color[:] = (0,0,255)
red = cv.bitwise_and(fill_Color,fill_Color,mask=mask2)

fill_Color = np.zeros_like(image)
fill_Color[:] = (255,0,0)
Dark_blue = cv.bitwise_and(fill_Color,fill_Color,mask=mask3)

# taking OR of each Maked Colored Image 
result1 = cv.bitwise_or(blue,yellow,None)
result2 = cv.bitwise_or(red,Dark_blue,None)

result = cv.bitwise_or(result1,result2,None)
cv.imshow('result',result)

cv.waitKey(0)
cv.destroyAllWindows()

