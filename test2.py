import cv2 as cv
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filter(img):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
    sobel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=np.float64)
    img_x = cv.filter2D(img, -1, sobel_x)
    img_y = cv.filter2D(img, -1, sobel_y)
    sobel_combined = np.sqrt(img_x**2 + img_y**2) # Calculating gradient
    theta = np.arctan2(img_y, img_x)
    edges = sobel_combined.astype(np.uint8) # Normalizing the values of pixels
    # theta = np.arctan2(sobel_y, sobel_x)
    return (edges, theta)

def non_max_suppression(img, theta): # Here D is the angle matrix calculated using theta in above sobel_filter function
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.float64) # Initialize matrix of same size as image with zeroes 
    angle = theta * 180 / np.pi # Convert in degrees
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
                grad_dir = angle[i,j]
                r = 255 # These are the pixel values to be given to black pixels for suppressing the edges
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    neighbors = [(i,j+1), (i,j-1)]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    neighbors = [(i-1, j+1), (i+1, j-1)]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    neighbors = [(i-1,j), (i+1,j)]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    neighbors = [(i-1, j-1), (i+1, j+1)]

                # Compare magnitude with neighbors
                if img[i, j] >= max(img[n[0], n[1]] for n in neighbors):
                    Z[i, j] = r
                else:
                    Z[i,j] = 0
    
    return Z

img = cv.imread("./table.png")
# cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_ary = np.array(grayscale_img, dtype=np.float64) # Image has been converted to np array with pixel values as floating point numbers
blurred_image = cv.filter2D(img_ary, -1, gaussian_kernel(5, sigma=1.0))
# grad = sobel_filter(blurred_image)
edges, theta = sobel_filter(blurred_image)
print(edges)
suppressed_img = non_max_suppression(edges, theta)
cv.imshow('Non_max_suppression', suppressed_img)
cv.waitKey(0)  
cv.destroyAllWindows() 