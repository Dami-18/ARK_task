import cv2 as cv
import numpy as np

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

img = cv.imread("/table.png")
cv.imshow('Original image', img)
cv.waitKey(0)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', grayscale_img)
cv.waitKey(0) 
cv.destroyAllWindows()
blurred_image = cv.filter2D(img, -1, gaussian_kernel(5, 1))
