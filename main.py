import cv2 as cv
from load_data import load_data
from comp_KR import comp_KR
from mosaic_global import mosaic_global
from mosaic_local_ori import mosaic_local_ori
import matplotlib.pyplot as plt

# data_path = 'images/APAP-railtracks/'
# imfile1 = data_path + 'railtracks_01.jpg'
# imfile2 = data_path + 'railtracks_02.jpg'

data_path = 'images/DHW-temple/'
imfile1 = data_path + '4.jpg'
imfile2 = data_path + '5.jpg'

# data_path = 'images/ANAP-intersection/'
# imfile1 = data_path + 'intersection_01.jpg'
# imfile2 = data_path + 'intersection_02.jpg'

# data_path = 'images/REW_worktable/'
# imfile1 = data_path + 'worktable_01.jpg'
# imfile2 = data_path + 'worktable_02.jpg'

im1 = cv.imread(imfile1)
im2 = cv.imread(imfile2)

X1, X2 = load_data(im1, im2)
print("Number of matches: ", X1.shape[1])

H, X1_ok, X2_ok = comp_KR(im1, im2, X1, X2)
print("H =\n", H)
print("H normalize =\n", H/H[2,2])

mosaic = mosaic_global(im1, im2, H)
cv.imwrite(data_path+'mosaic_global.jpg', mosaic)
cv.imshow("mosaic_global", mosaic)
cv.waitKey(1)

mosaic = mosaic_local_ori(im1, im2, H, X1_ok, X2_ok)    
cv.imwrite(data_path+'mosaic_REW.jpg', mosaic)
cv.imshow("mosaic_REW", mosaic)
cv.waitKey(0)
