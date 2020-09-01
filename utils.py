import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def appendimages(im1, im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.vstack((im1, np.zeros((rows2-rows1, im1.shape[1], 3))))
    elif rows1 > rows2:
        im2 = np.vstack((im2, np.zeros((rows1-rows2, im2.shape[1], 3))))
    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, ok):
    im3 = appendimages(im1, im2)
    plt.imshow(im3[:,:,::-1])
    plt.plot(locs1[0,~ok], locs1[1,~ok], 'r.')
    plt.plot(locs2[0,~ok]+im1.shape[1], locs2[1,~ok], 'r.')
    plt.plot(locs1[0,ok], locs1[1,ok], 'b.')
    plt.plot(locs2[0,ok]+im1.shape[1], locs2[1,ok], 'b.')
    for i in range(len(ok)):
        if ok[i]:
            plt.plot([locs1[0,i], locs2[0,i]+im1.shape[1]], [locs1[1,i], locs2[1,i]], 'b-')

def cv_draw_matches(im1, im2, locs1, locs2, ok):
    im3 = appendimages(im1, im2)
    for i in range(locs1.shape[1]):
        center1 = (int(round(locs1[0,i])), int(round(locs1[1,i])))
        center2 = (int(round(locs2[0,i]+im1.shape[1])), int(round(locs2[1,i])))
        if ok[i]==0:
            cv.circle(im3, center1, 3, (0,0,255), -1, cv.LINE_AA)
            cv.circle(im3, center2, 3, (0,0,255), -1, cv.LINE_AA)
        else:
            cv.circle(im3, center1, 3, (255,0,0), -1, cv.LINE_AA)
            cv.circle(im3, center2, 3, (255,0,0), -1, cv.LINE_AA)
            cv.line(im3, center1, center2, (255,0,0), 1, cv.LINE_AA)
    return im3


def cell(m, n=None):
    a = []
    for i in range(m):
        a.append([])
        if n is None:
            for j in range(m):
                a[i].append(None)
        else:
            for j in range(n):
                a[i].append(None)
    return a

def unique(A):
    ar, idx = np.unique(A, return_index=True, axis=1)
    return ar, idx
