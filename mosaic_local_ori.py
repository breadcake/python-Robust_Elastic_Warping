import numpy as np
from scipy import linalg
from scipy.ndimage import map_coordinates
from utils import unique, plot_matches, cv_draw_matches
import cv2 as cv
import matplotlib.pyplot as plt

def mosaic_local_ori(im1, im2, H, X1_ok, X2_ok):
    imsize1 = im1.shape[:2]
    imsize2 = im2.shape[:2]

    # Parameters
    lambd = 0.001 * imsize1[0]*imsize1[1] # weighting parameter to balance the fitting term and the smoothing term
    intv_mesh = 10 # interval in pixels for the computing of deformation functions
    K_smooth = 5 # the smooth transition width in the non-overlapping region is set to K_smooth times of the maximum bias.

    # Mosaic
    box1 = np.array([[0, im1.shape[1]-1, im1.shape[1]-1, 0],
                     [0, 0,              im1.shape[0]-1, im1.shape[0]-1],
                     [1, 1,              1,              1]])
    box2 = np.array([[0, im2.shape[1]-1, im2.shape[1]-1, 0],
                     [0, 0,              im2.shape[0]-1, im2.shape[0]-1],
                     [1, 1,              1,              1]])
    box2_ = linalg.solve(H, box2)
    box2_[0,:] = box2_[0,:] / box2_[2,:]
    box2_[1,:] = box2_[1,:] / box2_[2,:]
    u0 = min(0, min(box2_[0,:]))
    u1 = max(im1.shape[1]-1, max(box2_[0,:]))
    ur = np.arange(u0, u1+1)
    v0 = min(0, min(box2_[1,:]))
    v1 = max(im1.shape[0]-1, max(box2_[1,:]))
    vr = np.arange(v0, v1+1)
    mosaicw = len(ur)
    mosaich = len(vr)

    # align the sub coordinates with the mosaic coordinates
    margin = 0.2 * min(imsize1[0],imsize1[1]) # additional margin of the reprojected image region
    u0_im_ = max(min(box2_[0,:]) - margin, u0)
    u1_im_ = min(max(box2_[0,:]) + margin, u1)
    v0_im_ = max(min(box2_[1,:]) - margin, v0)
    v1_im_ = min(max(box2_[1,:]) + margin, v1)
    offset_u0_ = int(np.ceil(u0_im_ - u0))
    offset_u1_ = int(np.floor(u1_im_ - u0))
    offset_v0_ = int(np.ceil(v0_im_ - v0))
    offset_v1_ = int(np.floor(v1_im_ - v0))
    imw_ = int(np.floor(offset_u1_ - offset_u0_ + 1))
    imh_ = int(np.floor(offset_v1_ - offset_v0_ + 1))

    # boundaries of the overlapping region in the image coordiantes of image 2
    box1_2 = np.dot(H, box1)
    box1_2[0,:] = box1_2[0,:] / box1_2[2,:]
    box1_2[1,:] = box1_2[1,:] / box1_2[2,:]
    sub_u0_ = max([0, min(box1_2[0,:])])
    sub_u1_ = min([imsize2[1]-1, max(box1_2[0,:])])
    sub_v0_ = max([0, min(box1_2[1,:])]) - margin
    sub_v1_ = min([imsize2[0]-1, max(box1_2[1,:])])
    print(sub_u0_, sub_u1_, sub_v0_, sub_v1_)

    # TPS
    # merge the coincided points（重合点）
    ok_nd1 = np.full(X1_ok.shape[1], False)
    _, idx1 = unique(np.round(X1_ok))
    ok_nd1[idx1] = True
    ok_nd2 = np.full(X2_ok.shape[1], False)
    _, idx2 = unique(np.round(X2_ok))
    ok_nd2[idx2] = True
    ok_nd = ok_nd1 & ok_nd2
    X1_nd = X1_ok[:, ok_nd]
    X2_nd = X2_ok[:, ok_nd]
    print("Number of non-coincident points: ", X1_nd.shape[1])
    
    # form the linear system
    x1 = X1_nd[0,:]
    y1 = X1_nd[1,:]
    x2 = X2_nd[0,:]
    y2 = X2_nd[1,:]

    z1_ = H[2,0]*x1 + H[2,1]*y1 + H[2,2]
    x1_ = (H[0,0]*x1 + H[0,1]*y1 + H[0,2]) / z1_
    y1_ = (H[1,0]*x1 + H[1,1]*y1 + H[1,2]) / z1_
    gxn = x1_ - x2
    hyn = y1_ - y2

    n = len(x1_)
    xx = np.repeat(x1_, n).reshape(n,n).T
    yy = np.repeat(y1_, n).reshape(n,n).T
    dist2 = (xx - xx.T)**2 + (yy - yy.T)**2
    dist2.ravel()[::dist2.shape[1]+1] = 1
    K = 0.5 * dist2 * np.log(dist2)
    K.ravel()[::dist2.shape[1]+1] = lambd * 8*np.pi
    K_ = np.zeros((n+3, n+3))
    K_[0:n,0:n] = K
    K_[n,  0:n] = x1_
    K_[n+1,0:n] = y1_
    K_[n+2,0:n] = np.ones(n)
    K_[0:n,  n] = x1_
    K_[0:n,n+1] = y1_
    K_[0:n,n+2] = np.ones(n)
    G_ = np.zeros((n+3,2))
    G_[0:n,0] = gxn
    G_[0:n,1] = hyn

    # solve the linear system
    W_ = linalg.solve(K_, G_)
    wx = W_[0:n,0]
    wy = W_[0:n,1]
    a = W_[n:n+3,0]
    b = W_[n:n+3,1]
    
    # remove outliers based on the distribution of weights
    outlier = (abs(wx)>3*np.std(wx)) | (abs(wy)>3*np.std(wy))

    inlier_idx = np.arange(len(x1_))
    for kiter in range(10):
        if sum(outlier) < 0.0027*n:
            break
        ok = ~outlier
        inlier_idx = inlier_idx[ok]
        K_ = K_[np.concatenate((ok, [True,True,True])),:][:,np.concatenate((ok, [True,True,True]))]
        G_ = G_[np.concatenate((ok, [True,True,True])),:]
        W_ = linalg.solve(K_, G_)
        n = len(inlier_idx)
        wx = W_[0:n,0]
        wy = W_[0:n,1]
        a = W_[n:n+3,0]
        b = W_[n:n+3,1]
        outlier = (abs(wx)>3*np.std(wx)) | (abs(wy)>3*np.std(wy))
    ok = np.full(len(x1), False)
    ok[inlier_idx] = True
    x1 = x1[ok]
    y1 = y1[ok]
    x2 = x2[ok]
    y2 = y2[ok]
    x1_ = x1_[ok]
    y1_ = y1_[ok]
    gxn = gxn[ok]
    hyn = hyn[ok]

    img3 = cv_draw_matches(im1, im2, X1_nd, X2_nd, ok)
    cv.imshow("Bayesian Refinement of Feature Matches", img3)
    cv.waitKey(1)
    # plt.figure()
    # plot_matches(im1, im2, X1_nd, X2_nd, ok)
    # plt.title("Bayesian Refinement of Feature Matches")
    # plt.show()

    # deform image
    gx = np.zeros((mosaich, mosaicw))
    hy = np.zeros((mosaich, mosaicw))
    u,v = np.meshgrid(ur,vr)

    im1_p = np.zeros((mosaich, mosaicw, 3), np.uint8)
    for kc in range(3):
        im1_p[:,:,kc] = map_coordinates(im1[:,:,kc], [v, u])
    mask1 = np.ones((im1.shape[0],im1.shape[1]))
    warped_mask1 = np.ones((im1_p.shape[0], im1_p.shape[1]))
    warped_mask1 = map_coordinates(mask1, [v,u])

    z_ = H[2,0] * u + H[2,1] * v + H[2,2]
    u_ = (H[0,0] * u + H[0,1] * v + H[0,2]) / z_
    v_ = (H[1,0] * u + H[1,1] * v + H[1,2]) / z_
    u_im_ = u_[offset_v0_:offset_v1_+1:intv_mesh][:,offset_u0_:offset_u1_+1:intv_mesh]
    v_im_ = v_[offset_v0_:offset_v1_+1:intv_mesh][:,offset_u0_:offset_u1_+1:intv_mesh]
    gx_sub = np.zeros((int(np.ceil(imh_/intv_mesh)), int(np.ceil(imw_/intv_mesh))))
    hy_sub = np.zeros((int(np.ceil(imh_/intv_mesh)), int(np.ceil(imw_/intv_mesh))))
    for kf in range(n):
        dist2 = (u_im_ - x1_[kf])**2 + (v_im_ - y1_[kf])**2
        rbf = 0.5 * dist2 * np.log(dist2)
        gx_sub = gx_sub + wx[kf]*rbf
        hy_sub = hy_sub + wy[kf]*rbf
    gx_sub = gx_sub + a[0]*u_im_ + a[1]*v_im_ + a[2]
    hy_sub = hy_sub + b[0]*u_im_ + b[1]*v_im_ + b[2]
    gx_sub = cv.resize(gx_sub, (imw_, imh_))
    hy_sub = cv.resize(hy_sub, (imw_, imh_))
    gx[offset_v0_:offset_v1_+1][:,offset_u0_:offset_u1_+1] = gx_sub
    hy[offset_v0_:offset_v1_+1][:,offset_u0_:offset_u1_+1] = hy_sub

    #smooth tansition to global transform
    eta_d0 = 0 # lower boundary for smooth transition area
    eta_d1 = K_smooth * max(abs(np.concatenate([gxn, hyn]))) # higher boundary for smooth transition area
    sub_u0_ = sub_u0_ + min(gxn)
    sub_u1_ = sub_u1_ + max(gxn)
    sub_v0_ = sub_v0_ + min(hyn)
    sub_v1_ = sub_v1_ + max(hyn)
    dist_horizontal = np.maximum(sub_u0_-u_, u_-sub_u1_)
    dist_vertical = np.maximum(sub_v0_-v_, v_-sub_v1_)
    dist_sub = np.maximum(dist_horizontal, dist_vertical)
    dist_sub = np.maximum(0, dist_sub)
    eta = (eta_d1 - dist_sub) / (eta_d1 - eta_d0)
    eta[dist_sub < eta_d0] = 1
    eta[dist_sub > eta_d1] = 0
    gx = gx * eta
    hy = hy * eta

    u_ = u_ - gx
    v_ = v_ - hy

    im2_p = np.zeros((mosaich, mosaicw, 3), np.uint8)
    for kc in range(3):
        im2_p[:,:,kc] = map_coordinates(im2[:,:,kc], [v_, u_])
    mask2 = np.ones((im2.shape[0],im2.shape[1]))
    warped_mask2 = np.ones((im2_p.shape[0], im2_p.shape[1]))
    warped_mask2 = map_coordinates(mask2, [v_,u_])

    mass = warped_mask1 + warped_mask2
    mass[mass==0]=np.nan

    mosaic = np.zeros_like(im1_p)
    for kc in range(3):
        mosaic[:,:,kc] = (im1_p[:,:,kc]*warped_mask1 + im2_p[:,:,kc]*warped_mask2) / mass

    return mosaic
