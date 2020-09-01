import numpy as np
from scipy import linalg
from scipy.ndimage import map_coordinates

def mosaic_global(im1, im2, H):
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
    print(u0,u1,v0,v1,mosaich, mosaicw)

    u,v = np.meshgrid(ur, vr)
    im1_p = np.zeros((mosaich, mosaicw, 3), np.uint8)
    for kc in range(3):
        im1_p[:,:,kc] = map_coordinates(im1[:,:,kc], [v, u])
    mask1 = np.ones((im1.shape[0],im1.shape[1]))
    warped_mask1 = np.ones((im1_p.shape[0], im1_p.shape[1]))
    warped_mask1 = map_coordinates(mask1, [v,u])

    z_ = H[2,0] * u + H[2,1] * v + H[2,2]
    u_ = (H[0,0] * u + H[0,1] * v + H[0,2]) / z_
    v_ = (H[1,0] * u + H[1,1] * v + H[1,2]) / z_
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