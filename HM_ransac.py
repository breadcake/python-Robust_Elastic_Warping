import numpy as np
import random
from scipy import linalg
from utils import cell

random.seed(0)

def HM_ransac(X1, X2, Nr, min_dis):
    N = X1.shape[1]

    u = X1[0,:].reshape(-1,1)
    v = X1[1,:].reshape(-1,1)
    u_ = X2[0,:].reshape(-1,1)
    v_ = X2[1,:].reshape(-1,1)

    scale = 1 / np.mean(np.vstack((u, u_, v, v_)))
    u = u * scale
    v = v * scale
    u_ = u_ * scale
    v_ = v_ * scale

    A1 = np.hstack([np.zeros((N,3)),      -u, -v, -np.ones((N,1)), v_*u, v_*v, v_])
    A2 = np.hstack([u, v, np.ones((N,1)), np.zeros((N,3)),         -u_*u, -u_*v, -u_])
    # print(np.vstack((A1, A2)))
    if min_dis > 0:
        H = cell(Nr, 1); ok = cell(Nr, 1); score = np.zeros(Nr, 'int')
        A = cell(Nr, 1)
        for t in range(Nr):
            subset = random.sample(list(range(N)), 4)
            A[t] = np.vstack((A1[subset,:], A2[subset,:]))
            U,S,V = linalg.svd(A[t])
            h = V.T[:,8]
            H[t] = h.reshape(3,3)

            dis2 = np.dot(A1, h)**2 + np.dot(A2, h)**2
            ok[t] = dis2 < min_dis * min_dis
            score[t] = sum(ok[t])
        
        score, best = max(score), np.argmax(score)
        ok = ok[best]
        A = np.vstack((A1[ok,:], A2[ok,:]))
        U,S,V = linalg.svd(A, 0)
        h = V.T[:,8]
        H = h.reshape(3,3)
    else:
        A = np.vstack((A1, A2))
        U,S,V = linalg.svd(A, 0)
        h = V.T[:,8]
        H = h.reshape(3,3)

    H = np.dot(np.dot(np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]]), H),
               np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))

    return H, ok, score