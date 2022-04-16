import numpy as np
import numpy.linalg as la

from skimage.feature import corner_harris, corner_peaks, BRIEF, match_descriptors
from skimage.transform import EssentialMatrixTransform

def get_matched_keyppoints(imLgray, imRgray):
    """
    Takes a pair of gray scale images and returns matching features from them.
    (based on harrison corners and peaks)
    """
    keypointsL = corner_peaks(corner_harris(imLgray), threshold_rel=0.001, min_distance=15)
    keypointsR = corner_peaks(corner_harris(imRgray), threshold_rel=0.001, min_distance=15)
    print ('the number of features in images 1 and 2 are {:5d} and {:5d}'.format(keypointsL.shape[0], keypointsR.shape[0]))

    extractor = BRIEF()
    extractor.extract(imLgray, keypointsL)
    keypointsL = keypointsL[extractor.mask]
    descriptorsL = extractor.descriptors
    extractor.extract(imRgray, keypointsR)
    keypointsR = keypointsR[extractor.mask]
    descriptorsR = extractor.descriptors

    matchesLR = match_descriptors(descriptorsL, descriptorsR, cross_check=True)
    print ('the number of matches is {:2d}'.format(matchesLR.shape[0]))

    return keypointsL, keypointsR, matchesLR

def matchedpoints_to_pairpoints(keypointsL, keypointsR, matchesLR):
    """
    Take key points in the left image and right image as well as an array for matched features of shape (n, 2),
    returns two arrays where ptsL[i] and ptsR[i] are a pair of matched features in left and right image
    """
    ptsL = []
    ptsR = []
    for i in matchesLR:
        ptsL.append(keypointsL[i[0]])
        ptsR.append(keypointsR[i[1]])

    ptsL = np.array(ptsL)
    ptsR = np.array(ptsR)
    return ptsL, ptsR

def camera_calib_nomalization(pts, K):
    """
    Normalization of points in two images using K (camera calibration parameters)
    """
    # a. convert original points to homogeneous 3-vectors (append "1" as a 3rd coordinate using np.append function)
    pts_homogeneous = np.column_stack((pts, np.ones(pts.shape[0])))
    # b. transform the point by applying the inverse of K
    K_inverse = la.inv(K)
    n_pts_homogenous = np.matmul(K_inverse, pts_homogeneous.T).T
    # c. convert homogeneous 3-vectors to 2-vectors (in R2)
    # since the inverse of K the last row is [0, 0, 1], the scale is not changed
    return n_pts_homogenous[:, :2]

def get_epipolar_lines(n_ptsL, n_ptsR, E, K):
    """
    Generate epipolar line equations in two images (y = ax + b)
    Given a list of matched uncalibrated points, essential matrix (E) between the two images and
    K (camera calibration parameters), return the epipolar line equations
    """
    # reate an array of homogeneous normalized points sampled in image 1
    n_ptsL_homogeneous = np.column_stack((n_ptsL, np.ones(n_ptsL.shape[0])))
    n_ptsR_homogeneous = np.column_stack((n_ptsR, np.ones(n_ptsR.shape[0])))
    # create an array of the corresponding (uncalibrated) epipolar lines in image 2
    n_epipolar_lines_im2 = np.matmul(E, n_ptsL_homogeneous.T)
    n_epipolar_lines_im1 = np.matmul(E.T, n_ptsR_homogeneous.T)
    # we know that the l in the uncalibrated coordinate, is given by K^(-T) l
    K_inverse = la.inv(K)
    epipolar_lines_im2 = np.matmul(K_inverse.T, n_epipolar_lines_im2).T
    epipolar_lines_im1 = np.matmul(K_inverse.T, n_epipolar_lines_im1).T
    # given the homogeneous representation, get the a and b
    epipolar_a_im1 = -epipolar_lines_im1[:,0] / epipolar_lines_im1[:,1]
    epipolar_a_im2 = -epipolar_lines_im2[:,0] / epipolar_lines_im2[:,1]
    epipolar_b_im1 = -epipolar_lines_im1[:,2] / epipolar_lines_im1[:,1]
    epipolar_b_im2 = -epipolar_lines_im2[:,2] / epipolar_lines_im2[:,1]
    return epipolar_a_im1, epipolar_b_im1, epipolar_a_im2, epipolar_b_im2

def P_from_E(E):
    """
    Given an essential matrix E, return the possible projection matrices
    """
    Ue, Se, Ve = la.svd(E) # gives U, S and V^T
    W = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])

    R1 = np.matmul(Ue, np.matmul(W, Ve))
    R2 = np.matmul(Ue, np.matmul(W.T, Ve))
    T1 = Ue[:, 2]
    T2 = -Ue[:, 2]

    # first camera matrix (used as referencee)
    Pw = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])

    # four possible matrices for the second camera
    Pa = np.column_stack((R1, T1))
    Pb = np.column_stack((R1, T2))
    Pc = np.column_stack((R2, T1))
    Pd = np.column_stack((R2, T2))
    return Pw, Pa, Pb, Pc, Pd


def Afrom2pts(xa, xb, E):
    """
    Given two matched feature points (on images) that are inliers for essential matrix E and E
    returns 4 possible solutions to AX=0 where X represent 4 vectors, homogeneous representation
    of where the point is in the world coordinate
    """
    Ue, Se, Ve = la.svd(E) # gives U, S and V^T
    W = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])

    R1 = np.matmul(Ue, np.matmul(W, Ve))
    R2 = np.matmul(Ue, np.matmul(W.T, Ve))
    T1 = Ue[:, 2]
    T2 = -Ue[:, 2]

    # first camera matrix (used as referencee)
    Pw = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])

    # four possible matrices for the second camera
    Pa = np.column_stack((R1, T1))
    Pb = np.column_stack((R1, T2))
    Pc = np.column_stack((R2, T1))
    Pd = np.column_stack((R2, T2))

    # Each camera (projection matrix P) will define its own A
    Aa = np.array([[Pw[0,0]-Pw[2,0]*xa[0], Pw[0,1]-Pw[2,1]*xa[0], Pw[0,2]-Pw[2,2]*xa[0], Pw[0,3]-Pw[2,3]*xa[0]],
                   [Pw[1,0]-Pw[2,0]*xa[1], Pw[1,1]-Pw[2,1]*xa[1], Pw[1,2]-Pw[2,2]*xa[1], Pw[1,3]-Pw[2,3]*xa[1]],
                   [Pa[0,0]-Pa[2,0]*xb[0], Pa[0,1]-Pa[2,1]*xb[0], Pa[0,2]-Pa[2,2]*xb[0], Pa[0,3]-Pa[2,3]*xb[0]],
                   [Pa[1,0]-Pa[2,0]*xb[1], Pa[1,1]-Pa[2,1]*xb[1], Pa[1,2]-Pa[2,2]*xb[1], Pa[1,3]-Pa[2,3]*xb[1]]])

    Ab = np.array([[Pw[0,0]-Pw[2,0]*xa[0], Pw[0,1]-Pw[2,1]*xa[0], Pw[0,2]-Pw[2,2]*xa[0], Pw[0,3]-Pw[2,3]*xa[0]],
                   [Pw[1,0]-Pw[2,0]*xa[1], Pw[1,1]-Pw[2,1]*xa[1], Pw[1,2]-Pw[2,2]*xa[1], Pw[1,3]-Pw[2,3]*xa[1]],
                   [Pb[0,0]-Pb[2,0]*xb[0], Pb[0,1]-Pb[2,1]*xb[0], Pb[0,2]-Pb[2,2]*xb[0], Pb[0,3]-Pb[2,3]*xb[0]],
                   [Pb[1,0]-Pb[2,0]*xb[1], Pb[1,1]-Pb[2,1]*xb[1], Pb[1,2]-Pb[2,2]*xb[1], Pb[1,3]-Pb[2,3]*xb[1]]])

    Ac = np.array([[Pw[0,0]-Pw[2,0]*xa[0], Pw[0,1]-Pw[2,1]*xa[0], Pw[0,2]-Pw[2,2]*xa[0], Pw[0,3]-Pw[2,3]*xa[0]],
                   [Pw[1,0]-Pw[2,0]*xa[1], Pw[1,1]-Pw[2,1]*xa[1], Pw[1,2]-Pw[2,2]*xa[1], Pw[1,3]-Pw[2,3]*xa[1]],
                   [Pc[0,0]-Pc[2,0]*xb[0], Pc[0,1]-Pc[2,1]*xb[0], Pc[0,2]-Pc[2,2]*xb[0], Pc[0,3]-Pc[2,3]*xb[0]],
                   [Pc[1,0]-Pc[2,0]*xb[1], Pc[1,1]-Pc[2,1]*xb[1], Pc[1,2]-Pc[2,2]*xb[1], Pc[1,3]-Pc[2,3]*xb[1]]])

    Ad = np.array([[Pw[0,0]-Pw[2,0]*xa[0], Pw[0,1]-Pw[2,1]*xa[0], Pw[0,2]-Pw[2,2]*xa[0], Pw[0,3]-Pw[2,3]*xa[0]],
                   [Pw[1,0]-Pw[2,0]*xa[1], Pw[1,1]-Pw[2,1]*xa[1], Pw[1,2]-Pw[2,2]*xa[1], Pw[1,3]-Pw[2,3]*xa[1]],
                   [Pd[0,0]-Pd[2,0]*xb[0], Pd[0,1]-Pd[2,1]*xb[0], Pd[0,2]-Pd[2,2]*xb[0], Pd[0,3]-Pd[2,3]*xb[0]],
                   [Pd[1,0]-Pd[2,0]*xb[1], Pd[1,1]-Pd[2,1]*xb[1], Pd[1,2]-Pd[2,2]*xb[1], Pd[1,3]-Pd[2,3]*xb[1]]])
    return Aa, Ab, Ac, Ad

def generate_X_from_A(Aa, Ab, Ac, Ad):
    """
    Since AX=0 gives 4 equations for 3 unknowns. Use least square to solve for X for each A.
    Gives the location of each point in the world coordinate system.
    """
    # least squares for solving linear system A_{0:2} X_{0:2} = - A_3
    Aa_02 = Aa[:,:3]       # the first 3 columns of 4x4 matrix A
    Aa_3  = Aa[:,3]        # the last column on 4x4 matrix A
    Ab_02 = Ab[:,:3]
    Ab_3  = Ab[:,3]
    Ac_02 = Ac[:,:3]
    Ac_3  = Ac[:,3]
    Ad_02 = Ad[:,:3]
    Ad_3  = Ad[:,3]

    # Nx3 matrices: N rows with 3D point coordinates for N reconstructed points (N=num_inliers)
    inv_ATAa = la.inv(np.matmul(Aa_02.T, Aa_02))
    Xa = np.matmul(inv_ATAa, np.matmul(Aa_02.T, -Aa_3))

    inv_ATAb = la.inv(np.matmul(Ab_02.T, Ab_02))
    Xb = np.matmul(inv_ATAb, np.matmul(Ab_02.T, -Ab_3))

    inv_ATAc = la.inv(np.matmul(Ac_02.T, Ac_02))
    Xc = np.matmul(inv_ATAc, np.matmul(Ac_02.T, -Ac_3))

    inv_ATAd = la.inv(np.matmul(Ad_02.T, Ad_02))
    Xd = np.matmul(inv_ATAd, np.matmul(Ad_02.T, -Ad_3))
    return Xa, Xb, Xc, Xd

def get_proj_pts(P1, P2, pts3D, K):
    """
    Given projection matrix for image 1 and 2, and points in world coordinate system
    returns the location of points on the two images
    """
    # a. convert correct points (Xa, Xb, Xc, or Xd) to homogeneous 4 vectors
    pts3D_homogeneous = np.column_stack((pts3D, np.ones(pts3D.shape[0])))

    # b. project homogeneous 3D points (onto uncalibrated cameras) using correct Projection matrices (KPw and, e.g. KPa)
    ptsL_proj_homogeneous = np.matmul(np.matmul(K, P1), pts3D_homogeneous.T).T
    ptsR_proj_homogeneous = np.matmul(np.matmul(K, P2), pts3D_homogeneous.T).T
    ptsL_scale = np.column_stack((ptsL_proj_homogeneous[:, 2], ptsL_proj_homogeneous[:, 2]))
    ptsR_scale = np.column_stack((ptsR_proj_homogeneous[:, 2], ptsR_proj_homogeneous[:, 2]))

    # c. convert to regular (inhomogeneous) point
    ptsL_proj = ptsL_proj_homogeneous[:, :2] / ptsL_scale
    ptsR_proj = ptsR_proj_homogeneous[:, :2] / ptsR_scale

    return ptsL_proj, ptsR_proj

def estimate_ufl (src_pts, dst_pts, K = 100, gamma = 0.01, T = 1e-20) :
    """
    Estimates a homography from src_pts to dst_pts using the UFL method.
    K is the number of initial points to choose from
    """
    assert len(src_pts) == len(dst_pts)
    num_pts = len(src_pts)

    # Initial set of models
    models = []

    # 1. Randomly choose K lines
    for i in range(K) :
        # Randomly choose 8 points
        random_idx = np.random.choice(num_pts, 8, replace=False)
        src_random = src_pts[random_idx]
        dst_random = dst_pts[random_idx]
        # Fit a homography through them
        EMT = EssentialMatrixTransform()
        if EMT.estimate(src_random, dst_random) :
            models.append(EMT.params)
            # print(np.diag(np.hstack((dst_random, np.ones((8,1)))) @ EMT.params @ np.vstack((src_random.T, np.ones(8)))))

    # Remove duplicates in the initial set of models
    models = np.unique(np.array(models), axis=0)

    # 2. Nearest model for each point
    nearest_models = np.zeros(num_pts)-1
    # Nearest distances for each point
    distances = np.zeros(num_pts)+np.Inf

    # Homogeneous coordinates
    dst_hom = np.hstack((dst_pts, np.ones((num_pts,1))))
    src_hom = np.vstack((src_pts.T, np.ones(num_pts)))

    # Find nearest model for each point
    for i in range(len(models)) :
        # Distances for each point to the current model
        model_dists = np.diag(dst_hom @ models[i] @ src_hom)
        # Update points where the current model has smaller distance
        nearest_models[np.where((model_dists < distances) & (model_dists < T))] = i
        # Update distances
        distances = np.minimum(distances, model_dists)

    #print(distances)
    #print(nearest_models)

    # 3. The list of resulting models
    result_models = []
    # Match each point to a model index. An index of -1 means outlier.
    pts_to_models = np.zeros(num_pts)-1

    # Readjust parameters for each model to better fit the inliers
    for i in range(len(models)) :
        # The inlier index
        inlier_idx = np.where(nearest_models == i)[0]
        # Skip if no inliers
        if len(inlier_idx) < 8 :
            continue
        # Inlier points
        src_inliers = src_pts[inlier_idx]
        dst_inliers = dst_pts[inlier_idx]
        # Make the model more precise by re-estimating using only the inliers
        EMT = EssentialMatrixTransform()
        if EMT.estimate(src_inliers, dst_inliers) :
            pts_to_models[inlier_idx] = len(result_models)
            result_models.append(EMT.params)

    # 4. Iteration: decide whether or not to keep each model
    for i in range(len(result_models)) :
        # Points assigned to this model
        model_src_pts = src_pts[np.where(pts_to_models == i)]
        model_dst_pts = dst_pts[np.where(pts_to_models == i)]
        # Distances for each point to this model
        model_dists = np.diag(dst_hom @ result_models[i] @ src_hom)[np.where(pts_to_models == i)]
        # The cost of keeping this model is the cost of assigning points
        # to this model plus the maintenance gamma
        keep_cost = np.sum(model_dists) + gamma
        # Find the nearest model for each point excluding the current model
        remove_model_dists = np.zeros(num_pts)+np.Inf
        # Nearest model for each point after the current model is removed
        remove_nearest_models = pts_to_models
        for j in range(len(result_models)) :
            # Skip if it's the current model under consideration
            if j == i :
                continue
            dists = np.diag(dst_hom @ models[j] @ src_hom)
            # Distances for each point to the model
            remove_nearest_models[np.where(dists < remove_model_dists)] = j
            remove_model_dists = np.minimum(remove_model_dists, dists)
        # Cost of removing the model
        remove_cost = np.sum(remove_model_dists[np.where(result_models == i)])
        # Remove the model if the cost of keeping is greater
        if remove_cost < keep_cost :
            pts_to_models[np.where(pts_to_models) == i] = remove_nearest_models[np.where(pts_to_models) == i]

    return result_models, pts_to_models

    # num_models = len(models)
    # # Convert the models into a numpy array where the models are vertically stacked
    # Lambda = np.zeros((0,3))
    # for L in models :
    #     Lambda = np.vstack((Lambda, L))

    # print(Lambda)
    # print(Lambda @ np.vstack((src_pts[:2].T, np.ones(2))))

    # # Row indexes models, column indexes points
    # # Ex_src encodes E @ src_pts
    # Ex_src = Lambda @ np.vstack((src_pts.T, np.ones(num_pts)))
    # # Dest points but in homogeneous coordinates
    # dst_hom = np.hstack((dst_pts, np.ones((num_pts,1))))
    # # Duplicate the dst_hom horizontally for multiplication with Ex_src
    # dst_hom_dup = np.tile(dst_hom, (1, num_models))
    # print(Ex_src)
    # print(dst_hom_dup)

    # # Distance from each point to the respective lines
    # # distances =
