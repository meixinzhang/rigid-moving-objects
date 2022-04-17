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

    # Homogeneous coordinates
    dst_hom = np.hstack((dst_pts, np.ones((num_pts,1))))
    src_hom = np.vstack((src_pts.T, np.ones(num_pts)))

    # Iteration of steps 2 and 3
    def iterate (models, pts_to_models = np.zeros(num_pts)-1) :

        # 2a. Nearest model for each point
        nearest_models = np.zeros(num_pts)-1
        # Nearest distances for each point
        distances = np.zeros(num_pts)+np.Inf

        # Find nearest model for each point
        for i in range(len(models)) :
            # Distances for each point to the current model
            model_dists = np.diag(dst_hom @ models[i] @ src_hom)
            # Update points where the current model has smaller distance
            nearest_models[np.where((model_dists < distances) & (model_dists < T))] = i
            # Update distances
            distances = np.minimum(distances, model_dists)

        # 2b. The list of resulting models
        result_models = []

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

        # 3. Decide whether or not to keep each model
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
                # If after removing the model, the cost of a point to the next closest model
                # is larger than T then set that point to be an outlier
                pts_to_models[np.where((pts_to_models == i) & (remove_model_dists < T))] = remove_nearest_models[np.where((pts_to_models == i) & (remove_model_dists < T) )]
                pts_to_models[np.where((pts_to_models == i) & (remove_model_dists >= T))] = -1

        # Calculate the energy of the current configuration
        energy = 0
        for i in range(len(result_models)) :
            # Points assigned to this model
            model_src_pts = src_pts[np.where(pts_to_models == i)]
            model_dst_pts = dst_pts[np.where(pts_to_models == i)]
            # Distances for each point to this model
            model_dists = np.diag(dst_hom @ result_models[i] @ src_hom)[np.where(pts_to_models == i)]
            # The cost of keeping this model is the cost of assigning points
            # to this model plus the maintenance gamma
            energy = energy + np.sum(model_dists) + gamma

        return result_models, pts_to_models, energy

    result_models, pts_to_models, energy = iterate(models)
    print(f'energy = {energy}')
    iter_num = 0
    while iter_num < 4 :
        iter_num = iter_num+1
        result_models, pts_to_models, energy_update = iterate(result_models)
        print(f'energy = {energy_update}, num classes = {len(np.unique(pts_to_models))}')
        if energy_update == energy :
            return result_models, pts_to_models
        else :
            energy = energy_update

    return result_models, pts_to_models
