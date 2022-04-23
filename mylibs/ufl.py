import numpy as np
from skimage.transform import EssentialMatrixTransform

class UFL:
    def __init__(self, K=100, gamma=0.01, T=1):
        self.K = K
        self.gamma = gamma 
        self.T = T

    def initialize_models(self, src_pts, dst_pts):
        assert len(src_pts) == len(dst_pts)
        num_pts = len(src_pts)
        models = []
        
        for i in range(self.K) :
        # Randomly choose 8 points
            random_idx = np.random.choice(num_pts, 8, replace=True)
            src_random = src_pts[random_idx]
            dst_random = dst_pts[random_idx]
            # Fit a homography through them
            EMT = EssentialMatrixTransform()
            if EMT.estimate(src_random, dst_random):
                models.append(EMT.params)
        # Remove duplicates in the initial set of models
        models = np.unique(np.array(models), axis=0)
        return models
    
    def reestimate_models(self, models, src_pts, dst_pts, iter_num=None):
        """ Assign points to closest models and reestimate different models (step 2) """
        assert len(src_pts) == len(dst_pts)
        num_pts = len(src_pts)
        dst_hom = np.hstack((dst_pts, np.ones((num_pts,1))))
        src_hom = np.vstack((src_pts.T, np.ones(num_pts)))

        # 2a. Nearest model for each point
        nearest_models = np.full(num_pts, fill_value=-1)
        # Nearest distances for each point
        distances = np.full(num_pts, fill_value=np.Inf)
        
        # Find nearest model for each point
        for i in range(len(models)) :
            # Distances for each point to the current model
            model_dists = np.linalg.norm(dst_hom - (models[i] @ src_hom).T, axis=1)
            # Update points where the current model has smaller distance
            nearest_models[np.where((model_dists < distances) & (model_dists < self.T))] = i
            # Update distances
            distances = np.minimum(distances, model_dists)

        result_models = []
        pts_to_models = np.full(num_pts, fill_value=-1)
        # Map each model to its list of points
        model_pts_masks = [] 
        # Readjust parameters for each model to better fit the inliers
        for i in range(len(models)):
            # The inlier index
            inlier_idx = np.where(nearest_models == i)
            # Skip if no inliers
            if len(inlier_idx[0]) < 8 :
                pts_to_models[inlier_idx] = -1
                if iter_num :
                    print(f'Less than 8 inliers for model {i}, skipping...')
                continue

            # Inlier points
            src_inliers = src_pts[inlier_idx]
            dst_inliers = dst_pts[inlier_idx]
            # Make the model more precise by re-estimating using only the inliers
            EMT = EssentialMatrixTransform()
            if EMT.estimate(src_inliers, dst_inliers):
                pts_to_models[inlier_idx] = len(result_models)
                # print(f'Setting points {inlier_idx} to have model {len(result_models)}')
                result_models.append(EMT.params)
                model_pts_masks.append(inlier_idx)
            else :
                print('re-estiamte failed')
                pts_to_models[inlier_idx] = -1
        
        return result_models, pts_to_models, model_pts_masks

    def remove_models(self, src_pts, dst_pts, models, pts_to_models, model_pts_masks):
        """ removing lines L that are not worth keeping (step 3) """
        assert len(src_pts) == len(dst_pts)
        num_pts = len(src_pts)
        dst_hom = np.hstack((dst_pts, np.ones((num_pts,1))))
        src_hom = np.vstack((src_pts.T, np.ones(num_pts)))

        # 3. Decide whether or not to keep each model
        assert len(models) == int(max(pts_to_models))+1
        for i in range(len(models)):
            print('----------')
            # Points assigned to this model
            model_pts_mask = model_pts_masks[i]
            assert len(model_pts_mask[0]) >= 8, f"Less than 8 inliers for model {i}"
            print(f'inlier points index for model {i} is {model_pts_mask}')
            # The cost of keeping this model is the cost of assigning points
            # to this model plus the maintenance gamma
            model_dists = np.linalg.norm(dst_hom - (models[i] @ src_hom).T, axis=1)[model_pts_mask]
            keep_cost = np.sum(model_dists) + self.gamma
            print(f'cost to keep model {i} is {keep_cost}')

            # Find the nearest model for each point excluding the current model
            other_model_dists = np.full(num_pts, fill_value=np.Inf)
            # Nearest model for each point after the current model is removed
            other_nearest_models = pts_to_models
            for j in range(len(models)) :
                # Skip if it's the current model under consideration
                if j == i :
                    continue
                model_dists_j = np.linalg.norm(dst_hom - (models[j] @ src_hom).T, axis=1)
                # Distances for each point to the model
                other_nearest_models[np.intersect1d(np.where(model_dists_j < other_model_dists)[0], model_pts_mask[0])] = j
                other_model_dists = np.minimum(other_model_dists, model_dists_j)
            # Cost of removing the model
            remove_cost = np.sum(other_model_dists[model_pts_mask])
            #print(f'distances to nearest model after removing model {i} = ')
            #print(remove_model_dists)
            print(f'cost to remove model {i} is {remove_cost}')
            # Remove the model if the cost of keeping is greater
            if remove_cost < keep_cost :
                pts_to_models[model_pts_mask] = other_nearest_models[model_pts_mask]
                # print(f'remove model {i}')
                # print(f'pts_to_models = {pts_to_models}')

        # Calculate the energy of the current configuration
        model_dists = np.linalg.norm(dst_hom - (models[i] @ src_hom).T, axis=1)[np.where(pts_to_models > -1)]
        energy = np.sum(model_dists)
        return models, pts_to_models, energy

    def estimate_ufl(self, src_pts, dst_pts):
        """
        Estimates a homography from src_pts to dst_pts using the UFL method.
        K is the number of initial points to choose from
        """
        iter_num = 0
        energy = np.Inf
        initial_model = self.initialize_models(src_pts, dst_pts)
        result_models, pts_to_models, model_pts_masks = self.reestimate_models(initial_model, src_pts, dst_pts, iter_num=iter_num)
        while iter_num >= 0 :
            print('====================================================================================================')
            iter_num = iter_num+1
            result_models_update, pts_to_models_update, energy_update = self.remove_models(
                src_pts, dst_pts, result_models, pts_to_models, model_pts_masks)
            result_models_update, pts_to_models_update, model_pts_masks = self.reestimate_models(result_models_update, src_pts, dst_pts, iter_num=iter_num)
            print(f'energy = {energy_update}, num classes = {len(np.unique(pts_to_models_update))}')

            if energy_update < energy :
                energy = energy_update
                result_models, pts_to_models = result_models_update, pts_to_models_update
            else:
                return result_models, pts_to_models