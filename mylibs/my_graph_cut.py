import numpy as np
from numpy import linalg as la
import maxflow
from sklearn import mixture


class MaskBuilder:
    def __init__(self, num_rows, num_cols, fill_value):
        self.mask = np.full((num_rows, num_cols), fill_value, dtype='uint8')
        self.meshgrid = np.ogrid[:num_rows, :num_cols]

    def add_point(self, x, y, fill_value):
        self.mask[y, x] = fill_value

    def add_disk(self, x, y, radius, fill_value):
        mask = self.mask
        ys, xs = self.meshgrid
        disk_mask = np.less_equal(np.square(xs - x) + np.square(ys - y), radius * radius)
        mask[disk_mask] = fill_value

    def get_value_at(self, x, y):
        return self.mask[y,x]

    def get_mask(self):
        return self.mask

class MyGraphCuts:
    bgr_value = 0
    obj_value = 1
    none_value = -1
    
    def __init__(self, img, sigma=5, lam=50, epsilon=1e-6):
        self.num_rows = img.shape[0]
        self.num_cols = img.shape[1]
        self.img = img
        self.sigma = sigma
        self.lam = lam
        self.epsilon = epsilon

    def compute_labels(self, seed_mask):
        num_rows = self.num_rows
        num_cols = self.num_cols
        img = self.img
        sigma = self.sigma
        lam = self.lam
        epsilon = self.epsilon
        max_weight = 4 * lam # weight of terminal nodes

        # Create the graph.
        g = maxflow.GraphFloat()
        nodeids = g.add_grid_nodes((num_rows, num_cols))
        
        # Add edge weights
        horizontal_structure = np.array([[0, 0, 0],
                                         [0, 0, 1],
                                         [0, 0, 0]])
        shifted_img = np.roll(img.astype(float), shift=1, axis=1)
        shifted_img[:, 0] = 0
        horizontal_weights = lam * np.exp(-la.norm(img-shifted_img, axis=2)**2 / sigma**2)
        g.add_grid_edges(nodeids, weights=horizontal_weights, structure=horizontal_structure, symmetric=True)
        verticle_structure = np.array([[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 1, 0]])
        shifted_img = np.roll(img.astype(float), shift=1, axis=0)
        shifted_img[0] = 0
        verticle_weights = lam * np.exp(-la.norm(img-shifted_img, axis=2)**2 / sigma**2)
        g.add_grid_edges(nodeids, weights=verticle_weights, structure=verticle_structure, symmetric=True)
        
        # Add terminal nodes (similarity with expect obj or bgr intensity) and their weights
        obj_pts = img[np.where(seed_mask == self.obj_value)]
        bgr_pts = img[np.where(seed_mask == self.bgr_value)]
        obj_dist = mixture.GaussianMixture(n_components=6).fit(obj_pts) if obj_pts.size else None
        bgr_dist = mixture.GaussianMixture(n_components=6).fit(bgr_pts) if bgr_pts.size else None
        if obj_dist is not None:
            obj_likelihood = np.exp(obj_dist.score_samples(img.reshape((-1, 3)))).reshape((num_rows, num_cols))
            obj_seeds = -np.log(obj_likelihood + epsilon)
        else:
            obj_seeds = np.zeros((num_rows, num_cols))
        
        if bgr_dist is not None:
            bgr_likelihood = np.exp(bgr_dist.score_samples(img.reshape((-1, 3)))).reshape((num_rows, num_cols))
            bgr_seeds = -np.log(bgr_likelihood + epsilon)
        else:
            bgr_seeds = np.zeros((num_rows, num_cols))

        # Add terminal nodes (user input) and their weights
        obj_seeds[np.where(seed_mask == self.obj_value)] += max_weight
        bgr_seeds[np.where(seed_mask == self.bgr_value)] += max_weight
        g.add_grid_tedges(nodeids, obj_seeds, bgr_seeds)

        # Find the maximum flow.
        g.maxflow()
        
        # Get labels
        sgm = g.get_grid_segments(nodeids)
        label_mask = np.full((num_rows, num_cols), self.none_value, dtype='uint8')
        label_mask[sgm] = self.obj_value
        label_mask[~sgm] = self.bgr_value

        return label_mask