import json
import numpy as np
import cv2 as cv
from pypcd4 import PointCloud

Y_LIMIT = 10
X_LIMIT = 6

class Files:
    def __init__(self, path = "valores/group_B.json"):
        self.dicio : dict = json.load(open(path, "r"))
        self.keys = sorted([int(x) for x in self.dicio.keys()])
        self.first, self.last = self.keys[0], self.keys[-1]
        self.total = self.last - self.first + 1
        self.divider = round(self.total * 0.85)
    
    def filter_pcd(self, points : np.ndarray): # Hard Coded for the usual values
        return points[
            (points[:, 1] < Y_LIMIT) # Menor que 12 mestros
            & (abs(points[:,2]) < X_LIMIT) # Sem estar nas bordas
            & (~np.isin(points[:, 7], [3,4]))  # Sem candidates
            
            & (points[:, 6] > -20) # RCS
            & (points[:, 8] <= 4) & (points[:,8] != 0) # PDH
            & (np.isin(points[:, 9],[3,4])) # Ambiguidade
            & (~np.isin(points[:, 10],[1,2,3, 5,6,7,13,14])) # Cluster state
        ]

    def get(self, num):
        if num < self.first or num > self.last: return None, None

        # num = (num + self.first) % self.total
        files = self.dicio[str(num)]
        pcd = PointCloud.from_path(files['radar']['FILE'])
        points = self.filter_pcd(pcd.numpy())
        points = points[:, [2, 1]].copy()

        image = cv.imread(file) if (file := files['camera']['FILE']) != "NONE" else None

        # Return the image and a copy of the points
        return image, self.normalize(points)
    
    def normalize(self, points):
        points[:,0] = (2 * ((points[:,0] + X_LIMIT) / (2 * X_LIMIT))) - 1 # [-1,1]
        points[:,1] = (points[:,1]) / Y_LIMIT # [0, 1]
        return points

    
# a = Files()
# print(a.get(150, True)[1])
