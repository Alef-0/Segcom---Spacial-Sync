import numpy as np
import cv2 as cv

class Transformation():
    def __init__(self, intrinsic : np.ndarray, distortion : np.ndarray, homography : np.ndarray):
        self.k = intrinsic
        self.d = distortion
        self.h = homography

        self.fx, self.fy = self.k[0,0], self.k[1,1]
        self.cx, self.cy = self.k[0,2], self.k[1,2]

        self.k1, self.k2, self.p1, self.p2, self.k3 = self.d

        self.opt_matrix, self.roi = cv.getOptimalNewCameraMatrix(self.k, self.d, (1920, 1080), 1, (1920, 1080))

    def undistort_image(self, img : cv.Mat): return cv.undistort(img, self.k, self.d, None, self.opt_matrix)

    def normalize_coordinates(self, points : np.ndarray):
        # It needs to normalize into what it mapped
        points_normalized = np.zeros((points.shape[0], 2))
        points_normalized[:,0] = (points[:,0] - self.opt_matrix[0,2]) / self.opt_matrix[0,0]
        points_normalized[:,1] = (points[:,1] - self.opt_matrix[1,2]) / self.opt_matrix[1,1]
        return points_normalized

    def distort_points(self, points : np.ndarray):
    # Apply distortion model in normalized plane
        r2 = points[:, 0]**2 + points[:, 1]**2
        
        # Radial distortion (assuming standard OpenCV model)
        radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        
        # Tangential distortion
        points_distorted = np.zeros_like(points)
        points_distorted[:, 0] = points[:, 0] * radial + 2 * self.p1 *points[:, 0]*points[:, 1] + self.p2 * (r2 + 2*points[:, 0]**2)
        points_distorted[:, 1] = points[:, 1] * radial + self.p1 *(r2 + 2*points[:, 1]**2) + 2 * self.p2 * points[:, 0] * points[:, 1]

        # Convert back to pixel coordinates
        distorted_pixels = np.zeros_like(points_distorted)
        distorted_pixels[:, 0] = points_distorted[:, 0] * self.fx + self.cx
        distorted_pixels[:, 1] = points_distorted[:, 1] * self.fy + self.cy

        return distorted_pixels

    def radar_to_distorted(self, points : np.ndarray):
        # Apply Homogeneous matrix transformation [Same thing as cv.perspective transform]
        radar_homogenous = np.hstack([points, np.ones((points.shape[0],1))])
        radar_homogenous = (self.h @ radar_homogenous.T).T
        undistorted_pixels = radar_homogenous[:,:2] / radar_homogenous[:,2:3]
        
        # Convert to normalized camera coordinates
        points_normalized = self.normalize_coordinates(undistorted_pixels) # K-1
        distorted_pixels = self.distort_points(points_normalized) # K and D
        
        return undistorted_pixels, distorted_pixels

import time
from aux_files import Files
from aux_graph import Graph

if __name__ == '__main__':
    MATRIX = np.array([
	   [1680.24,    0,          644.38],
       [0,          1674.94,    360.14],
       [0,          0,          1]]
	   , dtype=np.float64)
    DISTORTION = np.array([-0.586, -1.263, 0.034, -0.025, 2.036])

    HOMOGRAPH = np.array([
                            [     42.819,      152.72,      1015.6],
                            [    -185.61,     59.863,      747.48],
                            [    -0.3605,      0.1478,           1]
                            ])


    files = Files()
    graph = Graph()
    mapping = Transformation(MATRIX, DISTORTION, HOMOGRAPH)

    for num in range(files.first, files.last):
        image, points = files.get(num)
        if image is None or not points.any(): continue

        distorted = image.copy()
        undistorted = mapping.undistort_image(image)

        und_points, dist_points = mapping.radar_to_distorted(points)
        for x, y in und_points:
            cv.circle(undistorted, (int(x), int(y)), 12, (0,0,255), -1)
        for x, y in dist_points:
            # print(x,y)
            if 0 < x < 1920 and 0 < y < 1080:
                cv.circle(distorted, (int(x), int(y)), 12, (255,0,0), -1)

        
        cv.imshow("CROPPED", cv.resize(undistorted, (1280, 720)))
        cv.imshow("ORIGINAL", cv.resize(distorted, (1280, 720)))
        key = cv.waitKey(30) & 0xFF
        
        match key:
            case 113: break # Q
            case 112: 
                    while True: # P
                        key = cv.waitKey(30) & 0xFF
                        match key:
                            case 112: break # Q
