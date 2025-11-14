import numpy as np
import cv2 as cv
import json
import math

import time
from aux_files import Files
from aux_graph import Graph
from aux_vision import Vision

# Correção a ser aplicada para os valores
CORRECTION_Y = 0.4
CORRECTION_X = 0.1
SLOPE = math.tan(math.radians(30))

class Transformation():
    def __init__(self, file, intrinsic : np.ndarray = None, distortion : np.ndarray = None, homography : np.ndarray = None):
        if file is not None:
            everything : dict = json.load(open(file, 'r'))
            self.k = np.array(everything["Intrinsic"], dtype = np.float64) if "Intrinsic" in everything else None
            self.d = np.array(everything["Distortion"][0], dtype = np.float64) if "Distortion" in everything else None # For some reason it's a list
            self.opt_matrix, _ = cv.getOptimalNewCameraMatrix(self.k, self.d, (1920, 1080), 1, (1920, 1080))

            self.fx, self.fy = self.k[0,0], self.k[1,1]
            self.cx, self.cy = self.k[0,2], self.k[1,2]
            self.k1, self.k2, self.p1, self.p2, self.k3 = self.d

            # print(self.k, self.d, self.opt_matrix)

        else:
            self.k = intrinsic
            self.d = distortion
            self.h = homography
            self.opt_matrix, _ = cv.getOptimalNewCameraMatrix(self.k, self.d, (1920, 1080), 1, (1920, 1080))

            self.fx, self.fy = self.k[0,0], self.k[1,1]
            self.cx, self.cy = self.k[0,2], self.k[1,2]
            self.k1, self.k2, self.p1, self.p2, self.k3 = self.d

            # print(self.opt_matrix)

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
        points_distorted[:, 0] = points[:, 0]  * radial + 2 * self.p1 *points[:, 0]*points[:, 1] + self.p2 * (r2 + 2*points[:, 0]**2)
        points_distorted[:, 1] = points[:, 1]  * radial + self.p1 *(r2 + 2*points[:, 1]**2) + 2 * self.p2 * points[:, 0] * points[:, 1]

        # Convert back to pixel coordinates
        complex_points = np.zeros_like(points_distorted)
        complex_points[:, 0] = points_distorted[:, 0] * self.fx + self.cx
        complex_points[:, 1] = points_distorted[:, 1] * self.fy + self.cy
        
        simple_points = np.zeros_like(points)
        simple_points[:,0] = points[:,0] * self.fx + self.cx
        simple_points[:,1] = points[:,1] * self.fy + self.cy

        return complex_points, simple_points

    def radar_to_distorted(self, points : np.ndarray):
        # Apply Homogeneous matrix transformation [Same thing as cv.perspective transform]
        radar_homogenous = np.hstack([points, np.ones((points.shape[0],1))])
        radar_homogenous = (self.h @ radar_homogenous.T).T
        undistorted_pixels = radar_homogenous[:,:2] / radar_homogenous[:,2:3]
        
        # Convert to normalized camera coordinates
        points_normalized = self.normalize_coordinates(undistorted_pixels) # K-1
        distorted_complex, distorted_simple = self.distort_points(points_normalized) # K and D
        
        return undistorted_pixels, distorted_complex, distorted_simple
    
    def visualize_result(self, files : Files):
        for num in range(files.first, files.last):
            image, points = files.get(num)
            if image is None or not points.any(): continue

            # Undistort image
            distorted = image.copy()
            undistorted = self.undistort_image(image)
                
            cv.imshow("CROPPED", cv.resize(undistorted, (1280, 720)))
            cv.imshow("ORIGINAL", cv.resize(distorted, (1280, 720)))
            key = cv.waitKey(16) & 0xFF


def part1_look_for_good_takes(mapping : Transformation, files : Files, vision : Vision, graph : Graph):
    all_goods = [922, 1005, 1100, 1440, 1588, 1650, 1695, 1840, 1935, 2052] # These were the good ones
    num = files.first + 100
    paused = False
    end = True

    while end:
        if num < files.first or num > files.last: break
        image, points = files.get(num)
        if image is None or not points.any(): 
            num += 1; continue

        cv.putText(image, f"{num}", (0,75), cv.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3)
        # Exibir a imagem original
        distorted = image.copy()
        cv.imshow("ORIGINAL", cv.resize(distorted, (1280, 720)))
        # Exibir a distorcida
        # undistorted = mapping.undistort_image(image)
        # cv.imshow("CROPPED", cv.resize(undistorted, (1280, 720)))
        # Exibir os pontos
        graph.show_points(points[:, 0], points[:, 1], len(points) * [(0,150,0)])

        first = True
        while paused or first:
            key = (cv.waitKey(0) & 0xFF) if paused else (cv.waitKey(30) & 0xFF) 
            first = False
            match key:
                case 113: end = False; break # Q
                case 112: paused = not paused; # P (Pausar)
                case 81:  num -= 1; break # Esquerda
                case 83:  num += 1; break # direita
                case 255: num += 1  
                case 97: all_goods.append(num); print(all_goods)
            # if key != 255: print(key)
     
    print(all_goods)
    return all_goods
        


if __name__ == '__main__':
    files = Files()
    vision = Vision()
    graph = Graph()
    mapping = Transformation("second_attempt.json")

    # Visualizar o video normalmente
    all_goods = [922, 1005, 1100, 1440, 1588, 1650, 1695, 1840, 1935, 2052] # part1_look_for_good_takes(mapping, files, vision, graph)
    

    # MATRIX = np.array([[6221.92422097575, 0.0, 907.0708040584582], [0.0, 7620.661949881767, 120.96692648284439], [0.0, 0.0, 1.0]], dtype=np.float64)
    # DISTORTION = np.array([-8.201663966549852, 93.00752542264131, 0.22430706498995545, 0.03268512101141362, -607.527225839154], dtype=np.float64)
    # HOMOGRAPH = np.array([
    #                         [     42.819,      152.72,      1015.6],
    #                         [    -185.61,     59.863,      747.48],
    #                         [    -0.3605,      0.1478,           1]
    #                         ])

    # files = Files()
    # graph = Graph()
    # # mapping = Transformation(None, MATRIX, DISTORTION, HOMOGRAPH)
    # mapping = Transformation("second_attempt.json")
    # # mapping.generate_homography(files)

    # for num in range(files.first, files.last):
    #     image, points = files.get(num)
    #     if image is None or not points.any(): continue
    #     graph.show_points(points[:, 0], points[:, 1], len(points) * [(0,200,0)])

    #     distorted = image.copy()
    #     undistorted = mapping.undistort_image(image)
    #     points_undistorted = mapping.radar_to_undistorted(points[:,0], points[:,1])
    #     print(points_undistorted)

    #     # und_points, dist_complex, dist_simple = mapping.radar_to_distorted(points)
    #     # for x, y in dist_complex:
    #     #     if 0 < x < 1920 and 0 < y < 1080:
    #     #         cv.circle(distorted, (int(x), int(y)), 12, (255,0,0), -1)
    #     # for x, y in dist_simple:
    #     #     if 0 < x < 1920 and 0 < y < 1080:
    #     #         cv.circle(distorted, (int(x), int(y)), 12, (0,255,0), -1)
    #     # for x, y in points_undistorted:
    #     #     cv.circle(undistorted, (int(x), int(y)), 12, (0,0,255), -1)
                
    #     cv.imshow("CROPPED", cv.resize(undistorted, (1280, 720)))
    #     cv.imshow("ORIGINAL", cv.resize(distorted, (1280, 720)))
    #     key = cv.waitKey(40) & 0xFF
        
    #     match key:
    #         case 113: break # Q
    #         case 112: 
    #                 while True: # P
    #                     key = cv.waitKey(16) & 0xFF
    #                     match key:
    #                         case 112: break # Q
    #         case _: print(key)
