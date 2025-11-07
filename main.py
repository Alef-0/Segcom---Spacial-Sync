from aux_files import Files
from aux_vision import Vision
from aux_graph import Graph
import cv2 as cv
import math
import numpy as np
import numpy.typing as npt
from typing import Union
from math import hypot as dst
import json

MATRIX = np.array([
	   [1.31030209e+03, 0.00000000e+00, 8.87851418e+02],
       [0.00000000e+00, 1.29733047e+03, 4.19861323e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	   , dtype=np.float64)
DISTORTION = np.array([[-0.45288733,  0.27704813,  0.01715661,  0.00394433, -0.09480632]])

WID_HEI = (1920, 1080)
GOOD_POINTS = [160, 212, 593, 1005, 1371]

def correct_image(img):
    # Find the new optimal camera matrix for the full view (alpha=1)
    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(MATRIX, DISTORTION, WID_HEI, 0, WID_HEI)
    return cv.undistort(img, MATRIX, DISTORTION, None, new_camera_matrix)

def check_distance(x, y, left : npt.NDArray, right : npt.NDArray):
        # Checando Os pontos
        center_r = 2 * ((left[2] + left[0]) / 2) / 1720 - 1
        center_b = 2 * ((right[2] + right[0]) / 2) / 1720 - 1
        height_r = (1080 - left[3]) / 540 
        height_b = (1080 - right[3]) / 540 

        # Proporção para o eixo y valer mais
        dist_r = dst((center_r - x) * 3, (height_r - y) * 5)
        dist_b = dst((center_b - x) * 3, (height_b - y) * 5)

        if dist_r < dist_b: return True
        else: return False

def determine_group(left : npt.NDArray, right : Union[npt.NDArray, None], points : npt.NDArray):
    points[:,0] = (2 * ((points[:,0] + 5) / 10)) - 1 # [-1,1]
    points[:,1] = (points[:,1]) / 10 # [0, 1]

    # Fazer a lógica
    colors = []
    for p in points:
        if right is None: 
            colors.append((0,0,255))
        else:  
            value = check_distance(p[0], p[1], left, right)
            colors.append((0,0,255) if value else (255,0,0))
    return colors

def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:  # Check for a left mouse button click
        print(f'Clicked at coordinates: ({x}, {y})')

class Homograph_Matrix:
    def __init__(self):
        self.next = True
        self.radar_position = []
        self.image_position = []
        self.count = 0

        self.transform_matrix = None
    
    def click_event(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:  # Check for a left mouse button click
            self.count = (self.count + 1) % 2
            self.image_position.append([x,y])
            if self.count == 0: self.next = False
    
    def get_points_center(self, points, colors): # This needs both colors to be valid
        RED = (0,0,255); BLUE = (255,0,0)
        red_points = np.stack([p for p,c in zip(points, colors) if c is RED])
        blue_points = np.stack([p for p,c in zip(points, colors) if c is BLUE])
        center_red : np.ndarray = np.mean(red_points, axis=0)
        center_blue : np.ndarray = np.mean(blue_points, axis=0)
        
        self.radar_position.extend([center_red.tolist(), center_blue.tolist()])
        

    def get_arrays(self):
        files = Files()
        vision = Vision()
        graph = Graph()

        for num in GOOD_POINTS:
            print(self.image_position, self.radar_position)
            image, points = files.get(num)
            new_img = None
            if image is not None: 
                new_img = correct_image(image)
                delimiters = vision.get_squares(new_img)
                left, right = vision.determine_left_right(delimiters, new_img)
                cv.putText(new_img, f"{num}", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                cv.imshow("CAMERA", new_img)
                cv.setMouseCallback('CAMERA', self.click_event)
            colors = determine_group(left, right, points.copy())
            graph.show_points(points[:, 0], points[:, 1], colors)

            # Achar os pontos
            self.get_points_center(points, colors)
            # Os pontos serão pegos em self.click_event

            while self.next: 
                key = cv.waitKey(1) & 0xFF
                match key:
                    case 27: cv.destroyAllWindows(); return  # Apertar ESC para sair
            self.next = True
        
        values_dicio = {"IMAGEM": self.image_position, "RADAR": self.radar_position}
        with open("arrays.json", "w") as f: 
            json.dump(values_dicio, f, indent=4)

    def create_homography(self):
        with open('arrays.json', 'r') as file: 
            data = json.load(file)
            self.image_position = np.array(data["IMAGEM"], dtype=np.float32)
            self.radar_position = np.array(data["RADAR"], dtype=np.float32)

        self.transform_matrix, mask = cv.findHomography(self.radar_position, self.image_position, cv.RANSAC, 5.0) 
        print(self.transform_matrix)
    
    def show_everything():
        pass




if __name__ == "__main__":
    m = Homograph_Matrix()
    # m.get_arrays()
    m.create_homography()

        
