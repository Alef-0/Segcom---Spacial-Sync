from aux_files import Files
from aux_vision import Vision
from aux_graph import Graph
import cv2 as cv
import math
import numpy as np
import numpy.typing as npt
from typing import Union
from math import hypot as dst

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

def determine_position(x, y, first : npt.NDArray, last : npt.NDArray):
        # Checando Os pontos
        center_r = 2 * ((first[2] + first[0]) / 2) / 1720 - 1
        center_b = 2 * ((last[2] + last[0]) / 2) / 1720 - 1
        height_r = (1080 - first[3]) / 540 
        height_b = (1080 - last[3]) / 540 

        # Proporção para o eixo y valer mais
        dist_r = dst((center_r - x) * 3, (height_r - y) * 5)
        dist_b = dst((center_b - x) * 3, (height_b - y) * 5)

        # if abs(x) > 1: 
        #     print(f"[{center_r :.3}, {height_r :.3}] and [{center_b:.3}, {height_b:.3}], para distancia {dist_r:.3} vs {dist_b:.3}, com [{x:.3}, {y:.3}]")
        # checando as distancias

        if dist_r < dist_b: return True
        else: return False


def determine_side(first : npt.NDArray, last : Union[npt.NDArray, None], points : npt.NDArray):
    points[:,0] = (2 * ((points[:,0] + 5) / 10)) - 1 # [-1,1]
    points[:,1] = (points[:,1]) / 10 # [0, 1]

    # Fazer a lógica
    colors = []
    for p in points:
        if last is None: 
            colors.append((0,0,255))
        else:  
            value = determine_position(p[0], p[1], first, last)
            colors.append((0,0,255) if value else (255,0,0))
    return colors

if __name__ == "__main__":
    files = Files()
    vision = Vision()
    graph = Graph()

    # Ficar mostrando
    i = 0 # 1200 -> Direto pro radar



    while True:
        image, points = files.get(GOOD_POINTS[i])
        if image is not None: 
            new_img = image.copy()
            new_img = correct_image(new_img)
            delimiters = vision.get_squares(new_img)
            left, right = vision.determine_left_right(delimiters, new_img)

            # Checar se é válido
            if not points.any(): i+=1; continue
            if (right is not None and
                not (left[2] < right[0] or  right[2] < left[0] or # não interseciona no X
                    left[3] < right[1] or right[3] < left[1])): # Não interseciona no y
                    i+=1; continue

            
            new_image = cv.resize(new_img, (1280, 720))
            cv.putText(new_image, f"{GOOD_POINTS[i]}", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            cv.imshow("CAMERA", new_image)
        
        colors = determine_side(left, right, points.copy())
        graph.show_points(points[:, 0], points[:, 1], colors) # Radar Graph Points
        key = cv.waitKey(0) & 0xFF
        match key:
            case 81: i -= 1 # Esquerda
            case 83: i += 1 # Direita
            case 27: break  # Apertar ESC para sair
            case _: print(key)
        
