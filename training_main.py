from aux_files import Files
from aux_vision import Vision
from aux_graph import Graph
import cv2 as cv
import math
import numpy as np

MATRIX = np.array([
	   [1.31030209e+03, 0.00000000e+00, 8.87851418e+02],
       [0.00000000e+00, 1.29733047e+03, 4.19861323e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	   , dtype=np.float64)
DISTORTION = np.array([[-0.45288733,  0.27704813,  0.01715661,  0.00394433, -0.09480632]])
wid_and_hei = (1920, 1080)
SLOPE = math.tan(math.radians(30))

CORRECTION_Y = 0.4
CORRECTION_X = 0.1

def correct_image(img):
    # Find the new optimal camera matrix for the full view (alpha=1)
    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(MATRIX, DISTORTION, wid_and_hei, 0, wid_and_hei)
    return cv.undistort(img, MATRIX, DISTORTION, None, new_camera_matrix)

def simple_points_transformation(x : np.ndarray, y : np.ndarray):
    # Delimitar Y
    y_normalized = np.minimum(y / 10, 1)  # Normalizar e ao mesmo tempo ter um offset minimo
    y_image_points = 1080 - ((CORRECTION_Y + (1 - CORRECTION_Y) * y_normalized) * 540) # Pegar do de baixo

    # Delimitar X
    x_image_points = [] # Needs to be based around the distance avaiable
    for dist, radius in zip(x,y):
        max_x = radius / SLOPE
        x_normalized = dist / max_x
        print(max_x, x_normalized)
        result = 960 + x_normalized * 960
        x_image_points.append(result)
        # print(x_normalized, result)
    
    return np.column_stack([x_image_points, y_image_points])

def put_radar_in_image(points, img):
    for x, y in points:
        cv.circle(img, (int(x),int(y)), 10, (0,200, 0), -1)

if __name__ == "__main__":
    files = Files()
    vision = Vision()
    graph = Graph()

    # Ficar mostrando
    i = 100 # 1200 -> Direto pro radar
    p_pressed = False
    leave = True

    min_lat = {1: 0, 2 : 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    max_lat = {1: 0, 2 : 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    while leave:
        img, points = files.get(i); i+=1   
        print("------------------")     
        print("THING I: ", i, " OF ", files.total)
        radar_to_image_points = (simple_points_transformation(points[:,0], points[:,1]))

        # Correct Camera
        new_img = img.copy()
        # new_img = correct_image(new_img)
        # Exhibit points in it
        delimiters = vision.get_squares(new_img)
        left, right = vision.determine_left_right(delimiters, new_img) # Image Bounding Box
        graph.show_points(points[:, 0], points[:, 1], len(points) * [(0,200,0)]) # Radar Graph Points
        put_radar_in_image(radar_to_image_points, new_img) # Put radar points in images
        # Resize and show image
        cv.putText(new_img, f"{i}", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv.line(new_img, (0, 540), (1920, 540), (255,255,255), 4)# Adding a vertical line
        cv.line(new_img, (960, 540), (960, 1080), (255,255,255), 4)# Adding a horizontal line
        new_img = cv.resize(new_img, (1280, 720)) 
        cv.imshow("CAMERA", new_img)
        
        while (True):
            key = chr(cv.waitKey(1) & 0xFF)    
            match key:
                case 'p': p_pressed = not p_pressed
                case 'q': leave = False; break
                case _: 
                    if not p_pressed: break
