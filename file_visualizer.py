import numpy as np
import cv2 as cv
from aux_files import Files
from aux_graph import Graph
from aux_vision import Vision
from math import radians
from gui_matrix import windows_control
import threading
import multiprocessing

def create_60_slope_points(): 
    x = np.arange(0, 6, 0.01); 
    y = x * radians(30);
    left = np.vstack([-x, y]) ; right = np.vstack([x,y])
    return np.hstack([left, right]).T
    
def radar_to_image(homography_matrix, points, image, color):
    radar_homogenous = np.hstack([points, np.ones((points.shape[0],1))])
    radar_homogenous = (homography_matrix @ radar_homogenous.T).T
    final_points = radar_homogenous[:,:2] / radar_homogenous[:,2:3]

    for x, y in final_points: cv.circle(image, (int(x), int(y)), 3, color, 5, -1)

def try_to_undistort_image(image, intrinsic, distortion):
    opt_matrix, _ = cv.getOptimalNewCameraMatrix(intrinsic, distortion, (1920, 1080), 1, (1920, 1080))
    return cv.undistort(image, intrinsic, distortion, None, opt_matrix)

def visualize_everything(menu : windows_control, stop : threading.Event):
    files = Files()
    vision = Vision()
    graph = Graph()
    slope = create_60_slope_points()

    # Código de visualização
    for num in range(files.first + 100, files.last - 1):
        image, points = files.get(num)
        if image is None or not points.any(): continue

        image = try_to_undistort_image(image, menu.intrinsic, menu.distortion)
        radar_to_image(menu.extrinsic, slope, image, (255,255,255)) # Drawing o the image
        radar_to_image(menu.extrinsic, points, image, (0,0,255))

        cv.imshow("CROPPED", cv.resize(image, (1280, 720)))
        key = cv.waitKey(60) & 0xFF
        if key in (27, ord("q")): break
        elif stop.is_set(): break

    cv.destroyAllWindows()

if __name__ == "__main__":
    menu = windows_control()
    stop = threading.Event()
    th = threading.Thread(target=visualize_everything, args=(menu,stop), daemon=True); th.start()
    menu.get_coefficients_from_json()
    menu.run_window()
    stop.set(); th.join()