from aux_files import Files
from aux_vision import Vision
from aux_graph import Graph
import cv2 as cv

if __name__ == "__main__":
    files = Files()
    vision = Vision()
    graph = Graph()

    # Ficar mostrando
    i = 100
    p_pressed = False
    leave = True
    while leave:
        img, points = files.get(i); i+=1
        delimiters = vision.get_squares(img)
        left, right = vision.determine_left_right(delimiters, img)
        
        # Exhibit it
        new_img = cv.resize(img, (1280, 720))
        graph.show_points(points[:, 0], points[:, 1], len(points) * [(0,200,0)])
        cv.imshow("CAMERA", new_img)
        
        while (True):
            key = chr(cv.waitKey(1) & 0xFF)    
            match key:
                case 'p': p_pressed = not p_pressed
                case 'q': leave = False; break
                case _: 
                    if not p_pressed: break
