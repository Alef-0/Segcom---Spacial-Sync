from ultralytics import YOLO
import numpy as np
from ultralytics.engine.results import Results
import cv2 as cv

class Vision:
    def __init__(self):
        self.model = YOLO("./yolo11x.pt")
    
    def get_squares(self, img):
        # Retorna uma lista, porque pode receber uma lista de imagens
        result : Results = self.model.predict(img, verbose = False)[0] 
        result.boxes = result.boxes[(result.boxes.cls == 0)]    # Pegar só os humanos
        delimiters = np.astype(result.boxes.xyxy.cpu().numpy(), np.int32)
        return delimiters
    
    def determine_left_right(self, delimiters : np.ndarray, img):
        first, last = None, None
        if len(delimiters) == 1:
            square : np.ndarray = delimiters[0]
            cv.rectangle(img, square[0:2], square[2:], (0,0,255), 3)
            first = square
        if len(delimiters) > 1:
            square1, square2 = delimiters[0], delimiters[1]
            if square1[0] < square2[0]: first : np.ndarray = square1; last  = square2
            else:                       first : np.ndarray = square2; last  = square1
            cv.rectangle(img, first[0:2], first[2:], (0,0,255), 3)
            cv.rectangle(img, last[0:2], last[2:], (255,0,0), 3)
        
        return first, last

    
