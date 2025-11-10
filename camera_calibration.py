import cv2 as cv
import numpy as np

class Checkerboard_Checker():
    def __init__(self, dim = (6,9)):
        self.dimensions = dim
        self.objp = np.zeros((1, dim[0] * dim[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objpoints = []
        self.imgpoints = []

        self.intrinsic_matrix = None
        self.distortion_matrix = None

    def look_for_checkerboard(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.dimensions, 
                            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            new_corners = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), self.criteria)
            new_img = cv.drawChessboardCorners(img, self.dimensions, new_corners, ret)
        else: new_img = img; new_corners = None

        return new_img, new_corners
    
    def calibrate_camera(self, w, h):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, (w, h), None, None)
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.intrinsic_matrix = new_camera_matrix
        self.distortion_matrix = dist

        print("ISSO É A ORIGINAL", mtx)
        print("E ESSA A OUTRA", new_camera_matrix)
        

def main():
    cam = cv.VideoCapture(0)
    checker = Checkerboard_Checker()
    while True:
        ret, img = cam.read()
        if not ret: break

        img, corners = checker.look_for_checkerboard(img)
        cv.imshow("CAMERA", img)
        
        # key checking
        key = chr(cv.waitKey(1) & 0xFF)
        if key == "q": break
        if key == "a": 
            if corners is  None: continue
            checker.objpoints.append(checker.objp)
            checker.imgpoints.append(corners)
            print("Temos mais uma imagem")
        if key == "r":
            checker.calibrate_camera(*img.shape[1:])
        
    cv.destroyAllWindows()

if __name__ == "__main__": main()