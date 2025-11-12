import cv2 as cv
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from datetime import datetime
import os


class Checkerboard_Checker(Node):
    def __init__(self, dim = (6,9)):
        super().__init__("CameraCalibration")
        self.sub = self.create_subscription(CompressedImage, "Segcom/Camera/M/compressed", self.receive_image, 10)
        self.dimensions = dim
        self.objp = np.zeros((1, dim[0] * dim[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.folder = datetime.now()
        self.i = 0; os.makedirs(f"{self.folder}", exist_ok=True)

        self.objpoints = []
        self.imgpoints = []

        self.intrinsic_matrix : np.ndarray  = None
        self.distortion_matrix : np.ndarray = None
        self.transformation_matrix : np.ndarray = None

        self.img_white = np.full((1080, 1920, 3), 255, dtype=np.uint8)

    def draw_debug_corners(self, corners : np.ndarray):
        current = self.img_white.copy()
        
        # Drawing the current
        if corners is not None: 
            list_corners = corners.reshape(-1, 2).astype(np.int32)
            for x, y in list_corners: 
                cv.circle(current, (x,y), 8, (255,0,0), -1)
            # cv.fillPoly(current, [list_corners], (100,0,0, 255), 4)

        return current
                


    def receive_image(self, msg : Image):
        # COMPRESSED IMAGE
        img = cv.imdecode(np.frombuffer(msg.data, np.uint8), cv.IMREAD_COLOR)
        img = cv.resize(img, (1920, 1080), interpolation=cv.INTER_CUBIC)
        img, corners = checker.look_for_checkerboard(img)

        seeing = self.draw_debug_corners(corners)

        # key checking
        cv.imshow("CAMERA", cv.resize(img, (640, 480)))
        cv.imshow("SEEING", cv.resize(seeing, (1280, 720)))

        key = chr(cv.waitKey(1) & 0xFF)
        if key == "p": 
            if corners is  not None:
                list_corners = corners.reshape(-1, 2).astype(np.int32)
                for x, y in list_corners: 
                    cv.circle(self.img_white, (x,y), 8, (0,0,255), -1)
                checker.objpoints.append(checker.objp)
                checker.imgpoints.append(corners)
                print(f"Temos mais uma imagem ({self.i})"); self.i+=1
                cv.imwrite(f"{self.folder}/image_{self.i:04}.jpeg", img)
        if key == "r" or key == "q":
            checker.calibrate_camera(1920, 1080)
            self.print_json_like()
            cv.imwrite(f"{self.folder}/COVERAGE.jpeg", seeing)
        if key == "q": rclpy.shutdown()

        # if key == 'p':
        #     print("TO DO")

    def print_json_like(self):
        text = ""
        print("--------------")
        new = "{\n"; text += new; print(new, end="")
        new = f"\t\"Intrinsic\": {self.intrinsic_matrix.tolist()},\n"; text += new; print(new, end="")
        new = f"\t\"Distortion\": {self.distortion_matrix.tolist()},\n"; text += new; print(new, end="")
        # new = f"\t\"Transformation\": {self.transformation_matrix.tolist()},\n"; text += new; print(new, end="") # Always better from scratch
        new = f"\t\"ret\": {self.ret}\n"; text += new; print(new, end="")
        new = "}"; text += new; print(new, end="")
        print("--------------")
        with open("most_current.json", "w") as file: file.write(text)

    def look_for_checkerboard(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.dimensions, 
                            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            new_corners : np.ndarray = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), self.criteria)
            new_img = cv.drawChessboardCorners(img, self.dimensions, new_corners, ret)
        else: new_img = img; new_corners = None

        return new_img, new_corners
    
    def calibrate_camera(self, w, h):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, (w, h), None, None)
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.intrinsic_matrix = mtx
        self.distortion_matrix = dist        
        self.transformation_matrix = new_camera_matrix
        self.ret = ret

if __name__ == "__main__": 
    
    rclpy.init()
    checker = Checkerboard_Checker()
    
    try: rclpy.spin(checker)
    except: pass
    
    if rclpy.ok(): rclpy.shutdown()
    cv.destroyAllWindows()
    # main()