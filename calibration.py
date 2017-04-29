import cv2
import numpy as np
import glob
import pickle
import os.path

def getCalibration():
    mtx = None
    dist = None
    setting_path = "./camera_cal/wide_dist_pickle.p"
    if os.path.exists(setting_path): 
        # Read in the saved calibration settings
        dist_pickle = pickle.load(open(setting_path, "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        print("Calibration settings loaded")
    else: 
        path = "./camera_cal/*.jpg"
        images = glob.glob(path)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        img_size = None


        for i in images:
            img = cv2.imread(i)
            if not img_size:
                img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                write_name = './output_images/corners_found_' + os.path.split(i)[-1]
                cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
        cv2.destroyAllWindows()

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(setting_path, "wb"))
        print("Calibration settings calculated and saved")
    return mtx, dist
