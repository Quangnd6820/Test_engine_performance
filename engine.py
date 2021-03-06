from face_detection.face_detector import face_detector
from utils import *
import numpy as np
import argparse
import time
import cv2
import sys


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_network', default='mobile0.25', type=str, # mobile0.25 or resnet50
                        help='model which is used for detect faces')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    detector = face_detector(args.face_network, cpu=False)
    if args.face_network == "mobile0.25":
        skip_frame = 1
    else:
        skip_frame = 1

    biometrics, names = load_biometric_data()
    # cap_1 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.81.201:554/profile2/media.smp")
    # cap_2 = cv2.VideoCapture("rtsp://admin:admin@123@192.168.81.202:554/profile2/media.smp")
    cap_1 = cv2.VideoCapture('./video/camera1.avi')
    cap_2 = cv2.VideoCapture('./video/camera2.avi')
    # cap = cv2.VideoCapture("./video_data/demo_zoom_x3.avi")

    if not cap_1.isOpened():
        sys.exit('Failed to open camera 1!')
    if not cap_2.isOpened():
        sys.exit('Failed to open camera 2!')
    
    frame_index = 0
    FPS_Max = 0
    FPS_Min = 100

    while True:
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()

        if (not ret_1) and (not ret_2):
            break

        if os.path.exists('box_faces'):
            shutil.rmtree(r'box_faces')

        os.mkdir('box_faces')

        # Check frame indx to process 
        if frame_index % skip_frame == 0:
            start_time = time.time()
            if ret_1:
                frame_1 = cam_predict(frame_1, "camera1", detector, biometrics, names, 0.98, 0.45)
            if ret_2:
                frame_2 = cam_predict(frame_2, "camera2", detector, biometrics, names, 0.98, 0.45)

            #Calculate Max FPS
            print("time:", time.time())
            try:
                if FPS_Max < (1 / (time.time() - start_time)):
                    FPS_Max = (1 / (time.time() - start_time))
                if FPS_Min > (1 / (time.time() - start_time)):
                    FPS_Min = (1 / (time.time() - start_time))
            except:
                pass

            # Show all camera
            if ret_1 and ret_2:
                Verti = np.concatenate((cv2.resize(frame_1, (1024, 768)), cv2.resize(frame_2, (1024, 768))), axis=1)
                # cv2.imshow('Camera', Verti)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break

        frame_index += 1

    print("Maximum FPS Predict: ", FPS_Max)
    print("Minimum FPS Predict: ", FPS_Min)
    cap_1.release()
    cap_2.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()