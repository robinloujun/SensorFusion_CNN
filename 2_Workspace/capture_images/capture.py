#!/home/lou/SensorFusion_CNN/2_Workspace/capture_images/
# -*- coding: utf-8 -*-
import logging
import shutil
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import cv2
import pyrealsense as pyrs
import datetime
import preprocessing
from pyrealsense.constants import rs_option
# Keylogger
import pyxhook
import os

# set the save path of the captured images
dirname = '/home/lou/Dataset/'+ datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
pos_dir_origin_color = dirname + '_original_color/pos'
pos_dir_origin_depth = dirname + '_original_depth/pos'
neg_dir_origin_color = dirname + '_original_color/neg'
neg_dir_origin_depth = dirname + '_original_depth/neg'
dirnames = [dirname + '_original_color', dirname + '_original_depth',
            dirname + '_resized_color', dirname + '_resized_depth']

# Label für Bild
pos = 0

# Befehl für Aufnahme
rec = 0
recorded = 0

# Events für Keylogger
def OnKeyPress(event):
    if event.Key is "w": 
        global pos
        pos = 1
    if event.Key is "r":
        global rec
        if rec is 0:
            rec = 1
            print("Start recording.\n")
        else:
            rec = 0
            print("Stop recording.\n")

def OnKeyRealease(event):
    if event.Key is "w":
        global pos
        pos = 0 


hm = pyxhook.HookManager()
hm.KeyDown = OnKeyPress
hm.KeyUp = OnKeyRealease

hm.HookKeyboard()
hm.start()

# load the calibration data
calibration_color = np.load('calibration_data_color.npz')
calibration_depth = np.load('calibration_data_depth.npz')
mtx_color = calibration_color['mtx']
dist_color = calibration_color['dist']
mtx_depth = calibration_depth['mtx']
dist_depth = calibration_depth['dist']

with pyrs.Service() as serv, serv.Device(device_id = 0,) as dev:

# with pyrs.Service() as serv, serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(width = 640, height = 480, fps = 60), pyrs.stream.DepthStream(width = 640, height = 480, fps = 60), pyrs.stream.DACStream(width = 640, height = 480, fps = 60)]) as dev:

    dev.apply_ivcam_preset(5)
        
    #try:  # set custom gain/exposure values to obtain good depth image
        #custom_options = [(rs_option.RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED, 50)]
        ##RS_OPTION_R200_LR_EXPOSURE
        #dev.set_device_options(*zip(*custom_options))
    #except pyrs.RealsenseError:
        #pass
        
    cnt = 0
    last = time.time()
    smoothing = 0.9
    fps_smooth = 60
    f_count = 0

    while True:
        cnt += 1
        if (cnt % 10) == 0:
            now = time.time()
            dt = now - last
            fps = 10/dt
            fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
            last = now

        dev.wait_for_frames()



        # get the color image
        color = dev.color
        color = cv2.flip(color,-1)
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color_undistort = cv2.undistort(color_bgr, mtx_color, dist_color, None)

        # get the depth image (aligned depth to color)
        depth = dev.dac * dev.depth_scale * 1000
        depth = depth.astype(np.uint8)
        depth = cv2.flip(depth,-1)
        depth_undistort = cv2.undistort(depth, mtx_depth, dist_depth, None)

        # manually translate the depth image to fit the color
        M = np.float32([[1,0,-10],[0,1,4]])
        depth_tran = cv2.warpAffine(depth_undistort,M,(640,480))

        # mix the gray & depth image
        gray = cv2.cvtColor(color_undistort, cv2.COLOR_BGR2GRAY)
        mix = cv2.addWeighted(gray,0.5,depth_tran,0.5,0)

        # define the name of images
        now = time.clock()
        c_name = 'rgb' + '_' + str(f_count) + '_' + str(now) + '_' + str(pos) + '.jpg'
        d_name = 'dep' + '_' + str(f_count) + '_' + str(now) + '_' + str(pos) + '.jpg'

        if rec is 1:
            f_count +=1 
            recorded = 1

            for dir in dirnames:
                if not os.path.exists(dir):
                    os.makedirs(dir + '/pos')
                    os.makedirs(dir + '/neg')

            if pos is 1:
                cv2.imwrite(os.path.join(pos_dir_origin_color, c_name),color_undistort)
                cv2.imwrite(os.path.join(pos_dir_origin_depth, d_name),depth_tran)
            else:
                cv2.imwrite(os.path.join(neg_dir_origin_color, c_name),color_undistort)
                cv2.imwrite(os.path.join(neg_dir_origin_depth, d_name),depth_tran)

        # show the images
        cv2.imshow('color',color_undistort)
        cv2.moveWindow('color',0,750)
        cv2.imshow('depth',depth_tran)
        cv2.moveWindow('depth',650,750)
        cv2.imshow('mix',mix)
        cv2.moveWindow('mix',1300,750)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            hm.cancel()
            cv2.destroyAllWindows()
            if recorded is 1:
                print("Starting preprocessing of images.")
                preprocessing.postprocess_images_from_folder(dirname)
            #for folder in foldernames:
                #if os.listdir(dir+folder) == []:
                    #shutil.rmtree(dir+folder)
                print("Done with preprocessing of images.")
            break
