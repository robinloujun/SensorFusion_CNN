import numpy as np
import cv2

def load_calibrationdata():
    '''load the calibration data'''
    mtx_depth = np.array([[476.716836, 0.000000, 316.163927], [0.000000, 474.934106, 239.044866], [0.000000, 0.000000, 1.000000]])
    mtx_color = np.array([[620.549550, 0.000000, 316.723218], [0.000000, 622.598518, 247.240599], [0.000000, 0.000000, 1.000000]])
    dist_depth = np.array([-0.110005, -0.031157, -0.003687, 0.003962])
    dist_color = np.array([0.156850, -0.340158, 0.001565, -0.001992])
    R = np.array([[701.547270, 0.000000, 158.045394], [0.000000, 701.547270, 239.300177], [0.000000, 0.000000, 1.000000]])
    T = np.array([[80.889927], [0.000000], [0.000000]])
    return [mtx_depth, mtx_color, dist_depth, dist_color, R, T]

def regist(depth_undistort, mtx_depth, mtx_color, R, T):
    '''regist the depth frame to rgb frame'''
    depth_regist = np.zeros((640,480))
    for j in range(640):
        for i in range(480):
            depth_value = depth_undistort[i,j]
            p_rgb = np.array([[i],[j],[depth_value]])
            P_rgb = np.dot(R, np.dot(mtx_depth.transpose(), p_rgb)) + np.array(T)
            p_depth = np.dot(mtx_color, P_rgb)
            m = round(p_depth[0])
            n = round(p_depth[1])
            [m, n] = p_depth[0:2]
            if n in range(640) and m in range(480):
                depth_regist[m,n] = p_depth[2]
    return depth_regist
