## Acquire the sensor data of TNS

```
roslaunch tnsrosnode TNSv3_1_gripper_joint.launch
rostopic echo /GripperJoint/tnsv3_1/sensor_data
rostopic hz /GripperJoint/tnsv3_1/sensor_data
```

```
rqt_plot
/GripperJoint/tnsv3_1/sensor_data/Sensors[0]/data[0]
```
\# data[0]-data[7]

## control the WSG_50

[source](https://code.google.com/archive/p/wsg50-ros-pkg/wikis/wsg_50.wiki)

```
roslaunch wsg50_drive wsg50_tcp.lanch
rosservice call /wsg50_driver/move "width: 50.0 speed: 100.0" 
rosservice call /wsg50_driver/grasp "width: 20.0 speed: 100.0" 
```
## get the images from RealSense camera SR300

```
roslaunch realsense_camera sr300_nodelet_rgbd.launch 
rosrun image_view image_view image:=/camera/rgb/image_color
rosrun image_view image_view image:=/camera/depth/image
```

## calibrate the camera

[source](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration)

```
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 image:=/camera/rgb/image_color camera:=/camera/rgb --no-service-check 
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 image:=/camera/depth/image_raw camera:=/camera/depth
```
regard the two cameras as a [stereo camera](http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration)
```
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.108 right:=/camera/rgb/image_raw left:=/camera/depth/image_raw --no-service-check
```