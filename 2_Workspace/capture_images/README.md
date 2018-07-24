# use the librealsense in /opt/ros/kinetic/
```
virtualenv --system-site-packages py2_virtualenv
cd py2_virtualenv
source bin/activate
pip install cython testresources xlib
git clone https://github.com/toinsson/pyrealsense.git
```
```
cd pyrealsense
export PYRS_INCLUDES=/opt/ros/kinetic/include/librealsense/
```
in setup.py line 34, 35
```
inc_dirs = [np.get_include(), '/opt/ros/kinetic/include/librealsense']
lib_dirs = ['/opt/ros/kinetic/lib']
```
```
python setup.py install
```
# use the librealsense locally
```
virtualenv --system-site-packages py2_virtualenv
cd py2_virtualenv
source bin/activate
pip install cython testresources xlib
git clone https://github.com/IntelRealSense/librealsense.git
git clone https://github.com/toinsson/pyrealsense.git
```
```
cd librealsense
git checkout v1.12.1
export DESTDIR=/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv
mkdir build && cd build
cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv/bin/python
make -j4 install
```
```
cd ../../pyrealsense
export PYRS_INCLUDES=/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv/usr/local/include/librealsense/
```
in setup.py line 34, 35
```
inc_dirs = [np.get_include(), '/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv/usr/local/include/librealsense']
lib_dirs = ['/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv/usr/local/lib']
```
```
python setup.py install
```
# Run the script
```
cd ../dataset_scripts/capture_images
mkdir images
python laim_cap.py
```
if crashed -> ```pkill -f laim_cap.py```

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

used ```grep -rl "/usr/local/include/librealsense/"``` -> 3 files
- librealsense/build/install_manifest.txt
- pyrealsense/build/lib.linux-x86_64-2.7/pyrealsense/constants.py
- pyrealsense/pyrealsense/constants.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Meaning of device.depth_scale

These units are expressed as depth in meters corresponding to a depth value of 1. 
For example if we have a depth pixel with a value of 2 and the depth scale units 
are 0.5 then that pixel is 2 X 0.5 = 1 meter away from the camera.
