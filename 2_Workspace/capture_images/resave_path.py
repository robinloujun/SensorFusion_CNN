import os
from PIL import Image
import glob
import cv2
import numpy as np

folder = '/home/lou/Feature_Embedding/2_Workspace/py2_virtualenv/dataset_scripts/capture_images/Dataset/'
resaved_folder = '/home/lou/Dataset/'
max_files_pos_color = 0
max_files_pos_depth = 0
max_files_neg_color = 0
max_files_neg_depth = 0

print("Start resave with subfix %s.\n" % folder)

for folder_name in os.listdir(folder):

    dirname = folder + folder_name
    if not os.path.exists(resaved_folder+folder_name+"_color/pos/"):
        os.makedirs(resaved_folder+folder_name+"_color/pos/")
    for filename in glob.glob(dirname + "/pos/*rgb*.jpg"):
        img = Image.open(filename)
        max_files_pos_color += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img.save(resaved_folder+folder_name+"_color/pos/"+full_name)
print("Resaved %i color images with label pos.\n" % max_files_pos_color)


for folder_name in os.listdir(folder):

    dirname = folder + folder_name
    if not os.path.exists(resaved_folder+folder_name+"_depth/pos/"):
        os.makedirs(resaved_folder+folder_name+"_depth/pos/")
    for filename in glob.glob(dirname + "/pos/dep*.jpg"):
        img = Image.open(filename)
        max_files_pos_depth += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img.save(resaved_folder+folder_name+"_depth/pos/"+full_name)
print("Resaved %i depth images with label pos.\n" % max_files_pos_depth)

#for folder_name in os.listdir(folder):

    #dirname = folder + folder_name
    #if not os.path.exists(resaved_folder+folder_name+"_depth/pos/"):
        #os.makedirs(resaved_folder+folder_name+"_depth/pos/")
    #for filename in glob.glob(dirname + "/pos/*dep*.jpg"):
        #img = cv2.imread(filename)
        #max_files_pos_depth += 1
        #img_np = np.asarray(img)
        #img_dep = np.sum(img_np, axis=-1)
        #img_uint8 = Image.fromarray(img_dep.astype('uint8'))
        #split_path = filename.split("/")
        #full_name = split_path[-1]
        #img_uint8.save(resaved_folder+folder_name+"_depth/pos/"+full_name)
#print("Resaved %i depth images with label pos.\n" % max_files_pos_depth)

for folder_name in os.listdir(folder):

    dirname = folder + folder_name
    if not os.path.exists(resaved_folder+folder_name+"_color/neg/"):
        os.makedirs(resaved_folder+folder_name+"_color/neg/")
    for filename in glob.glob(dirname + "/neg/*rgb*.jpg"):
        img = Image.open(filename)
        max_files_neg_color += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img.save(resaved_folder+folder_name+"_color/neg/"+full_name)
print("Resaved %i color images with label neg.\n" % max_files_neg_color)

for folder_name in os.listdir(folder):

    dirname = folder + folder_name
    if not os.path.exists(resaved_folder+folder_name+"_depth/neg/"):
        os.makedirs(resaved_folder+folder_name+"_depth/neg/")
    for filename in glob.glob(dirname + "/neg/dep*.jpg"):
        img = Image.open(filename)
        max_files_neg_depth += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img.save(resaved_folder+folder_name+"_depth/neg/"+full_name)
print("Resaved %i depth images with label neg.\n" % max_files_neg_depth)
