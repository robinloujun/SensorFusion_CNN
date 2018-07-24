import os
from PIL import Image
import glob

def postprocess_images_from_folder(folder):
    print("Start postprocessing folders with subfix %s.\n" % folder)
    max_files_pos = 0
    max_files_neg = 0

    for filename in glob.glob(folder + "_original_color/pos/*.jpg"):
        img = Image.open(filename)
        half_width = img.size[0]/2
        half_height = img.size[1]/2
        height = img.size[1]
        img_crop = img.crop((half_width - 240, 0, half_width +240, height))
        img_299 = img_crop.resize((299, 299), Image.ANTIALIAS)
        max_files_pos += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img_299.save(folder+"_resized_color/pos/"+full_name)

    for filename in glob.glob(folder + "_original_depth/pos/*.jpg"):
        img = Image.open(filename)
        half_width = img.size[0]/2
        half_height = img.size[1]/2
        height = img.size[1]
        img_crop = img.crop((half_width - 240, 0, half_width +240, height))
        img_299 = img_crop.resize((299, 299), Image.ANTIALIAS)
        split_path = filename.split("/")
        full_name = split_path[-1]
        img_299.save(folder+"_resized_depth/pos/"+full_name)

    print("Finished postprocessing %i images with label pos.\n" % max_files_pos)

    for filename in glob.glob(folder + "_original_color/neg/*.jpg"):
        img = Image.open(filename)
        half_width = img.size[0]/2
        half_height = img.size[1]/2
        height = img.size[1]
        img_crop = img.crop((half_width - 240, 0, half_width +240, height))
        img_299 = img_crop.resize((299, 299), Image.ANTIALIAS)
        max_files_neg += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img_299.save(folder+"_resized_color/neg/"+full_name)

    for filename in glob.glob(folder + "_original_depth/neg/*.jpg"):
        img = Image.open(filename)
        half_width = img.size[0]/2
        half_height = img.size[1]/2
        height = img.size[1]
        img_crop = img.crop((half_width - 240, 0, half_width +240, height))
        img_299 = img_crop.resize((299, 299), Image.ANTIALIAS)
        split_path = filename.split("/")
        full_name = split_path[-1]
        img_299.save(folder+"_resized_depth/neg/"+full_name)

    print("Finished postprocessing %i images with label neg.\n" % max_files_neg)

def resize_img(folder, width, height):
    for filename in glob.glob(folder + "/*.jpg"):
        img = Image.open(filename)
        img_resized = img.resize((width,height), Image.ANTIALIAS)
        max_files_pos += 1
        split_path = filename.split("/")
        full_name = split_path[-1]
        img_resized.save(folder+"_resized_"+full_name)
    print("%i images have been resized to %ix%i.\n" % max_files_neg, width, height)
