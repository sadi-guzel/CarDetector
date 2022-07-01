import os
import cv2
import glob
import json
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImagePath, ImageOps
import gc
import shutil

classes = dict()
classes["stone"] = "#ffffff"

# prepare dataset folders
dataset_path = "./bdd100k-dataset/"  # path for dataset to be created
val_json_path = './bdd100k/labels/pan_seg/polygons/pan_seg_val.json'  # BDD100K polygons path
val_image_path = './10k/val/'  # BDD100K images path

with open(val_json_path, 'r', encoding='utf-8') as file:
    annotations = json.load(file)
    for a in annotations:
        try:
            img_path = val_image_path + a['name']
            if os.path.exists(img_path):
                imgg = Image.open(img_path)
                blank_img = Image.new("RGB", (imgg.size[0], imgg.size[1]), "#000000")
                for lbl in a['labels']:
                    if lbl['category'] in 'car, trailer, bus, rider, motorcycle, truck':
                        draw = ImageDraw.Draw(blank_img)
                        tuple_points = []
                        for p in lbl['poly2d'][0]['vertices']:
                            tuple_points.append(tuple(p))
                        draw.polygon(tuple_points, fill="#FFFFFF")
                blank_img.save(dataset_path + 'masks/' + os.path.basename(img_path)[:-4] + '.png', "PNG")
                imgg.save(dataset_path + 'images/' + os.path.basename(img_path)[:-4] + '.png', "PNG")
        except Exception as e:
            print(str(e))
