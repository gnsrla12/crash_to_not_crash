from PIL import Image
from scipy.misc import imresize
import numpy as np
import json
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def open_and_preprocess_image(image):
    image = np.array(Image.open(image))
    image = image[100:360,:,0:3]
    image = imresize(image,(130,355))
    return image

def json_reader(root, data):
	file_no = data[1].split('/')[-1].split('.')[0]
	try:
		if data[0] == 0:
			frame_info = json.load(open(root + 'accident/' + file_no + '.json'))
		elif data[0] == 1:
			frame_info = json.load(open(root + 'nonaccident/' + file_no + '.json'))
	except:
		return []

	bbox_dict_list = []
	for bbox in frame_info['BoundingBoxInfo']:
		try:
			if bbox['collidingObj'] == "true":
				colliding_obj = True
			else:
				colliding_obj = False
		except:
			colliding_obj = False
		bbox_dict={
			"colliding_obj" : colliding_obj,
			"hashcode" : bbox['hashcode'],
			"x" : int(bbox['x']),
			"y" : int(bbox['y']),
			"width" : int(bbox['width']),
			"height" : int(bbox['height']),
		}
		bbox_dict_list.append(bbox_dict)
	return bbox_dict_list

def generate_bbox_info(root, batch_data):
	bbox_info = []
	for data in batch_data:
		bbox_dict_list = json_reader(root, data)
		bbox_info.append(bbox_dict_list)

	return bbox_info