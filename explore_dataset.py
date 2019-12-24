import argparse
import PIL
import os
import matplotlib.pyplot as plt
import numpy as np

import utils

SHOW_IMAGES = True

MAP_PART_TO_DARKNESS = {}
darkness_increment = 1
darkness = 1

result_dir = "results_map"
 
MAP_PART_TO_CLASS = {
    'head': 1, 'hair': 2, 'torso': 3, 'llarm': 4, 'luarm': 5, 'llleg': 6, 'luleg': 7, 'lfoot': 8, 'rlleg': 9, 'ruleg': 10, 'rfoot': 11, 'lhand': 12, 'rlarm': 13, 'rhand': 14, 'ruarm': 15, 'fwheel': 16, 'bwheel': 17, 'handlebar': 18, 'chainwheel': 19, 'saddle': 20, 'beak': 21, 'rleg': 22, 'tail': 23, 'frontside': 24, 'rightside': 25, 'window_1': 26, 'headlight_1': 27, 'wheel_1': 28, 'fliplate': 29, 'rightmirror': 30, 'headlight_2': 31, 'wheel_2': 32, 'window_2': 33, 'lear': 34, 'rear': 35, 'leye': 36, 'reye': 37, 'lebrow': 38, 'rebrow': 39, 'mouth': 40, 'nose': 41, 'neck': 42, 'pot': 43, 'plant': 44, 'leftside': 45, 'body': 46, 'cap': 47, 'door_1': 48, 'lwing': 49, 'rwing': 50, 'muzzle': 51, 'lfleg': 52, 'lfpa': 53, 'rfleg': 54, 'rfpa': 55, 'rbleg': 56, 'rbpa': 57, 'lbleg': 58, 'lbpa': 59, 'lleg': 60, 'screen': 61, 'leftmirror': 62, 'window_3': 63, 'coach_1': 64, 'coach_2': 65, 'lflleg': 66, 'lfuleg': 67, 'lfho': 68, 'rflleg': 69, 'rfho': 70, 'lblleg': 71, 'lbuleg': 72, 'lbho': 73, 'stern': 74, 'engine_1': 75, 'coach_3': 76, 'coach_4': 77, 'cleftside_1': 78, 'cleftside_2': 79, 'cleftside_3': 80, 'cleftside_4': 81, 'hfrontside': 82, 'hleftside': 83, 'cfrontside_1': 84, 'rfuleg': 85, 'rbuleg': 86, 'crightside_1': 87, 'hrightside': 88, 'hroofside': 89, 'backside': 90, 'wheel_3': 91, 'window_4': 92, 'window_5': 93, 'window_6': 94, 'bliplate': 95, 'crightside_2': 96, 'crightside_3': 97, 'croofside_1': 98, 'croofside_2': 99, 'croofside_3': 100, 'lhorn': 101, 'rhorn': 102, 'roofside': 103, 'rblleg': 104, 'headlight_3': 105, 'window_7': 106, 'window_8': 107, 'window_9': 108, 'rbho': 109, 'door_2': 110, 'door_3': 111, 'headlight_4': 112, 'headlight_5': 113, 'headlight_6': 114, 'hbackside': 115, 'engine_2': 116, 'window_10': 117, 'wheel_4': 118, 'wheel_5': 119, 'wheel_6': 120, 'coach_5': 121, 'coach_6': 122, 'cleftside_5': 123, 'cleftside_6': 124, 'cbackside_1': 125, 'cfrontside_2': 126, 'wheel_7': 127, 'wheel_8': 128, 'engine_3': 129, 'engine_4': 130, 'coach_7': 131, 'coach_8': 132, 'coach_9': 133, 'cleftside_7': 134, 'cleftside_8': 135, 'cleftside_9': 136, 'cfrontside_3': 137, 'cfrontside_4': 138, 'cfrontside_5': 139, 'cfrontside_6': 140, 'cfrontside_7': 141, 'cfrontside_9': 142, 'cbackside_2': 143, 'headlight_7': 144, 'window_11': 145, 'window_12': 146, 'window_13': 147, 'window_14': 148, 'window_15': 149, 'window_16': 150, 'window_17': 151, 'window_18': 152, 'window_19': 153, 'window_20': 154, 'crightside_4': 155, 'crightside_5': 156, 'croofside_4': 157, 'headlight_8': 158, 'crightside_6': 159, 'crightside_7': 160, 'croofside_5': 161, 'engine_5': 162, 'engine_6': 163, 'door_4': 164, 'crightside_8': 165
}

# Plot a mask composed by 0s and 1s with a certain title
# and compare it with the original image:
def plot_mask(img, mask, filename):
    mask = PIL.Image.fromarray(mask)
    # bodypart_mask = PIL.Image.fromarray(bodypart_mask * 255)
    fig = plt.figure(dpi=300)
    # fig.canvas.set_window_title(windowtitle)
    fig.suptitle(filename)
    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(img)
    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    new_filename = ''.join(filename.split('.')[0:-1]) + '.png'
    mask.convert('RGB').save(os.path.join(result_dir, new_filename))
    plt.imshow(mask)
    # plt.show()
    print("Number of parts:", len(MAP_PART_TO_DARKNESS))
    x = input()
    if x == 'exit':
        exit(0)

# Load annotations from the annotation folder of PASCAL-Part dataset:
if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Extract data from PASCAL-Part Dataset")
    parser.add_argument("--annotation_folder", default="datasets/trainval/Annotations_Part", help="Path to the PASCAL-Part Dataset annotation folder")
    parser.add_argument("--images_folder", default="datasets/VOCdevkit/VOC2010/JPEGImages", help="Path to the PASCAL VOC 2010 JPEG images")
    args = parser.parse_args()

    # Stats on the dataset:
    obj_cnt = 0
    bodypart_cnt = 0

    mat_filenames = os.listdir(args.annotation_folder)

    # Iterate through the .mat files contained in path:
    for idx, annotation_filename in enumerate(mat_filenames):
        annotations = utils.load_annotations(os.path.join(args.annotation_folder, annotation_filename))
        image_filename = annotation_filename[:annotation_filename.rfind(".")] + ".jpg" # PASCAL VOC image have .jpg format

        obj_cnt += len(annotations["objects"])

        # Show original image with its mask:
        img = PIL.Image.open(os.path.join(args.images_folder, image_filename))
        img_width, img_height = img.size
        total_mask = np.zeros([img_height, img_width])
        for obj in annotations["objects"]:
            bodypart_cnt += len(obj["parts"])
            print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt), end="\r")
            if SHOW_IMAGES:
                for body_part in obj["parts"]:
                    if body_part['part_name'] not in MAP_PART_TO_DARKNESS:
                        MAP_PART_TO_DARKNESS[body_part['part_name']] = darkness
                        darkness += darkness_increment
                    total_mask = np.maximum(total_mask, body_part["mask"] * MAP_PART_TO_DARKNESS[body_part['part_name']])
        
        print(MAP_PART_TO_DARKNESS)
        plot_mask(img, total_mask, image_filename)


    print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt))

    print(MAP_PART_TO_DARKNESS)