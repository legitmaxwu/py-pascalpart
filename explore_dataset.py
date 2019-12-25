import argparse
import PIL
import os
import matplotlib.pyplot as plt
import numpy as np

import utils

from mappings import MAP_PART_TO_CLASS, OBJECTS_FILTER


MAP_CLASS_TO_DARKNESS = {}
CLASS_COUNTS = {}
darkness_increment = 1
darkness = 1
skip_to_end = False

result_dir = "results_map"
 
# TODO: mkdir results_map


# Plot a mask composed by 0s and 1s with a certain title
# and compare it with the original image:
def plot_mask(img, mask, filename):
    mask = PIL.Image.fromarray(mask)
    new_filename = ''.join(filename.split('.')[0:-1]) + '.png'
    mask.convert('RGB').save(os.path.join(result_dir, new_filename))
    # print("Number of parts:", len(MAP_CLASS_TO_DARKNESS))
    
    global skip_to_end
    if skip_to_end == False:
        x = input()
        if x == 'skip':
            skip_to_end = True
        elif x == 'show':
            fig = plt.figure(dpi=300)
            fig.suptitle(filename)
            fig.add_subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(img)
            fig.add_subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(mask)
            plt.show()
        elif x == 'print':   
            print()
            print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt))
            print("Class to darkness mapping:")
            print(MAP_CLASS_TO_DARKNESS)
            print("Class counts:")
            print(CLASS_COUNTS)
        elif x == 'exit':
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
    processed = 0
    skipped = 0

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

        labelExists = False
        for obj in annotations["objects"]:
            obj_name = obj['class']
            if obj_name not in OBJECTS_FILTER:
                continue
            bodypart_cnt += len(obj["parts"])

            for body_part in obj["parts"]:
                part_name = body_part['part_name']
                if part_name in MAP_PART_TO_CLASS:
                    labelExists = True
                    class_name = MAP_PART_TO_CLASS[part_name]
                    real_class = obj_name + '_' + class_name
                    if real_class not in MAP_CLASS_TO_DARKNESS:
                        MAP_CLASS_TO_DARKNESS[real_class] = darkness
                        darkness += darkness_increment
                    total_mask = np.maximum(total_mask, body_part["mask"] * MAP_CLASS_TO_DARKNESS[real_class])

                    if real_class not in CLASS_COUNTS:
                        CLASS_COUNTS[real_class] = 0
                    else:
                        CLASS_COUNTS[real_class] += 1
        
        if labelExists == True:
            # print(MAP_CLASS_TO_DARKNESS)
            processed += 1
            plot_mask(img, total_mask, image_filename)
        else:
            skipped += 1
            # print("No parts found... skipping image.")

        print("processed: {} - skipped: {}".format(processed, skipped), end="\r")
        


    print()
    print("obj_cnt: {} - bodypart_cnt: {}".format(obj_cnt, bodypart_cnt))
    print("Class to darkness mapping:")
    print(MAP_CLASS_TO_DARKNESS)
    print("Class counts:")
    print(CLASS_COUNTS)