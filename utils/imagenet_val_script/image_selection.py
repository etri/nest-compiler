import shutil
import sys
import tempfile

import argparse
import os
import pexpect

import PIL.Image as Image
import torchvision
import time

parser = argparse.ArgumentParser(
    description="Image Selection for Profile-Guided Quantization.")

parser.add_argument(
    "--train-images-dir",
    metavar="DIR",
    # default="/home/jemin/hdd/imagenet/train_processed_299",
    default="/home/jemin/hdd/imagenet/train_processed",
    # default="/home/jemin/hdd/imagenet/train_processed_299"
    help="Path to the directory containing the training set "
    "of images. Subdirectories are expected to be organized "
    "such that when sorted their index corresponds to their "
    "label. For example, if the validation_images_dir contains "
    "{'abc/', 'def/', 'ghi/'}, then this should correspond to "
    "labels {0, 1, 2} respectively.")

parser.add_argument(
    "--num-of-images",
    default=1,
    type=int,
    metavar="N",
    help="The number of images for profiling.",
)

parser.add_argument(
    "--file-name", type = str, default ="input.txt", help='file_name')


def get_sorted_img_subdirs(validation_images_dir):
    img_dir_paths = []
    for img_dir in os.listdir(validation_images_dir):
        dir_path = os.path.join(validation_images_dir, img_dir)
        if os.path.isdir(dir_path):
            img_dir_paths.append(img_dir)
    img_dir_paths.sort()

    return img_dir_paths


# @returns two lists of the same length found in directory
# @param validation_images_dir; the first list contains paths to all images
# found, and the second list contains the corresponding labels of the image.
def get_img_paths_and_labels(validation_images_dir):
    img_subdirs = get_sorted_img_subdirs(validation_images_dir)

    # Create lists holding paths to each image to be classified and the label
    # for that image.
    img_paths = []
    img_labels = []
    curr_label_idx = 0
    for img_subdir in img_subdirs:
        img_subdir_path = os.path.join(validation_images_dir, img_subdir)
        print("%d label, %s, # of images: %d" % (curr_label_idx,img_subdir.split("/")[-1], len(os.listdir(img_subdir_path))))
        for img in os.listdir(img_subdir_path):
            full_img_path = os.path.join(img_subdir_path, img)
            if os.path.isfile(full_img_path):
                img_paths.append(full_img_path)
                img_labels.append(curr_label_idx)
        curr_label_idx = curr_label_idx + 1
    return img_paths, img_labels

# Given an image located at @param img_path, transform the image
# and save it to the path @param path_to_new_img.

# Sequentially pick 1000 images per category
def get_img_paths_and_labels_sequence(validation_images_dir, pick_num):
    img_subdirs = get_sorted_img_subdirs(validation_images_dir)

    # Create lists holding paths to each image to be classified and the label
    # for that image.
    img_paths = []
    img_labels = []
    curr_label_idx = 0
    for img_subdir in img_subdirs:
        img_subdir_path = os.path.join(validation_images_dir, img_subdir)
        images_list = os.listdir(img_subdir_path)
        for img in images_list[0:pick_num]:
            full_img_path = os.path.join(img_subdir_path, img)
            if os.path.isfile(full_img_path):
                img_paths.append(full_img_path)
                img_labels.append(curr_label_idx)
        curr_label_idx = curr_label_idx + 1
    return img_paths, img_labels


def main():
    # Parse the recognized command line arguments into args.
    args = parser.parse_args()

    images = args.num_of_images
    file_name = args.file_name

    start = time.time()

    # Path to the directory containing the validation set of images.
    # Subdirectories are expected to be organized such that when sorted their
    # index corresponds to their label. For example, if the
    # validation_images_dir contains {'abc/', 'def/', 'ghi/'}, then this should
    # correspond to labels {0, 1, 2} respectively.
    train_images_dir = os.path.join(args.train_images_dir)
    assert os.path.exists(train_images_dir), (
        "Validation directory does not exist: " + train_images_dir)

    # get image paths
    #img_paths, img_labels = get_img_paths_and_labels(train_images_dir)
    img_paths, img_labels = get_img_paths_and_labels_sequence(train_images_dir,images)

    print("# of images: %d"%(len(img_paths)))

    # save input files
    input_txt_file = open(file_name,"w")
    for img_path in img_paths:
        input_txt_file.write(img_path+"\n")
    input_txt_file.close()

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
