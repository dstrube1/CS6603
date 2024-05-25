import os
import time
import pandas as pd
from IPython.core.display_functions import display
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from PIL import Image
from PIL import ImageChops
import re


# https://pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(vals_1, vals_2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((vals_1[0] - vals_2[0]) ** 2)
    err /= float(vals_1[1] * vals_1[2])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def diff_pil(file_name_1, file_name_2, folder_path):
    path_one = folder_path + file_name_1
    path_two = folder_path + file_name_2
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    try:
        diff = ImageChops.difference(image_one, image_two)
        if diff.getbbox() is None:
            # same
            print('same: ' + file_name_1 + ', ' + file_name_2)
        else:
            print('different: ' + file_name_1 + ', ' + file_name_2)
    except ValueError as e:
        print('Caught ValueError')


def diff_skimage(file_name_1, file_name_2, folder_path):
    image_a = cv2.imread(folder_path + file_name_1)
    image_b = cv2.imread(folder_path + file_name_2)
    # convert the images to grayscale
    image_one = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_two = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    s = ssim(image_one, image_two, gaussian_weights=True)
    print('ssim: ' + str(s))


def get_images_values(files, folder_path):
    image_values = {}

    for image_file in files:
        image = cv2.imread(folder_path + image_file)
        if image is not None:
            image_values[image_file] = [image.astype('float'), image.shape[0], image.shape[1]]
    return image_values


def get_time(seconds):
    # Copied from my CS 7641 Assignment 2
    if int(seconds / 60) == 0:
        if int(seconds) == 0:
            return str(round(seconds, 3)) + ' seconds'
        return str(int(seconds)) + ' second(s)'
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    if int(minutes / 60) == 0:
        return str(minutes) + ' minute(s) and ' + str(seconds) + ' second(s)'
    hours = int(minutes / 60)
    minutes = int(minutes % 60)
    # Assuming this won't be called for any time span greater than 24 hours
    return str(hours) + ' hour(s), ' + str(minutes) + ' minute(s), and ' + str(seconds) + ' second(s)'


# https://www.geeksforgeeks.org/python-list-files-in-a-directory/
path = '../crop_part1/'
dir_list = os.listdir(path)
dir_list.sort()
compare_list = os.listdir(path)
compare_list.sort()
images_values = get_images_values(dir_list, path)

print('files count: ' + str(len(dir_list)))
"""
# fileA = '96_1_0_20170110183855839.jpg.chip.jpg'  # same
# fileB = '96_1_1_20170110183853718.jpg.chip.jpg'  # same
# fileC = '96_1_0_20170110182515404.jpg.chip.jpg'  # different
# m = mse(images_values[fileA], images_values[fileB])
# print('mse (A & B): ' + str(m))  # 0.0
# m = mse(images_values[fileA], images_values[fileC])
# print('mse (A & C): ' + str(m))  # 7251.330525


# fileA = '90_1_0_20170110182841384.jpg.chip.jpg'  # same, but blurry :/, mse > 100
# fileB = '96_1_0_20170110182019881.jpg.chip.jpg'
# m = mse(fileA, fileB, path)
# print('mse (A & B): ' + str(m)) # MSE = 187.775225
# Taking this as a baseline for nearly identical images
"""

duplicates = []
anomalies = []
index = 0
print('Looking for duplicates & bad file names. "." = 1 file checked. "#" = duplicate found...')
start = time.time()
for file in dir_list:
    if file not in images_values.keys():
        # File didn't make it into images_values (like .DS_Store) - skip it
        compare_list.remove(file)
        anomalies.append(file)
        continue
    file_parts = file.split('_')
    if len(file_parts) < 4:
        print('\nbad file name: ' + file)
    if file not in compare_list:
        # Previously detected duplicate
        continue
    compare_list.remove(file)
    for duplicate in duplicates:
        if duplicate in compare_list:
            compare_list.remove(duplicate)
    for file_other in compare_list:
        m = mse(images_values[file], images_values[file_other])
        if m < 188:
            # print(file + ' seems to be a duplicate of ' + file_other)
            print('#', end='')
            # duplicates.append(file)
            duplicates.append(file_other)
            """
            12 duplicates found below 0.35% (unsorted), including:
            21_0_4_20161223214826657.jpg.chip.jpg, 6_1_4_20170103230723185.jpg.chip.jpg
            I highly doubt subject is really 21 years old in this picture
            """
    print('.', end='')
    index += 1
    if index % 100 == 0:
        print()

end = time.time()
print('\nDone in ' + get_time(end - start))
print('Found ' + str(len(duplicates)) + ' duplicates.')
for duplicate in duplicates:
    if duplicate in dir_list:
        dir_list.remove(duplicate)

for anomaly in anomalies:
    if anomaly in dir_list:
        dir_list.remove(anomaly)

# Data has been cleaned up some. (Could probably do more, but this should be good enough for now)

genders = {0: 'male', 1: 'female'}
races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other'}
img_names = []
for file in dir_list:
    img_names.append(file.replace('.jpg.chip.jpg', ''))

age_pattern = re.compile(r"(^\d{1,3}).+")
gender_pattern = re.compile(r"^\d{1,3}_(\d).+")
race_pattern = re.compile(r"^\d{1,3}_\d_(\d).+")
age_list = []
gender_list = []
race_list = []
for name in img_names:
    age_list.append(int(re.match(age_pattern, name).group(1)))
    gender_list.append(int(re.match(gender_pattern, name).group(1)))
    race_list.append(int(re.match(race_pattern, name).group(1)))

img_df = pd.DataFrame({
    "img_name": img_names, "ages": age_list,
    "genders": gender_list, "races": race_list
    })

bins = pd.IntervalIndex.from_tuples([(0, 20), (21, 40), (41, 60), (61, 80), (81, 116)])
img_df["age"] = pd.cut(img_df["ages"], bins)

img_df["gender"] = img_df["genders"].map(genders)

img_df["race"] = img_df["races"].map(races)

df = img_df[["age", "gender", "race"]]

age_value_counts = df["age"].value_counts()
age_group_max = age_value_counts[age_value_counts == age_value_counts.max()]
age_proportion_max = 100 * (age_group_max.values[0] / age_value_counts.sum())
age_group_min = age_value_counts[age_value_counts == age_value_counts.min()]
age_proportion_min = 100 * (age_group_min.values[0] / age_value_counts.sum())
print(f" - Age group with largest representation: {age_group_max.index[0]} ({age_proportion_max:.2f}%)")


gender_value_counts = df["gender"].value_counts()
gender_group_max = gender_value_counts[gender_value_counts == gender_value_counts.max()]
gender_proportion_max = 100 * (gender_group_max.values[0] / gender_value_counts.sum())
gender_group_min = gender_value_counts[gender_value_counts == gender_value_counts.min()]
gender_proportion_min = 100 * (gender_group_min.values[0] / gender_value_counts.sum())
print(f" - Gender group with largest representation: {gender_group_max.index[0]} ({gender_proportion_max:.2f}%)")

race_value_counts = df["race"].value_counts()
race_group_max = race_value_counts[race_value_counts == race_value_counts.max()]
race_proportion_max = 100 * (race_group_max.values[0] / race_value_counts.sum())
race_group_min = race_value_counts[race_value_counts == race_value_counts.min()]
race_proportion_min = 100 * (race_group_min.values[0] / race_value_counts.sum())
print(f" - Race group with largest representation: {race_group_max.index[0]} ({race_proportion_max:.2f}%)")

df1 = df[["age", "gender"]].value_counts().reset_index(drop=False).rename({0: "n"}, axis=1)
df1 = df1.pivot(index="gender", columns="age", values="n").rename_axis(None)
df1.columns.name = None

df2 = df[["age", "race"]].value_counts().reset_index(drop=False).rename({0: "n"}, axis=1)
df2 = df2.pivot(index="race", columns="age", values="n").rename_axis(None)
df2.columns.name = None

dfc = pd.concat([df1, df2])
dfc["total"] = dfc.sum(axis=1)

display(dfc)


