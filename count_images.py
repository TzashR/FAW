import os
import argparse
import re

ap=argparse.ArgumentParser()

ap.add_argument('images_root_dir',type=str)
args=ap.parse_args()


existing_images = []
for subdir, dirs, files in os.walk(args.images_root_dir):
    for filename in files:
        filepath = subdir = os.sep = filename
        if re.search("[0-9]*_[0-9]*.jpg", filepath):
            existing_images.append(re.findall("[0-9]*_[0-9]*.jpg", filepath)[0][:-4])
print(f"existing images len = {len(existing_images)}")
reported_pics = {}
# for image in id_image_list:
#     reported_pics[re.findall("[0-9]*_[0-9]*.jpg", image[1])[0][:-4]] = 1
#
#
# missing = 0
# valid = 0

# f = open("missing_images.txt",'w')
#
# for im in existing_images:
#     if im in reported_pics:
#         valid += 1
#     else:
#         # print(im)
#         # f.write(im + "\n")
#         # missing += 1

# print(f"missing  = {missing}")