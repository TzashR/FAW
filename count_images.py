import os
import argparse
import re
import cv2
import pickle
def main():
    ap=argparse.ArgumentParser()

    ap.add_argument('images_root_dir',type=str)
    args=ap.parse_args()
    wanted_shapes = {(960,1600,3), (1600,960,3)}
    existing_images = []
    bad_shaped = []
    for subdir, dirs, files in os.walk(args.images_root_dir):
        for filename in files:
            if re.search("[0-9]*_[0-9]*.jpg", filename):
                fp = os.path.join(args.images_root_dir,subdir,filename)
                img = cv2.imread(fp)
                shape = img.shape
                if shape not in wanted_shapes:
                    bad_shaped.append(fp)

                # count images with bad shape:
                # img = cv2.imread(filepath)
                # print(img.shape)

            existing_images.append(re.findall("[0-9]*_[0-9]*.jpg", filename)[0][:-4])
    print(f"existing images len = {len(existing_images)}")
    print(f'bad shaped {len(bad_shaped)}')

    with open('bad shaped images2','wb') as f:
        pickle.dump(set(bad_shaped),f)

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

        # print(f"missing  = {missing}"

