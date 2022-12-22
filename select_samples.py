import argparse
import numpy as np
import os
import os.path as osp
import sys
from glob import glob
import cv2
import pdb
from PIL import Image

import random

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]  

def make_parser():
    parser = argparse.ArgumentParser("Args for dastaset preperation")
    parser.add_argument("--data_root", type=str, default='./archive/')
    parser.add_argument("--save_root", type=str, default='./medical')
    parser.add_argument("--num_samp", type=int, default=3000)
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def main(args):

    if osp.isdir(args.data_root):
        files = get_image_list(args.data_root)
        files.sort()
    else:
        raise ValueError('Wrong data directory')        

    output_dir = osp.join(f'{args.save_root}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{args.save_root}/0', exist_ok=True)
    os.makedirs(f'{args.save_root}/1', exist_ok=True)

    for _, patients, _ in os.walk(f'{args.data_root}'):

        idx = list(range(0, len(patients)))
        random.shuffle(idx)
        idx0 = idx[:int(len(patients)/2)]
        idx1 = idx[int(len(patients)/2):]

        patient_for_0 = [patients[i] for i in idx0]
        patient_for_1 = [patients[i] for i in idx1]

        candidates_0 = []
        for i in patient_for_0:
            curr_folder = f'{args.data_root}/{patient_for_0[0]}/0/'
            files = glob(os.path.join(curr_folder, '*'))
            candidates_0.extend(files)
        random.shuffle(candidates_0)
        candidates_0 = candidates_0[:args.num_samp]

        candidates_1 = []
        for i in patient_for_1:
            curr_folder = f'{args.data_root}/{patient_for_1[0]}/1/'
            files = glob(os.path.join(curr_folder, '*'))
            candidates_1.extend(files)
        random.shuffle(candidates_1)
        candidates_1 = candidates_1[:args.num_samp]

        for i in range(0, len(candidates_0)):
            img = cv2.imread(candidates_0[i])
            cv2.imwrite(f'{args.save_root}/0/{i:04d}.png', img)

        for i in range(0, len(candidates_1)):
            img = cv2.imread(candidates_1[i])
            cv2.imwrite(f'{args.save_root}/1/{i:04d}.png', img)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
