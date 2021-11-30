from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import glob
import random
import cv2

from libs import utils_custom
from PIL import Image, ImageFont
from davisinteractive.utils.visualization import *

from config import *
Config = Config()

def _pascal_color_map(N=256, normalized=True):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def visualize_interactionIoU(IoUlist, figtitle, annotated_frames=None, save_dir=None, thresholding_probmap=False, show_propagated_region = False):
    '''
    param IoUlist - [n_interaction,n_frames]
    param save_dir:
    return:
    '''
    n_interact = len(IoUlist)
    n_frames = len(IoUlist[0])
    x = np.arange(n_frames)
    axs = dict()
    fig = plt.figure(figsize=(10,n_interact*2.5))
    fig.suptitle('{}'.format(figtitle),fontsize=20)

    IoUlist = np.array(IoUlist)


    for idx in range(1,n_interact+1): # the first figure
        axs[idx] = fig.add_subplot(n_interact*100+10+idx)  # the first subplot in the first figure
        axs[idx].plot(x,IoUlist[idx-1])
        if annotated_frames is not None:
            annotated_frames = np.array(annotated_frames)
            if thresholding_probmap:
                axs[idx].plot(annotated_frames[0], IoUlist[0][annotated_frames[0]], 'ro')
                axs[idx].plot(annotated_frames[1:], IoUlist[0][annotated_frames[1:]], 'gx')

            else:
                axs[idx].plot(annotated_frames[idx - 1], IoUlist[idx - 1][annotated_frames[idx - 1]],'ro')
                if idx>=2:
                    axs[idx].plot(annotated_frames[:idx - 1], IoUlist[idx - 1][annotated_frames[:idx - 1]], 'gx')

            if show_propagated_region:
                prop_list = utils_custom.get_prop_list(annotated_frames[:idx], annotated_frames[idx-1], n_frames, proportion=Config.test_propagation_proportion)
                prop_list = list(range(np.min(prop_list), np.max(prop_list)+1))
                axs[idx].plot(prop_list, IoUlist[idx - 1][prop_list], 'r')
                axs[idx].plot(prop_list, IoUlist[idx - 1][prop_list], 'r')

        axs[idx].set_ylim([0, 1])
        axs[idx].set_xlim([0, n_frames-1])
        axs[idx].yaxis.grid(True)

    plt.savefig(save_dir)
    plt.close(fig)




def visualize_scrib_interaction(scribbles, anno_dict, sequence, save_dir, scribbles_in_image = False):
    images_dirlist = sorted(glob.glob(os.path.join(Config.davis_dataset_dir, 'JPEGImages/480p', sequence, '*.jpg')))
    gt_dirlist = sorted(glob.glob(os.path.join(Config.davis_dataset_dir, 'Annotations/480p', sequence, '*.png')))
    font_helveltica = Config.font_dir + 'helvetica.ttf'
    selected_font = ImageFont.truetype(font_helveltica, size=45)
    if 'frames' in anno_dict :
        scrib_frame_list = anno_dict['frames']
        imgshape = anno_dict['annotated_masks'][0].shape
    else:
        scrib_frame_list = [anno_dict['annotated_frame']]
        imgshape = plt.imread(images_dirlist[0]).shape[:2]
        # anno_dict['masks'] = np.zeros([len(images_dirlist),imgshape[0],imgshape[1]])

    h = imgshape[0]
    w = imgshape[1]

    canvasImg = Image.new('RGB', (w*3, h*len(scrib_frame_list)))
    draw = ImageDraw.Draw(canvasImg)

    for i, scrib_idx in enumerate(scrib_frame_list):
        if i == 0:
            imgnp = plt.imread(images_dirlist[scrib_idx])
            gtnp = (plt.imread(gt_dirlist[scrib_idx])*255.999).astype("uint8")
            after_np = overlay_mask(imgnp, anno_dict['annotated_masks'][i], alpha=0.5, colors=None, contour_thickness=2)

            gt_Img = Image.fromarray(gtnp)
            if scribbles_in_image:
                points_np = scribbles[i]
                kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
                points_np = cv2.dilate(points_np.astype(np.uint8), kernel=kernel_fg)
                befor_Img = overlay_mask(Image.fromarray(imgnp), points_np.astype(np.uint8), alpha=0, colors=None, contour_thickness=2)
                befor_Img = Image.fromarray(befor_Img)
            else:
                befor_Img = draw_scribble(Image.fromarray(imgnp), scribbles, output_size=(w,h), frame=scrib_idx, width=7)

            after_Img = Image.fromarray(after_np)

            canvasImg.paste(gt_Img, (0, 0))
            canvasImg.paste(befor_Img, (w, 0))
            canvasImg.paste(after_Img, (w*2, 0))

            draw.multiline_text((5,h*i+5), 'Frame: '+str(scrib_idx), fill=(255, 255, 255, 255),
                                font=selected_font, spacing=1.5, align="right")

        else:
            imgnp = plt.imread(images_dirlist[scrib_idx])
            gtnp = (plt.imread(gt_dirlist[scrib_idx])*255.999).astype("uint8")
            after_np = overlay_mask(imgnp, anno_dict['annotated_masks'][i], alpha=0.5, colors=None, contour_thickness=2)

            gt_Img = Image.fromarray(gtnp)
            if scribbles_in_image:
                points_np = scribbles[i]
                kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                points_np = cv2.dilate(points_np.astype(np.uint8), kernel=kernel_fg)
                befor_Img = overlay_mask(Image.fromarray(imgnp), points_np.astype(np.uint8), alpha=0, colors=None, contour_thickness=3)
                befor_Img = Image.fromarray(befor_Img)
            else:
                befor_np = overlay_mask(imgnp, anno_dict['masks_tobe_modified'][i], alpha=0.5, colors=None, contour_thickness=2)
                befor_Img = draw_scribble(Image.fromarray(befor_np), scribbles, output_size=(w,h), frame=scrib_idx, width=7)
            after_Img = Image.fromarray(after_np)

            canvasImg.paste(gt_Img, (0, h*i))
            canvasImg.paste(befor_Img, (w, h*i))
            canvasImg.paste(after_Img, (w * 2, h*i))

            draw.multiline_text((5, h * i + 5), 'Frame: ' + str(scrib_idx), fill=(255, 255, 255, 255),
                                font=selected_font, spacing=1.5, align="right")


    canvasImg = canvasImg.resize((w*3//2,h*len(scrib_frame_list)//2), Image.ANTIALIAS)
    canvasImg.save(save_dir)


