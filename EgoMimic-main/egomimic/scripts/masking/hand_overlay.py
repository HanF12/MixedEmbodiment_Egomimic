from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from egomimic.utils.egomimicUtils import nds, cam_frame_to_cam_pixels, ARIA_INTRINSICS, draw_dot_on_frame
import argparse
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from egomimic.scripts.masking.utils import Inpainter, inpaint_hdf5, SAM

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"mask_{i}.png")
        plt.close()


def get_bounds(binary_image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary image
    # _,binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store max and min x and y values
    max_x = max_y = 0
    min_x = min_y = float('inf')

    if len(contours) == 0:
        return None, None, None, None

    # Loop through all contours to find max and min x and y values
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x, max_x, min_y, max_y

def line_on_hand(images, masks, arm):
    """
    Draw a line on the hand
    images: np.array of shape (n, h, w, c)
    masks: np.array of shape (n, h, w)
    arm: str, "left" or "right"
    """
    overlayed_imgs = np.zeros_like(images)
    for k, (image, mask) in enumerate(zip(images, masks)):
        min_x, max_x, min_y, max_y = get_bounds(mask.astype(np.uint8))
        if min_x is None:
            overlayed_imgs[k] = image
            continue

        gamma = 0.8
        alpha = 0.2
        scale = max_y - min_y
        min_x = int(max_x + gamma * (min_x - max_x))
        min_y = int(max_y + gamma * (min_y - max_y))
        max_x = int(max_x - scale * alpha)

        if arm == "right":
            line_image = cv2.line(image.copy(), (min_x,min_y),(max_x,max_y),color=(255,0,0), thickness=25)
        elif arm == "left":
            line_image = cv2.line(image.copy(), (min_x,max_y),(max_x,min_y),color=(255,0,0), thickness=25)
        else:
            raise ValueError(f"Invalid arm: {arm}")
        overlayed_imgs[k] = line_image
    
    return overlayed_imgs


def sam_processing(dataset, debug=False):
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    sam = SAM()

    with h5py.File(dataset, "r+") as data:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(len(data["data"].keys()))):
                demo = data[f"data/demo_{i}"]
                imgs = demo["obs/front_img_1"]
                ee_poses = demo["obs/ee_pose"]

                overlayed_imgs, masked_imgs, raw_masks = sam.get_hand_mask_line_batched(imgs, ee_poses, ARIA_INTRINSICS, debug=debug)
                
                if "front_img_1_masked" in demo["obs"]:
                    print("Deleting existing masked images")
                    del demo["obs/front_img_1_masked"]
                if "front_img_1_line" in demo["obs"]:
                    print("Deleting existing line images")
                    del demo["obs/front_img_1_line"]
                if "front_img_1_mask" in demo["obs"]:
                    print("Deleting existing masks")
                    del demo["obs/front_img_1_mask"]

                demo["obs"].create_dataset("front_img_1_masked", data=masked_imgs, chunks=(1, 480, 640, 3))
                demo["obs"].create_dataset("front_img_1_mask", data=raw_masks, chunks=(1, 480, 640), dtype=bool)
                demo["obs"].create_dataset("front_img_1_line", data=overlayed_imgs, chunks=(1, 480, 640, 3))

def main(args):
    if args.sam:
        sam_processing(args.dataset, args.debug)

    if args.inpaint:
        inpainter = Inpainter()
        inpaint_hdf5(args.dataset, inpainter)


if __name__ == '__main__':
    '''
    usage:     
    python hand_overlay.py --hdf5_file /coc/flash7/datasets/egoplay/_OBOO_ARIA/oboo_yellow_jun12/converted/oboo_yellow_jun12_ACTGMMCompat_masked.hdf5 --debug
    '''
    parser = argparse.ArgumentParser(description='Process an HDF5 file.')
    parser.add_argument('--dataset', type=str, help='HDF5 file name')
    parser.add_argument('--sam', action='store_true', help='Use SAM2 for masking')
    parser.add_argument('--inpaint', action='store_true', help='Inpaint the masked images')
    # parser.add_argument('--change_type', type=int, help='Type of change: 0-masked image 1-line on unmasked image 2-line on masked image')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    main(args)