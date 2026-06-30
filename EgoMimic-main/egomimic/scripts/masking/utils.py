import torch
# from diffusers import AutoPipelineForInpainting
# from diffusers.utils import load_image, make_image_grid
import h5py
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TVF
import matplotlib.pyplot as plt
import egomimic
import os
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from egomimic.utils.egomimicUtils import AlohaFK, ee_pose_to_cam_pixels, ARIA_INTRINSICS, EXTRINSICS, ee_pose_to_cam_frame, cam_frame_to_cam_pixels, draw_dot_on_frame
import cv2

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

def get_valid_points(points, img_shape):
    """
    tuple of 3 (x,y) points
    """
    pt1, pt2, pt3 = points
    keep_pt1 =True
    keep_pt2 =True
    keep_pt3 =True

    if (pt1[0][0] < 0 or pt1[0][0] >= img_shape[1]) or (pt1[0][1] < 0 or pt1[0][1] >= img_shape[0]):
        keep_pt1 = False
    if (pt2[0][0] < 0 or pt2[0][0] >= img_shape[1]) or (pt2[0][1] < 0 or pt2[0][1] >= img_shape[0]):
        keep_pt2 = False
    if (pt3[0][0] < 0 or pt3[0][0] >= img_shape[1]) or (pt3[0][1] < 0 or pt3[0][1] >= img_shape[0]):
        keep_pt3 = False

    if keep_pt1 and keep_pt2 and keep_pt3:
        # print("1st case")
        input_point = np.concatenate([pt1, pt2, pt3], axis=0)
        input_label = np.array([1, 1, 1])
    elif keep_pt1 and not keep_pt2 and keep_pt3:
        # print("2nd case")
        input_point = np.concatenate([pt1, pt3], axis=0)
        input_label = np.array([1, 1])
    elif not keep_pt1 and keep_pt2 and keep_pt3:
        # print("3rd case")
        input_point = np.concatenate([pt3], axis=0)
        input_label = np.array([1])
    elif keep_pt1 and keep_pt2 and not keep_pt3:
        # print("4th case")
        input_point = np.concatenate([pt1, pt2], axis=0)
        input_label = np.array([1, 1])
    elif keep_pt1 and not keep_pt2 and not keep_pt3:
        # print("5th case")
        input_point = pt1
        input_label = np.array([1])
    elif not keep_pt1 and keep_pt2 and not keep_pt3:
        # print("6th case")
        input_point = pt2
        input_label = np.array([1])
    elif not keep_pt1 and not keep_pt2 and keep_pt3:
        # print("7th case")
        input_point = pt3
        input_label = np.array([1])
    else:
        # print("8th case")
        input_point = np.array([])
        input_label = np.array([])

    return input_point, input_label

class SAM:
    def __init__(self):
        sam2_checkpoint = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/sam2_hiera_tiny.pt"
        )
        model_cfg = "sam2_hiera_t.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

        predictor = SAM2ImagePredictor(sam2_model)

        self.predictor = predictor
        self.fk = AlohaFK()
    
    def get_robot_mask_line_batched_from_qpos(self, images, qpos, extrinsics, intrinsics, arm="right"):
        """
            images: tensor (B, H, W, 3)
            qpos: B, 7
        """
        px_dict = self.project_joint_positions_to_image(qpos, extrinsics, intrinsics, arm=arm)
        mask_images, line_images =  self.get_robot_mask_line_batched(images, px_dict, arm=arm)
        return mask_images, line_images
    
    def get_hand_mask_line_batched(self, imgs, ee_poses, intrinsics, debug=False):
        ## both hands
        if ee_poses.shape[-1] == 6:
            prompts_l = cam_frame_to_cam_pixels(ee_poses[:, :3], intrinsics)[:, :2]
            prompts_r = cam_frame_to_cam_pixels(ee_poses[:, 3:], intrinsics)[:, :2]

            masked_img_l, raw_masks_l = self.get_hand_mask_batched(imgs, prompts_l, neg_prompts=prompts_r)
            mask = np.arange(640)[None, :] < prompts_l[:, [0]] + 100
            mask = mask[:, None, :]
            raw_masks_l = raw_masks_l & mask

            masked_img_r, raw_masks_r = self.get_hand_mask_batched(imgs, prompts_r, neg_prompts=prompts_l)
            mask = np.arange(640)[None, :] > prompts_r[:, [0]] - 100
            mask = mask[:, None, :]
            raw_masks_r = raw_masks_r & mask


            masked_imgs = imgs[:].copy()
            masked_imgs[raw_masks_l] = 0
            masked_imgs[raw_masks_r] = 0
            # masked_imgs[raw_masks_l, 0] = 255
            # masked_imgs[raw_masks_r, 1] = 255
            raw_masks = raw_masks_l | raw_masks_r
            
            overlayed_imgs = line_on_hand(masked_imgs, raw_masks_r, "right")
            overlayed_imgs = line_on_hand(overlayed_imgs, raw_masks_l, "left")

        elif ee_poses.shape[-1] == 3:
            prompts_l = None
            prompts_r = cam_frame_to_cam_pixels(ee_poses, intrinsics)[:, :2]

            masked_imgs, raw_masks = self.get_hand_mask_batched(imgs, prompts_r)

            overlayed_imgs = line_on_hand(masked_imgs, raw_masks, "right")
        else:
            raise ValueError(f"Invalid shape for ee_poses: {ee_poses.shape}")
        
        #cv2 imsave the masked_img_l and masked_img_r, bgr to rgb as well
        if debug:
            breakpoint()
            for j in range(overlayed_imgs.shape[0]):
                overlayed_imgs[j] = cv2.cvtColor(overlayed_imgs[j], cv2.COLOR_BGR2RGB)
                overlayed_imgs[j] = draw_dot_on_frame(overlayed_imgs[j], prompts_l[[j]], palette="Set1")
                overlayed_imgs[j] = draw_dot_on_frame(overlayed_imgs[j], prompts_r[[j]], palette="Set2")
                cv2.imwrite(f"./overlays/overlayed_img_{j}.png", overlayed_imgs[j])
                cv2.imwrite(f"./overlays/masked_img_{j}.png", cv2.cvtColor(masked_imgs[j], cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./overlays/mask_{j}.png", raw_masks[j].astype(np.uint8) * 255)
        
        return overlayed_imgs, masked_imgs, raw_masks

    def get_hand_mask_batched(self, images, pos_prompts, neg_prompts=None):
        """
            images: tensor (B, H, W, 3)
            pos_prompts: tensor (B, 2)

            returns: raw_masks
        """
        masked_imgs = np.zeros_like(images)
        raw_masks = np.zeros((images.shape[0], 480, 640)).astype(bool)

        for k in range(images.shape[0]):
            img = images[k]

            if neg_prompts is not None:
                output = self.get_hand_mask(img, pos_prompts[[k]], neg_prompt=neg_prompts[[k]])
            else:
                output = self.get_hand_mask(img, pos_prompts[[k]])
            if output is None:
                continue
            else:
                masked_img, raw_mask = output

            raw_masks[k] = raw_mask
            masked_imgs[k] = masked_img

        return masked_imgs, raw_masks

    def get_hand_mask(self, img, pos_prompt, neg_prompt=None):
        """
        Image:
        Pos_prompt: (2) array containing x, y coordinates of the point
        Neg_prompt: (2) array containing x, y coordinates of the point
        
        Returns:
        Masked image, mask, score, logits
        """
        input_point = pos_prompt
        if input_point[0, 0] > 640 or input_point[0, 1] > 480 or input_point[0, 0] < 0 or input_point[0, 1] < 0:
            masked_img = img
            return None
        input_label = np.array([1])
        if neg_prompt is not None:
            input_point = np.concatenate([input_point, neg_prompt], axis=0)
            input_label = np.array([1, 0])

        masked_img, masks, scores, logits = self.get_mask(img.copy(), input_point, input_label)

        raw_mask = masks[0].astype(bool)
        return masked_img, raw_mask



    def get_mask(self, image, points, label):
        """
        Image:
        Points: (N, 2) array containing x, y coordinates of the points
        Label: (N) array containing 0 or 1 specifying negative or positive prompts
        
        Returns:
        Masked image, masks, scores, logits
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        masked_img = image
        masked_img[masks[0] == 1] = 0

        return masked_img, masks, scores, logits
    



    def project_single_joint_position_to_image(self, qpos, extrinsics, intrinsics, arm="right"):
        joint_pos = self.fk.chain.forward_kinematics(qpos, end_only=False)
        fk_positions = joint_pos['vx300s/ee_gripper_link'].get_matrix()[:, :3, 3]
        wrist_positions = joint_pos['vx300s/wrist_link'].get_matrix()[:, :3, 3]
        elbow_positions = joint_pos['vx300s/upper_forearm_link'].get_matrix()[:, :3, 3]
        arm_positions = joint_pos['vx300s/ee_arm_link'].get_matrix()[:, :3, 3]
        lower_forearm_positions = joint_pos['vx300s/lower_forearm_link'].get_matrix()[:, :3, 3]


        fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)[:, :3]
        wrist_positions = ee_pose_to_cam_frame(wrist_positions, extrinsics)[:, :3]
        elbow_positions = ee_pose_to_cam_frame(elbow_positions, extrinsics)[:, :3]
        arm_positions = ee_pose_to_cam_frame(arm_positions, extrinsics)[:, :3]
        lower_forearm_positions = ee_pose_to_cam_frame(lower_forearm_positions, extrinsics)[:, :3]

        px_val_gripper = cam_frame_to_cam_pixels(fk_positions, intrinsics)[:, :2]
        px_val_wrist = cam_frame_to_cam_pixels(wrist_positions, intrinsics)[:, :2]
        px_val_elbow = cam_frame_to_cam_pixels(elbow_positions, intrinsics)[:, :2]
        px_val_arm = cam_frame_to_cam_pixels(arm_positions, intrinsics)[:, :2]
        px_val_lower_forearm = cam_frame_to_cam_pixels(lower_forearm_positions, intrinsics)[:, :2]

        if arm == "right":
            px_dict = {
                "px_val_gripper_right": px_val_gripper,
                "px_val_wrist_right": px_val_wrist,
                "px_val_elbow_right": px_val_elbow,
                "px_val_arm_right": px_val_arm,
                "px_val_lower_forearm_right": px_val_lower_forearm,
            }
        elif arm == "left":
            px_dict = {
                "px_val_gripper_left": px_val_gripper,
                "px_val_wrist_left": px_val_wrist,
                "px_val_elbow_left": px_val_elbow,
                "px_val_arm_left": px_val_arm,
                "px_val_lower_forearm_left": px_val_lower_forearm,
            }
        else:
            raise ValueError("Arm must be either 'right' or 'left'")
        return px_dict

    def project_joint_positions_to_image(self, qpos, extrinsics, intrinsics, arm="right"):
        if arm == "both":
            ## process left
            px_dict_left = self.project_single_joint_position_to_image(qpos[:, :6], extrinsics["left"], intrinsics, arm="left")
            ## process right
            px_dict_right = self.project_single_joint_position_to_image(qpos[:, 7:13], extrinsics["right"], intrinsics, arm="right")
            return {**px_dict_left, **px_dict_right}
        elif arm == "right":
            if qpos.shape[1] == 14:
                qpos = qpos[:, 7:]
            return self.project_single_joint_position_to_image(qpos[:, :-1], extrinsics["right"], intrinsics, arm="right")
        elif arm == "left":
            if qpos.shape[1] == 14:
                qpos = qpos[:, :7]
            return self.project_single_joint_position_to_image(qpos[:, :-1], extrinsics["left"], intrinsics, arm="left")
        else:
            raise ValueError("Arm must be 'both, 'right' or 'left'")

    def get_robot_mask_line_batched(self, images, px_dict, arm="right"):
        line_images = np.zeros_like(images)
        mask_images = np.zeros_like(images)

        if arm == "both":
            pt1_left = px_dict["px_val_wrist_left"]
            pt2_left = px_dict["px_val_gripper_left"]
            pt3_left = px_dict["px_val_arm_left"] #(pt1_left + pt2_left)/2
            pt1_right = px_dict["px_val_wrist_right"]
            pt2_right = px_dict["px_val_gripper_right"]
            pt3_right = px_dict["px_val_arm_right"] #(pt1_right + pt2_right)/2
        elif arm == "left":
            pt1_left = px_dict["px_val_wrist_left"]
            pt2_left = px_dict["px_val_gripper_left"]
            pt3_left =  px_dict["px_val_arm_left"] #(pt1_left + pt2_left)/2
        elif arm == "right":
            pt1_right = px_dict["px_val_wrist_right"]
            pt2_right = px_dict["px_val_gripper_right"]
            pt3_right =  px_dict["px_val_arm_right"] #(pt1_right + pt2_right)/2


        for i,image in enumerate(images[:]):
            # get the point between px_val_lower_forearm
            # if i == 100:
            #     break
            masked_img = image.copy()

            ## init arrays
            input_point_left = np.array([])
            input_label_left = np.array([])
            input_point_right = np.array([])
            input_label_right = np.array([])

            ## Get Valid Points
            if arm == "both" or arm == "left":
            ## process left
                left1, left2, left3 = pt1_left[[i]], pt2_left[[i]], pt3_left[[i]]
                input_point_left, input_label_left = get_valid_points((left1, left2, left3), line_images[0].shape)
            if arm == "both" or arm == "right":
                ## process right
                right1, right2, right3 = pt1_right[[i]], pt2_right[[i]], pt3_right[[i]]
                input_point_right, input_label_right = get_valid_points((right1, right2, right3), line_images[0].shape)

            # print("i", i)
            # print("left", left1, left2, left3)
            # print("right", right1, right2, right3)

            ## Set Input Points
            if input_point_left.size == 0 and input_point_right.size > 0:
                input_point = input_point_right
                input_label = input_label_right
            elif input_point_right.size == 0 and input_point_left.size > 0:
                input_point = input_point_left
                input_label = input_label_left
            elif  input_point_right.size > 0 and input_point_left.size > 0:       
                input_point = np.concatenate([input_point_left, input_point_right], axis=0)
                input_label = np.concatenate([input_label_left, input_label_right], axis=0)
            else:
                mask_images[i] = masked_img
                line_images[i] = masked_img.copy()
                continue
            
            if input_point_left.size > 0:
                masked_img, masks, scores, logits = self.get_mask(masked_img, input_point_left, input_label_left)
            if input_point_right.size > 0:
                masked_img, masks, scores, logits = self.get_mask(masked_img, input_point_right, input_label_right)
            # masked_img, masks, scores, logits = self.get_mask(image, input_point, input_label)

            line_img = masked_img.copy()

            if arm == "both" or arm == "left":
                ## draw left line
                line_img = cv2.line(
                    line_img,
                    (int(px_dict["px_val_gripper_left"][i, 0]), int(px_dict["px_val_gripper_left"][i, 1])),
                    (int(px_dict["px_val_elbow_left"][i, 0]), int(px_dict["px_val_elbow_left"][i, 1])),
                    color=(255, 0, 0),
                    thickness=25
                )
            if arm == "both" or arm == "right":
                ## draw right line
                line_img = cv2.line(
                    line_img,
                    (int(px_dict["px_val_gripper_right"][i, 0]), int(px_dict["px_val_gripper_right"][i, 1])),
                    (int(px_dict["px_val_elbow_right"][i, 0]), int(px_dict["px_val_elbow_right"][i, 1])),
                    color=(255, 0, 0),
                    thickness=25
                )


            mask_images[i] = masked_img
            line_images[i] = line_img

            # masked_img = draw_dot_on_frame(line_img, input_point, palette="tab10")
            # breakpoint()
            
            # for pt in input_point:
            #     image = cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            #     masked_img = cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            #     line_img = cv2.circle(line_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # # print("WRITING")
            # cv2.imwrite(f"/nethome/dpatel756/flash/egoPlay_unified/EgoPlay/mimicplay/scripts/masking/overlays/masked_img_{i}.png", cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))


        return mask_images, line_images         