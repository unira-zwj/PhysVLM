import os
import math
import json
import cv2
import numpy as np
import logging

import pybullet as p  



class Camera:
    def __init__(self, position, target, up_vector=[0, 0, 1], fov=60, aspect=1.0, near=0.1, far=100,
                 image_width=640, image_height=480):
        self.position = np.array(position)
        self.target = np.array(target)
        self.up_vector = np.array(up_vector)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.image_width = image_width
        self.image_height = image_height

        self.view_matrix, self.projection_matrix = self.set_camera()
        self.intrinsics = self.compute_intrinsics()
        self.rotation_matrix, self.translation_vector = self.compute_pose()

    def set_camera(self):
        view_matrix = p.computeViewMatrix(self.position.tolist(), self.target.tolist(), self.up_vector.tolist())
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        return view_matrix, projection_matrix

    def compute_intrinsics(self):
        fov_rad = math.radians(self.fov)
        fx = (self.image_width / 2) / math.tan(fov_rad / 2)
        fy = fx / self.aspect
        cx = self.image_width / 2
        cy = self.image_height / 2

        intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        }
        logging.debug(f"相机内参: {intrinsics}")
        return intrinsics

    def compute_pose(self):
        forward = self.target - self.position
        forward /= np.linalg.norm(forward)

        right = np.cross(self.up_vector, forward)
        right /= np.linalg.norm(right)

        up_corrected = np.cross(forward, right)

        rotation_matrix = np.vstack([right, up_corrected, forward]).T
        translation_vector = self.position

        logging.debug(f"Camera Rotation Matrix:\n{rotation_matrix}")
        logging.debug(f"Camera Translation Vector:\n{translation_vector}")

        return rotation_matrix, translation_vector.tolist()

    def capture_images(self):
        img = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        width, height, rgba, depth, _ = img

        rgb_image = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)

        depth_buffer = np.reshape(depth, [height, width])
        depth_image = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)

        return rgb_image, depth_image

    def save_images_and_data(self, save_dir, rgb_image, depth_image, camera_intrinsics,
                             camera_to_base_rotation, camera_to_base_translation,
                             dh_parameters, models, selected_robot, index, bounding_boxes):
        os.makedirs(save_dir, exist_ok=True)
        rgb_path = os.path.join(save_dir, f"{selected_robot}_rgb_{index}.jpg")
        depth_path = os.path.join(save_dir, f"{selected_robot}_depth_{index}.png")
        data_path = os.path.join(save_dir, f"{selected_robot}_data_{index}.json")

        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(depth_path, (depth_image * 1000).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 3])

        data = {
            "camera_intrinsics": camera_intrinsics,
            "camera_to_base_rotation": camera_to_base_rotation,
            "camera_to_base_translation": camera_to_base_translation,
            "dh_parameters": dh_parameters,
            "models": models,
            "bounding_boxes": bounding_boxes
        }

        with open(data_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.debug(f"data saved to{data_path}")
