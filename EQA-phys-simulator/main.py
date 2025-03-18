import os
import json
import random
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse
import pybullet as p

from simulation_utils import Simulation, URDFDataset, Robot, ObjectManager, DH_PARAMETERS
from camera_utils import Camera
from data_utils import compute_bounding_box

random.seed(11)
np.random.seed(11)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Run a single simulation
def run_simulation(index, selected_robot, urdf_dataset, folder_path, dh_parameters, image_save_dir):
    try:
        # Initialize the simulation
        simulation = Simulation(robot_type=selected_robot, use_gui=False)
        simulation.connect()

        # Initialize the robot
        robot = Robot(robot_id=simulation.robot_id, robot_type=selected_robot)
        target_position = [0., -0.3, 0.8]
        target_orientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        robot.set_initial_pose(target_position, target_orientation)

        # Initialize the object manager
        object_manager = ObjectManager(urdf_df=urdf_dataset.df, folder_path=folder_path)
        num_objects = 4
        object_ids, items_name = object_manager.load_objects(num_objects=num_objects, height=0.3)

        # Initialize the camera position
        camera_target = np.array([0.8, 0, 0.2])
        initial_camera_position = np.array([-0.6, 0.2, 1.2])
        direction = initial_camera_position - camera_target
        distance = np.linalg.norm(direction)
        target_z = camera_target[2]
        min_z = 1
        cos_phi_min = (min_z - target_z) / distance
        cos_phi_min = np.clip(cos_phi_min, -1, 1)
        phi_max = np.arccos(cos_phi_min)

        # Get a random position around the target
        def get_random_position_around_target(target, distance, phi_max):
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, phi_max)
            x = distance * np.sin(phi) * np.cos(theta)
            y = distance * np.sin(phi) * np.sin(theta)
            z = distance * np.cos(phi)
            new_position = target + np.array([x, y, z])
            return new_position.tolist()

        camera_position = get_random_position_around_target(camera_target, distance, phi_max)

        # Initialize the camera
        camera = Camera(
            position=camera_position,
            target=camera_target.tolist(),
            fov=60,
            aspect=1.0,
            image_width=640,
            image_height=480
        )

        # Get the base position and orientation of the robot
        base_position = robot.base_position
        base_orientation = robot.base_orientation
        R_base_world = robot.get_base_orientation_matrix()

        # Get the camera rotation and translation matrix
        R_cam_world = camera.rotation_matrix
        t_cam_world = camera.translation_vector

        # Calculate the rotation and translation from camera to base
        R_base_world_inv = R_base_world.T
        R_camera_to_base = R_base_world_inv @ R_cam_world
        t_camera_to_base = R_base_world_inv @ (np.array(t_cam_world) - base_position)

        camera_to_base_rotation = R_camera_to_base.tolist()
        camera_to_base_translation = t_camera_to_base.tolist()

        # Capture RGB and depth images
        rgb_image, depth_image = camera.capture_images()

        # Compute bounding boxes for objects
        bounding_boxes = {}
        for obj_id, item_name in zip(object_ids, items_name):
            bbox = compute_bounding_box(
                obj_id=obj_id,
                view_matrix=camera.view_matrix,
                projection_matrix=camera.projection_matrix,
                image_width=camera.image_width,
                image_height=camera.image_height
            )
            if bbox:
                bounding_boxes[item_name] = [round(x, 4) for x in bbox]
                logging.debug(f"Bounding box for object '{item_name}': {bbox}")
            else:
                logging.warning(f"Failed to compute bounding box for object '{item_name}'.")

        # Save images and data
        camera.save_images_and_data(
            save_dir=image_save_dir,
            rgb_image=rgb_image,
            depth_image=depth_image,
            camera_intrinsics=camera.intrinsics,
            camera_to_base_rotation=camera_to_base_rotation,
            camera_to_base_translation=camera_to_base_translation,
            dh_parameters=dh_parameters,
            models=items_name,
            selected_robot=selected_robot,
            index=index,
            bounding_boxes=bounding_boxes
        )

        # Check if the robot can reach the objects
        inrange = {'true': [], 'false': []}
        for obj_id, item_name in zip(object_ids, items_name):
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            place_position = [obj_pos[0], obj_pos[1], obj_pos[2]]
            can_reach = robot.can_reach_position(place_position, tolerance=0.05)
            if item_name[-1].isdigit():
                item_name = item_name[:-2]
            inrange[str(can_reach).lower()].append(item_name.replace("_", " "))

        # Generate the label
        label = {
            'image': f"{selected_robot}_rgb_{index}.jpg",
            'true': inrange['true'],
            'false': inrange['false']
        }

        # Disconnect the simulation
        simulation.disconnect()
        return label

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        simulation.disconnect()
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Get robot data from simulator")
    parser.add_argument("--dataset", type=str, default="val", help="train or val (train 5000, val 50)")
    parser.add_argument("--robot", type=str, default="PANDA", help="UR5, FR5, CR5, PANDA")

    args = parser.parse_args()
    dataset = args.dataset
    robot = args.robot
    
    # Create the save directory
    save_dir = os.path.join(dataset, robot)
    os.makedirs(save_dir, exist_ok=True)

    # Load the URDF dataset
    urdf_dataset = URDFDataset()
    if not urdf_dataset.load_dataset():
        logging.error("Unable to continue execution because the URDF dataset could not be loaded.")
        exit(1)
        
    folder_path = os.path.join(os.getcwd(), 'URDF_100models')
    dh_parameters = DH_PARAMETERS.get(robot, [])
    if not dh_parameters:
        logging.warning(f"DH parameters for robot '{robot}' are not defined.")

    inrange_labels = []
    # Run simulations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        if dataset == "train":
            number = 5000
        else:
            number = 50
        futures = [
            executor.submit(
                run_simulation,
                index,
                robot,
                urdf_dataset,
                folder_path,
                dh_parameters,
                save_dir
            )
            for index in range(number)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=number):
            result = future.result()
            if result:
                inrange_labels.append(result)

    # Save the labels to a JSON file
    labels_path = f'./{dataset}/{robot}_inrange.json'
    with open(labels_path, 'w') as f:
        json.dump(inrange_labels, f, indent=4)
    logging.info(f"All labels have been saved to {labels_path}")