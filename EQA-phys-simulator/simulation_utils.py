import os
import random
import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
import logging

ROBOT_TYPES = {
    'UR5': 'my_robot_urdf/ur5/ur5.urdf',
    'FR5': 'my_robot_urdf/fr5/fr5_robot.urdf',
    'CR5': 'my_robot_urdf/cr5/cr5_robot.urdf',
    "PANDA": 'my_robot_urdf/panda/panda_arm.urdf'
}

EE_LINK_NAMES = {
    'UR5': "wrist_3_link",
    'FR5': 'wrist3_Link',
    'CR5': 'Link6',
    'PANDA': 'panda_link7'
}

MAX_REACH = {
    'UR5': 0.91,
    'FR5': 0.92,
    'CR5': 0.91,
    'PANDA': 0.855
}


DH_PARAMETERS = {
    'UR5': [
        [0, -90, 0.08916],
        [-0.425, 0, 0],
        [-0.39225, 0, 0],
        [0, -90, 0.10915],
        [0, 90, 0.09465],
        [0, 0, 0.0823]
    ],
    'FR5': [
        [0, 90, 0.152],
        [-0.425, 0, 0],
        [-0.395, 0, 0],
        [0, 90, 0.102],
        [0, -90, 0.102],
        [0, 0, 0.10]
    ],
    'CR5': [
        [0, 0, 0.147],
        [0, 0, 90],
        [0.427, 0, 0],
        [0.357, -90, 0.141],
        [0, 90, -0.116],
        [0, -90, 0.105]
    ],
    'PANDA': [
        [0, 90, 0.333],
        [0, -90, 0],
        [0, 90, 0.316],
        [0.0825, 90, 0.384],
        [-0.0825, -90, 0],
        [0.088, 90, 0],
        [0.107, 0, 0.126]
    ]
}

URDF_DATASET_PATH = os.path.join(os.getcwd(), 'URDF_100models', 'urdf_dataset.xlsx')


class Simulation:
    def __init__(self, robot_type='UR5', use_gui=False):
        self.robot_type = robot_type
        self.use_gui = use_gui
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.table_id = None

    def connect(self):
        logging.info("Connecting to PyBullet.")
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.load_plane()
        self.load_table()
        self.load_robot()

    def load_plane(self):
        logging.info("Loading the ground plane.")
        self.plane_id = p.loadURDF("plane.urdf")

    def load_table(self, table_position=[0.8, 0, 0.2], table_size=[1.2, 1.2, 0.02]):
        logging.info("Loading the table.")
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size / 2 for size in table_size])
        rand_color = [random.randint(0, 255) / 255.0 for _ in range(3)] + [1]
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[size / 2 for size in table_size], rgbaColor=rand_color
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=table_position
        )

    def load_robot(self):
        logging.info(f"Loading robot type: {self.robot_type}")
        if self.robot_type not in ROBOT_TYPES:
            raise ValueError(f"Unsupported robot type. Available types: {list(ROBOT_TYPES.keys())}")

        urdf_path = ROBOT_TYPES[self.robot_type]
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file for robot '{self.robot_type}' not found at path: {urdf_path}")

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.2],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True
        )
        logging.info(f"Robot loaded successfully, ID: {self.robot_id}")

    def disconnect(self):
        if self.physics_client is not None:
            p.disconnect()
            logging.info("Disconnected from PyBullet.")


class URDFDataset:
    def __init__(self, dataset_path=URDF_DATASET_PATH):
        self.dataset_path = dataset_path
        self.df = pd.DataFrame()

    def load_dataset(self):
        if not os.path.exists(self.dataset_path):
            logging.error(f"URDF dataset file not found at path: {self.dataset_path}")
            return False

        try:
            self.df = pd.read_excel(self.dataset_path)
            logging.info("URDF dataset loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to load URDF dataset: {e}")
            return False


class Robot:
    def __init__(self, robot_id, robot_type):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.ee_link_name = EE_LINK_NAMES.get(robot_type)
        self.ee_link_index = self.get_end_effector_link_index()
        self.base_position, self.base_orientation = self.get_base_pose()

    def get_end_effector_link_index(self):
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == self.ee_link_name:
                logging.info(f"End effector link '{link_name}' found at joint index {i}.")
                return i
        raise ValueError(f"End effector link '{self.ee_link_name}' for robot '{self.robot_type}' not found.")

    def get_base_pose(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(base_pos), base_orn

    def get_base_orientation_matrix(self):
        return np.array(p.getMatrixFromQuaternion(self.base_orientation)).reshape(3, 3)

    def set_initial_pose(self, target_pos, target_orn):
        logging.info("Setting the initial pose of the robot.")
        joint_positions = p.calculateInverseKinematics(self.robot_id, self.ee_link_index, target_pos, target_orn)

        num_joints = p.getNumJoints(self.robot_id)
        movable_joints = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        if len(joint_positions) != len(movable_joints):
            raise ValueError(
                f"Number of joint positions returned by inverse kinematics ({len(joint_positions)}) "
                f"does not match the number of movable joints ({len(movable_joints)})."
            )

        for joint_index, position in zip(movable_joints, joint_positions):
            p.resetJointState(self.robot_id, joint_index, position)
            logging.debug(f"Joint {joint_index} set to position {position}")

        logging.info("Initial pose of the robot set successfully.")

    def can_reach_position(self, target_pos, tolerance=0.05):
        target_pos = np.array(target_pos)
        distance = np.linalg.norm(target_pos - self.base_position)
        max_reach = MAX_REACH.get(self.robot_type, 1.0)

        # logging.info(f"Target position distance: {distance:.3f} m, Maximum reachable distance: {max_reach:.3f} m")
        return distance <= (max_reach + tolerance)


class ObjectManager:
    def __init__(self, urdf_df, folder_path, min_distance=0.1):
        self.all_objects = urdf_df[
            ~urdf_df['subfolder_name'].str.contains('table')
        ]
        self.folder_path = folder_path
        self.object_ids = []
        self.selected_objects = []
        self.min_distance = min_distance
        self.preloaded_paths = self._preload_paths()

    def _preload_paths(self):
        paths = []
        for _, row in self.all_objects.iterrows():
            relative_path = row['relative_path']
            object_path = os.path.join(self.folder_path, relative_path)
            if os.path.exists(object_path):
                paths.append({
                    'name': row['subfolder_name'],
                    'path': object_path,
                    'euler': [row['euler_x'], row['euler_y'], row['euler_z']]
                })
        logging.info(f"Preloaded URDF paths for {len(paths)} objects.")
        return paths

    def load_objects(self, num_objects=10, placement_area=[0.3, 1.3], height=0.2, max_attempts=100):
        if not self.preloaded_paths:
            logging.warning("No preloaded object paths available, cannot load objects.")
            return self.object_ids, self.selected_objects

        placed_positions = []
        has_name = []
        for _ in range(num_objects):
            for attempt in range(max_attempts):
                obj = random.choice(self.preloaded_paths)
                object_name = obj['name']
                if object_name in has_name:
                    continue
                x = random.uniform(placement_area[0], placement_area[1])
                y = random.uniform(-0.5, 0.5)
                z = height
                new_position = np.array([x, y, z])

                if any(np.linalg.norm(new_position - pos) < self.min_distance for pos in placed_positions):
                    logging.debug(f"Attempted position {new_position} is too close to existing objects, regenerating.")
                    continue

                orn = p.getQuaternionFromEuler(obj['euler'])
                try:
                    obj_id = p.loadURDF(obj['path'], [x, y, z], orn, useFixedBase=False)
                    self.object_ids.append(obj_id)
                    self.selected_objects.append(object_name)
                    placed_positions.append(new_position)
                    has_name.append(object_name)
                    logging.info(f"Loaded object '{object_name}' at position {[x, y, z]} with rotation {obj['euler']}")
                    break
                except Exception as e:
                    logging.error(f"Failed to load object '{object_name}': {e}")
            else:
                logging.warning(f"Could not place object '{object_name}' within {max_attempts} attempts. Skipping this object.")

        return self.object_ids, self.selected_objects


class GraspInterface:
    def __init__(self, robot: Robot):
        self.robot = robot

    def grasp_objects_in_range(self, object_ids, x_range=(0.5, 1.0)):
        for obj_id in object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if x_range[0] <= pos[0] <= x_range[1]:
                logging.info(f"Object {obj_id} is within the grasping range.")
            else:
                logging.info(f"Object {obj_id} is outside the grasping range.")
    