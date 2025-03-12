import json
import cv2
import numpy as np
import open3d as o3d
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
import os
import time
from tqdm import tqdm
import argparse

# Global variables for multiprocessing
global_kdtree = None
global_pcb_point_cloud = None

def initializer(pcb_file_path):
    """
    Initialization function for the multiprocessing pool.
    Load the PCB point cloud and build a cKDTree in each worker process.
    """
    global global_kdtree
    global global_pcb_point_cloud
    pcb_point_cloud = DepthMapProcessor.load_pcb_point_cloud(pcb_file_path)
    global_pcb_point_cloud = pcb_point_cloud
    global_kdtree = cKDTree(pcb_point_cloud)

class DepthMapProcessor:
    def __init__(self, datas, visualize=False):
        """
        Initialize the utility class and load the configuration file.
        :param datas: List containing depth image information.
        :param visualize: Whether to enable visualization.
        """
        self.datas = datas
        self.visualize = visualize

    @staticmethod
    def load_pcb_point_cloud(pcb_file_path):
        """
        Read point cloud data.
        :param pcb_file_path: Path to the point cloud file.
        :return: Coordinate array of the point cloud.
        """
        pcd = o3d.io.read_point_cloud(pcb_file_path)
        return np.asarray(pcd.points)

    @staticmethod
    def transform_points(points, R, T):
        """
        Transform the point cloud from the camera coordinate system to the robot coordinate system
        using the rotation matrix R and translation vector T.
        :param points: Point cloud coordinates (N, 3).
        :param R: Rotation matrix (3, 3).
        :param T: Translation vector (3,).
        :return: Transformed point cloud coordinates (N, 3).
        """
        transformed_points = np.dot(R, points.T).T + T
        return transformed_points

    def initialize_kdtree(self):
        """
        Initialize the global KDTree and PCB point cloud in single-threaded mode.
        """
        pcb_file = self.datas[0]['pcb_file']
        print(f"Main process {os.getpid()} initializing KDTree with PCB file: {pcb_file}")
        pcb_point_cloud = self.load_pcb_point_cloud(pcb_file)
        global global_kdtree, global_pcb_point_cloud
        global_pcb_point_cloud = pcb_point_cloud
        global_kdtree = cKDTree(pcb_point_cloud)

    def process_image_single(self, data):
        """
        Process a single depth image, generate a pseudo-color depth map with a workspace mask,
        and visualize it if necessary.
        :param data: Dictionary containing depth image information.
        """
        depth_path = data['raw_path']
        rgb_path = data.get('rgb_path', None)
        depth_image_save_path = data['sp_map_save_path']
        rgb_image_save_path = data.get('rgb_map_save_path', 'media/rgb_map.jpg')

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Depth image {depth_path} not found.")
            return

        depth_image = depth_image.astype(np.float32)

        if rgb_path:
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_image is None:
                print(f"RGB image {rgb_path} not found.")
                rgb_image = None
            else:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = None

        R = np.array(data['rotation_matrix'])
        T = np.array(data['translation_vector'])

        fx, fy = data['camera_intrinsics']['fx'], data['camera_intrinsics']['fy']
        cx, cy = data['camera_intrinsics']['cx'], data['camera_intrinsics']['cy']

        h, w = depth_image.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = (cx - x) * depth_image / fx
        y = (cy - y) * depth_image / fy
        z = depth_image

        depth_points = np.stack((x, y, z), axis=-1).reshape(-1, 3) / 1000.0

        depth_points_transformed = self.transform_points(depth_points, R, T)

        if global_kdtree is None:
            print("KDTree not initialized.")
            return

        distances, _ = global_kdtree.query(depth_points_transformed)
        workspace_mask = (distances > data['workspace_radius']).reshape(h, w).astype(np.uint8)

        max_depth_value = min(3000, int(np.max(depth_image)))
        min_depth_value = max(0, int(np.min(depth_image)))
        depth_image_clipped = np.clip(depth_image, min_depth_value, max_depth_value)

        depth_normalized = (depth_image_clipped - min_depth_value) / (max_depth_value - min_depth_value)
        depth_normalized = np.clip(depth_normalized, 0, 1)

        depth_normalized_uint8 = (depth_normalized * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_normalized_uint8, cv2.COLORMAP_JET)

        alpha = 0.7
        gray_overlay = np.full_like(depth_colored, 128, dtype=np.uint8)

        depth_result = depth_colored.copy()
        depth_result[workspace_mask == 1] = cv2.addWeighted(
            depth_colored[workspace_mask == 1], 1 - alpha,
            gray_overlay[workspace_mask == 1], alpha, 0
        )

        mask_uint8 = (workspace_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_result, contours, -1, (0, 0, 0), thickness=1)

        cv2.imwrite(depth_image_save_path, depth_result)

        if rgb_image is not None:
            rgb_image_resized = cv2.resize(rgb_image, (w, h))

            workspace_mask_expanded = workspace_mask[..., np.newaxis]

            alpha_rgb = 0.7
            gray_overlay_rgb = np.full_like(rgb_image_resized, 128, dtype=np.uint8)

            rgb_result = rgb_image_resized.copy()
            rgb_result[workspace_mask == 1] = cv2.addWeighted(
                rgb_image_resized[workspace_mask == 1], 1 - alpha_rgb,
                gray_overlay_rgb[workspace_mask == 1], alpha_rgb, 0
            )

            rgb_masked_bgr = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_image_save_path, rgb_masked_bgr)
        else:
            print("No RGB image provided, skipping RGB processing.")

        if self.visualize:
            try:
                in_workspace_mask = distances < data['workspace_radius']
                in_workspace_points = depth_points_transformed[in_workspace_mask]

                if rgb_image is not None:
                    rgb_colors = rgb_image.reshape(-1, 3)[in_workspace_mask]
                    rgb_colors = rgb_colors / 255.0
                    all_rgb_colors = rgb_image.reshape(-1, 3)
                    all_rgb_colors = all_rgb_colors / 255.0
                else:
                    rgb_colors = np.ones_like(in_workspace_points) * np.array([0, 1, 0])

                workspace_pcd = o3d.geometry.PointCloud()
                workspace_pcd.points = o3d.utility.Vector3dVector(global_pcb_point_cloud)
                workspace_pcd.paint_uniform_color([1, 0, 0])

                all_depth_pcd = o3d.geometry.PointCloud()
                all_depth_pcd.points = o3d.utility.Vector3dVector(depth_points_transformed)
                all_depth_pcd.colors = o3d.utility.Vector3dVector(all_rgb_colors)

                depth_pcd = o3d.geometry.PointCloud()
                depth_pcd.points = o3d.utility.Vector3dVector(in_workspace_points)
                if rgb_image is not None:
                    depth_pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
                else:
                    depth_pcd.paint_uniform_color([0, 1, 0])

                robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.3, origin=[0, 0, 0]
                )

                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.3, origin=[0, 0, 0]
                )

                camera_transform = np.eye(4)
                camera_transform[:3, :3] = R
                camera_transform[:3, 3] = T

                camera_frame.transform(camera_transform)

                coordinate_frames = [robot_frame]

                o3d.visualization.draw_geometries(
                    [all_depth_pcd] + coordinate_frames,
                    window_name=f"Visualization: {depth_path}",
                    width=800,
                    height=600,
                    left=50,
                    top=50,
                    point_show_normal=False
                )

                o3d.visualization.draw_geometries(
                    [depth_pcd] + coordinate_frames,
                    window_name=f"Visualization: {depth_path}",
                    width=800,
                    height=600,
                    left=50,
                    top=50,
                    point_show_normal=False
                )

                o3d.visualization.draw_geometries(
                    [workspace_pcd, all_depth_pcd] + coordinate_frames,
                    window_name=f"Visualization: {depth_path}",
                    width=800,
                    height=600,
                    left=50,
                    top=50,
                    point_show_normal=False
                )
            except Exception as e:
                print(f"Error during visualization: {e}")

    def process_image(self, data):
        """
        Wrapper method for use in multiprocessing.
        """
        self.process_image_single(data)
        return True

    def process_all_images(self, workers=cpu_count()):
        """
        Process all depth images using multiprocessing.
        If visualization is enabled, process images one by one to avoid conflicts.
        """
        if self.visualize:
            self.initialize_kdtree()
            for data in tqdm(self.datas, desc="Processing Images"):
                self.process_image(data)
        else:
            unique_pcb_files = set(data['pcb_file'] for data in self.datas)
            if len(unique_pcb_files) > 1:
                print("Warning: Multiple PCB files detected. Unable to share KDTree. Consider using a single PCB file for better performance.")

            pcb_file = self.datas[0]['pcb_file']
            pool = Pool(processes=workers, initializer=initializer, initargs=(pcb_file,))
            t1 = time.time()
            with tqdm(total=len(self.datas), desc="Processing Images") as pbar:
                for _ in pool.imap_unordered(self.process_image, self.datas):
                    pbar.update()
            print(f"Processing time: {time.time() - t1:.2f} seconds")
            pool.close()
            pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get robot data from simulator")
    parser.add_argument("--robot", type=str, default="PANDA", help="UR5, FR5, CR5, PANDA")
    parser.add_argument("--visualize", type=bool, default=False, help="use ui")
    args = parser.parse_args()
    arm = args.robot
    visualize = args.visualize
    root_path = f'./val/{arm}'
    pcb_file = f'./arm_ws_pcb/{arm}_ws.pcd'

    all_datas = []

    json_files = sorted([x for x in os.listdir(root_path) if x.endswith('.json')])
    for json_file in tqdm(json_files, desc="Collecting Data"):
        if json_file.startswith('.'):
            continue
        rgb_file = json_file.replace("_data_", "_rgb_").replace(".json", ".jpg")
        depth_file = json_file.replace("_data_", "_depth_").replace(".json", ".png")
        sp_file = json_file.replace("_data_", "_sp_map_").replace(".json", ".jpg")
        rgb_map_file = json_file.replace("_data_", "_rgb_map_").replace(".json", ".jpg")
        rgb_ws_mask_file = json_file.replace("depth", "_rgb_ws_mask_").replace(".json", ".jpg")

        with open(os.path.join(root_path, json_file), 'r') as f:
            data = json.load(f)
            camera_intrinsics = data['camera_intrinsics']
            camera_to_base_rotation = data['camera_to_base_rotation']
            camera_to_base_translation = data['camera_to_base_translation']

        data_entry = {
            'raw_path': os.path.join(root_path, depth_file),
            'rgb_path': os.path.join(root_path, rgb_file),
            'sp_map_save_path': os.path.join(root_path, sp_file),
            'rgb_map_save_path': os.path.join(root_path, rgb_map_file),
            'rgb_ws_mask_save_path': os.path.join(root_path, rgb_ws_mask_file),
            'pcb_file': pcb_file,
            'rotation_matrix': camera_to_base_rotation,
            'translation_vector': camera_to_base_translation,
            'camera_intrinsics': camera_intrinsics,
            'workspace_radius': 0.05
        }

        all_datas.append(data_entry)

    if not all_datas:
        print("No valid data found to process.")
        exit(0)

    save_paths = []
    for data in all_datas[:1]:
        save_paths.extend([
            data.get('sp_map_save_path'),
            data.get('rgb_map_save_path'),
            data.get('rgb_ws_mask_save_path')
        ])

    for path in save_paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    processor = DepthMapProcessor(all_datas, visualize=visualize)
    processor.process_all_images(workers=32)
    print(f"sp map save at {root_path}")
    