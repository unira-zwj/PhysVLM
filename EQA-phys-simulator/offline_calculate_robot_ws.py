import numpy as np
import open3d as o3d
from itertools import product
import math
import tqdm
from multiprocessing import Pool, cpu_count
import random
import argparse


# DH parameters for different robot models
DH_PARAMETERS = {
    'UR5': [
        [0, -90, 0.08916, 0],
        [-0.425, 0, 0, 0],
        [-0.39225, 0, 0, 0],
        [0, -90, 0.10915, 0],
        [0, 90, 0.09465, 0],
        [0, 0, 0.0823, 0]
    ],
    'FR5': [
        [0, 90, 0.152, 0],
        [-0.425, 0, 0, 0],
        [-0.395, 0, 0, 0],
        [0, 90, 0.102, 0],
        [0, -90, 0.102, 0],
        [0, 0, 0.10, 0]
    ],
    'CR5': [
        [0, 0, 0.147, 0],
        [0, 0, 90, 0],
        [0.427, 0, 0, 0],
        [0.357, -90, 0.141, 0],
        [0, 90, -0.116, 0],
        [0, -90, 0.105, 0]
    ],
    'PANDA': [
        [0, 90, 0.333, 0],
        [0, -90, 0, 0],
        [0, 90, 0.316, 0],
        [0.0825, 90, 0.384, 0],
        [-0.0825, -90, 0, 0],
        [0.088, 90, 0, 0],
        [0.107, 0, 0.126, 0]
    ]
}

def degrees_to_radians(degrees):
    """
    Convert degrees to radians.
    """
    return degrees * math.pi / 180.0

def dh_transformation(a, alpha, d, theta):
    """
    Calculate a single DH transformation matrix.
    Args:
        a (float): Link length
        alpha (float): Link twist (radians)
        d (float): Link offset
        theta (float): Joint angle (radians)
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics_single(args):
    """
    Calculate the end position for a single joint angle combination.
    Args:
        args (tuple): (dh_params, joint_angles)
    Returns:
        np.ndarray: End effector position [x, y, z]
    """
    dh_params, joint_angles = args
    T = np.identity(4)
    for i, (params, angle_deg) in enumerate(zip(dh_params, joint_angles)):
        a, alpha_deg, d, theta_offset_deg = params
        alpha = degrees_to_radians(alpha_deg)
        theta_offset = degrees_to_radians(theta_offset_deg)
        theta = degrees_to_radians(angle_deg)
        theta_total = theta + theta_offset
        T_i = dh_transformation(a, alpha, d, theta_total)
        T = np.dot(T, T_i)
    position = T[:3, 3]
    return position

def sample_joint_space_uniform(joint_ranges, samples_per_joint):
    """
    Sample the joint space uniformly.
    Args:
        joint_ranges (list): Range of each joint in degrees, e.g., [(-180, 180), ...]
        samples_per_joint (list): Number of samples for each joint
    Returns:
        list: All possible joint angle combinations
    """
    axes = []
    for (min_angle, max_angle), samples in zip(joint_ranges, samples_per_joint):
        axes.append(np.linspace(min_angle, max_angle, samples))
    return list(product(*axes))

def sample_joint_space_random(joint_ranges, total_samples):
    """
    Sample the joint space randomly.
    Args:
        joint_ranges (list): Range of each joint in degrees, e.g., [(-180, 180), ...]
        total_samples (int): Total number of samples
    Returns:
        list: Randomly generated joint angle combinations
    """
    samples = []
    for _ in range(total_samples):
        angles = [random.uniform(min_angle, max_angle) for (min_angle, max_angle) in joint_ranges]
        samples.append(tuple(angles))
    return samples

def points_to_voxels(points, voxel_size):
    """
    Convert point cloud to voxel grid.
    Args:
        points (np.ndarray): Point cloud data, shape (N, 3)
        voxel_size (float): Voxel size
    Returns:
        open3d.geometry.VoxelGrid: Voxel grid
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid

def voxel_grid_to_numpy(voxel_grid):
    """
    Convert VoxelGrid to numpy array.
    Returns the center coordinates of the voxels.
    Args:
        voxel_grid (open3d.geometry.VoxelGrid): Voxel grid
    Returns:
        np.ndarray: Voxel center coordinates, shape (M, 3)
    """
    voxels = voxel_grid.get_voxels()
    voxel_centers = np.array([
        voxel.grid_index for voxel in voxels
    ]) * voxel_grid.voxel_size + voxel_grid.origin + 0.5 * voxel_grid.voxel_size
    return voxel_centers

def filter_voxels(voxel_grid, min_voxel_volume=0.001):
    """
    Filter out isolated voxels or small voxel groups.
    Args:
        voxel_grid (open3d.geometry.VoxelGrid): Voxel grid
        min_voxel_volume (float): Minimum voxel volume, voxels below this threshold will be removed
    Returns:
        open3d.geometry.VoxelGrid: Filtered voxel grid
    """
    voxel_centers = voxel_grid_to_numpy(voxel_grid)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = cl
    filtered_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filtered_pcd, voxel_size=voxel_grid.voxel_size)
    return filtered_voxel_grid

def simplify_voxel_grid(voxel_grid, target_voxel_count=10000):
    """
    Simplify the voxel grid structure, e.g., by adjusting the voxel size.
    Args:
        voxel_grid (open3d.geometry.VoxelGrid): Voxel grid
        target_voxel_count (int): Target number of voxels
    Returns:
        open3d.geometry.VoxelGrid: Simplified voxel grid
    """
    current_voxel_count = len(voxel_grid.get_voxels())
    if current_voxel_count <= target_voxel_count:
        return voxel_grid

    scaling_factor = (current_voxel_count / target_voxel_count) ** (1 / 3)
    new_voxel_size = voxel_grid.voxel_size * scaling_factor

    voxel_centers = voxel_grid_to_numpy(voxel_grid)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)

    simplified_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=new_voxel_size)
    return simplified_voxel_grid

def main(dh_params, joint_ranges, save_path):
    # Sampling strategy
    sampling_strategy = 'random'

    # Sampling parameters
    if sampling_strategy == 'uniform':
        samples_per_joint = [10, 10, 10, 10, 10, 10]
        joint_samples = sample_joint_space_uniform(joint_ranges, samples_per_joint)
        total_samples = len(joint_samples)
    elif sampling_strategy == 'random':
        total_samples = 1000000
        joint_samples = sample_joint_space_random(joint_ranges, total_samples)
    else:
        raise ValueError("Unsupported sampling strategy. Choose 'uniform' or 'random'.")

    print(f"Sampling Strategy: {sampling_strategy}")
    print(f"Total Samples: {len(joint_samples)}")

    # Parallel computation of forward kinematics
    print("Computing forward kinematics (using multiprocessing)...")
    num_processes = int(cpu_count() / 2) if cpu_count() > 1 else 1
    print(f"Number of processes: {num_processes}")

    with Pool(processes=num_processes) as pool:
        args = [(dh_params, angles) for angles in joint_samples]
        points = list(tqdm.tqdm(pool.imap(forward_kinematics_single, args), total=len(args), desc="Computing Forward Kinematics"))

    points = np.array(points)
    print(f"Total Reachable Points: {points.shape[0]}")

    # Convert to voxel grid
    voxel_size = 0.05
    print("Converting to voxel grid...")
    voxel_grid = points_to_voxels(points, voxel_size)

    # Save as PCD format
    pcd_filename = save_path
    print(f"Saving point cloud to {pcd_filename}...")

    voxel_numpy = voxel_grid_to_numpy(voxel_grid)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_numpy)
    o3d.io.write_point_cloud(pcd_filename, pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot workspace calculation script")
    parser.add_argument("--robot", type=str, default="UR5", help="Robot model, [UR5, FR5, CR5, PANDA]")

    args = parser.parse_args()
    robot = args.robot
    save_path = f"arm_ws_pcb/{robot}_ws.pcd"
    
    dh_parameters = DH_PARAMETERS.get(robot)
    if dh_parameters is None:
        raise ValueError(f"Unsupported robot model: {robot}")
    joint_num = len(dh_parameters)
    joint_ranges = [(-180, 180)] * joint_num

    save_path = f"arm_ws_pcb/{robot}_ws.pcd"
    main(dh_parameters, joint_ranges, save_path)
    print(f"robot {robot}'s reachable workspace is being calculated and save to {save_path}...")
    