# -*- coding: utf-8 -*-
"""
@Description :   Create URDF dataset (index, path, object name, size, fingerprint, etc.)
@Author      :   Yan Ding 
@Time        :   2023/12/30 20:43:44
"""

import os
from utils import *
import pandas as pd

# ----------------------------------------------------------------
# Get path
# ----------------------------------------------------------------
utils_path = os.path.join(os.getcwd(), 'utils.py')
folder_path = os.path.dirname(utils_path)

# ----------------------------------------------------------------
# Initialize pybullet GUI
# ----------------------------------------------------------------
demo = Client()
demo.enable_vertical_view(5.0, [1.9, 0, -4], -90, -65)

# ----------------------------------------------------------------
# Load urdf dataset
# ----------------------------------------------------------------
file_path = folder_path + '/' + 'urdf_dataset.xlsx'
df = pd.read_excel(file_path)
indexs = df['index']
subfolder_names = df['subfolder_name']
relative_paths = df['relative_path']
euler_xs = df['euler_x']
euler_ys = df['euler_y']
euler_zs = df['euler_z']

# ----------------------------------------------------------------
# Load i-th object
# ----------------------------------------------------------------

for index in range(0, len(subfolder_names)):
    print(f'object index:{indexs[index]}')
    object_path = folder_path + '/' + relative_paths[index]
    object_pose = [0.0, 0.0, 0.3]
    object_orientation = euler_to_quaternion(euler_xs[index], euler_ys[index], euler_zs[index])
    object_id = demo.add_object(object_path, object_pose, object_orientation)
    line_ids = demo.draw_aabb(object_id)
    demo.remove_aabb_lines(line_ids)
    [min_x, min_y, min_z], [max_x, max_y, max_z] = demo.get_bounding_box(object_id, print_output=True)
    print(f'size_x:{abs(max_x-min_x)} size_y:{abs(max_y-min_y)} size_z:{abs(max_z-min_z)}')
    color, depth, segment = demo.render_image()
    demo.wait(5)
    demo.remove_object(object_id)

demo.disconnect_pybullet()