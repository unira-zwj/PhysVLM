import cv2
import numpy as np


def draw_bounding_boxes(image, bounding_boxes, color=(255, 0, 0), thickness=2):
    """
    在图像上绘制边界框。

    Args:
        image (numpy array): RGB图像.
        bounding_boxes (dict): 每个物体的边界框.
        color (tuple): 边界框颜色 (B, G, R).
        thickness (int): 边界框线条厚度.

    Returns:
        numpy array: 带有边界框的图像.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for name, bbox in bounding_boxes.items():
        x0, y0, x1, y1 = bbox
        top_left = (int(x0 * image_bgr.shape[1]), int(y0 * image_bgr.shape[0]))
        bottom_right = (int(x1 * image_bgr.shape[1]), int(y1 * image_bgr.shape[0]))
        cv2.rectangle(image_bgr, top_left, bottom_right, color, thickness)
        cv2.putText(image_bgr, name, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def project_point(p, view_matrix, projection_matrix, image_width, image_height):
    """
    将三维空间中的点投影到二维图像平面上。

    Args:
        p (list): 三维点的坐标 [x, y, z].
        view_matrix (list): 视图矩阵.
        projection_matrix (list): 投影矩阵.
        image_width (int): 图像宽度.
        image_height (int): 图像高度.

    Returns:
        list: 投影后的二维点坐标 [u, v]，如果点不在裁剪空间内则返回 None.
    """
    point = np.array([p[0], p[1], p[2], 1.0])
    view = np.array(view_matrix).reshape(4, 4).T
    proj = np.array(projection_matrix).reshape(4, 4).T
    clip_space = proj @ view @ point

    w = clip_space[3]
    if w == 0:
        return None

    ndc = clip_space[:3] / w

    if not (-1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1 and 0 <= ndc[2] <= 1):
        return None

    u = (ndc[0] + 1) * 0.5 * image_width
    v = (1 - (ndc[1] + 1) * 0.5) * image_height

    return [u, v]


def compute_bounding_box(obj_id, view_matrix, projection_matrix, image_width, image_height):
    """
    计算物体在图像平面上的二维边界框。

    Args:
        obj_id (int): 物体的ID.
        view_matrix (list): 视图矩阵.
        projection_matrix (list): 投影矩阵.
        image_width (int): 图像宽度.
        image_height (int): 图像高度.

    Returns:
        list: 归一化后的边界框坐标 [x0_norm, y0_norm, x1_norm, y1_norm]，如果没有可见点则返回 None.
    """
    import pybullet as p
    min_bb, max_bb = p.getAABB(obj_id)
    corners = np.array([
        [min_bb[0], min_bb[1], min_bb[2]],
        [max_bb[0], min_bb[1], min_bb[2]],
        [max_bb[0], max_bb[1], min_bb[2]],
        [min_bb[0], max_bb[1], min_bb[2]],
        [min_bb[0], min_bb[1], max_bb[2]],
        [max_bb[0], min_bb[1], max_bb[2]],
        [max_bb[0], max_bb[1], max_bb[2]],
        [min_bb[0], max_bb[1], max_bb[2]],
    ])

    # 添加一维以便进行批量矩阵乘法
    ones = np.ones((corners.shape[0], 1))
    corners_homogeneous = np.hstack([corners, ones])

    view = np.array(view_matrix).reshape(4, 4).T
    proj = np.array(projection_matrix).reshape(4, 4).T

    clip_space = (proj @ view @ corners_homogeneous.T).T

    w = clip_space[:, 3]
    valid = (w != 0)

    ndc = clip_space[:, :3] / w[:, np.newaxis]

    # 过滤出在裁剪空间内的点
    mask = (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) & \
           (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1) & \
           (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)

    visible = ndc[mask]
    if visible.size == 0:
        return None

    u = (visible[:, 0] + 1) * 0.5 * image_width
    v = (1 - (visible[:, 1] + 1) * 0.5) * image_height

    x0, y0 = np.min(u), np.min(v)
    x1, y1 = np.max(u), np.max(v)

    x0_norm = x0 / image_width
    y0_norm = y0 / image_height
    x1_norm = x1 / image_width
    y1_norm = y1 / image_height

    return [x0_norm, y0_norm, x1_norm, y1_norm]
