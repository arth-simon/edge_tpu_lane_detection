import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image
import json
# from datasets.cvat_dataset.cvat_dataset_dataset_builder import _resize_transformed_labels
import tensorflow as tf
import sys

if "__name__" == "__main__":
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    config = json.load(open('cvat_config2.json'))
    dataset_path = '../../../source/IMG_ROOTS/1280x960_CVATROOT/images'
    n_config = json.load(open('cvat_config3.json'))


def interp_between_points(p1, p2):
    distance = int(np.linalg.norm(np.array(p1) - np.array(p2)))
    if distance < 1:
        yield p2
        return
    normalized_vector = (np.array(p2) - np.array(p1)) / distance
    # anchor_scale = (32/256)
    # p1_x_anchor_prev = int(p1[0] * anchor_scale) / anchor_scale
    # p1_y_anchor_prev = int(p1[1] * anchor_scale) / anchor_scale
    p = p1
    for i in range(distance):
        p += normalized_vector
        yield p
        # p_anchorized = (int(p[0] * anchor_scale) / anchor_scale, int(p[1] * anchor_scale) / anchor_scale)
        # if p_anchorized != (p1_x_anchor_prev, p1_y_anchor_prev):
        #     yield p[0], p_anchorized[1]
        #     p1_x_anchor_prev, p1_y_anchor_prev = p_anchorized
    return
    # x1, y1 = p1
    # x2, y2 = p2
    # x_points = np.linspace(x1, x2, num_points)
    # y_points = np.linspace(y1, y2, num_points)
    # return x_points, y_points


def get_cvat_lanes(dataset_name, dataset_path, augmentation_deg: list[int] | None = None):
    tree = ET.parse(os.path.join(dataset_path, dataset_name))
    root = tree.getroot()

    for image in root.findall('image'):
        lanes = []
        day_of_recording = image.attrib['name'][0:10]
        image_path = str(os.path.join(str(dataset_path), str(image.attrib['name'])))
        image_mat = cv2.imread(image_path)
        # with Image.open(image_path) as img:
        #     image_mat = np.asarray(img)
        if image_mat.shape[0] != 960 or image_mat.shape[1] != 1280:
            image_mat = cv2.resize(image_mat, (1280, 960))
        polylines = [p for p in image.findall('polyline')]
        if len(polylines) == 0:
            continue
        polylines = sorted(polylines, key=lambda x: x.attrib['label'])
        for polyline in polylines:
            y_coordinates = []
            x_coordinates = []
            points = polyline.attrib['points'].split(";")
            if len(points) < 2:
                continue

            prev_point = None
            for point in points:
                point = point.split(',')
                point = [float(point[0]), float(point[1])]
                if prev_point is not None:
                    for x, y in interp_between_points(prev_point, point):
                        x_coordinates.append(int(x))
                        y_coordinates.append(int(y))

                else:
                    x_coordinates.append(int(point[0]))
                    y_coordinates.append(int(point[1]))
                prev_point = point
            lanes.append([x_coordinates, y_coordinates])
        if augmentation_deg is None:
            yield lanes, image_mat, day_of_recording, 0
        else:
            for ref_idx in range(len(augmentation_deg)):
                yield lanes, image_mat, day_of_recording, ref_idx

        # yield lanes, image_mat, day_of_recording


def to_bev_lanes_and_labels(lanes, image, ml_input_size, cutoffs, p_matrix):
    # perspective_info = n_config['perspective_info']
    # src_points = perspective_info[day_of_recording]['src_points']
    # dst_points = perspective_info[day_of_recording]['dst_points']
    image_size = image.shape[:2]  # perspective_info[day_of_recording]['image_size']
    # cutoffs = perspective_info[day_of_recording]['cutoffs']
    # ml_input_size = n_config['model_info']['input_image_size']

    l_c, r_c, u_c, d_c = cutoffs
    p_matrix = np.matmul(p_matrix, np.eye(3, 3))

    # if p_matrix is None:
    #     p_matrix = cv2.getPerspectiveTransform(np.array(src_points, np.float32), np.array(dst_points, np.float32))
    tbev_image = cv2.warpPerspective(image, p_matrix, (image_size[0], image_size[1]))
    cut_bev_image = tbev_image[u_c:d_c, l_c:r_c]
    cut_bev_image = cv2.resize(cut_bev_image, (ml_input_size[0], ml_input_size[1]))
    tlanes_bev = []

    anchor_scale = (32 / ml_input_size[0])

    lane_grid = np.zeros((32, 32, 4), dtype=np.float32)
    lane_grid[:, :, 0:2] = [0, 1]
    lane_grid[:, :, 2] = 0.0001

    prev_dist_to_anchor = -1
    lane_instance = 1
    lane_added = False
    for x_coordinates, y_coordinates in lanes:
        prev_x_anchor = None
        prev_y_anchor = None
        lane_coordinates: list[list[int]] = []
        for x, y in zip(x_coordinates, y_coordinates):
            x, y, z = np.matmul(p_matrix, [[x], [y], [1.0]])
            if z == 0:
                continue
            x /= z
            y /= z
            # if x < 0 or x > image_size[0] or y < 0 or y > image_size[1]:
            #     continue
            x, y = x[0], y[0]
            xy: tuple[float, float] = _resize_transformed_labels((x, y),
                                                                 image_size,
                                                                 (r_c - l_c, d_c - u_c),
                                                                 ml_input_size, cutoffs)
            x, y = xy if xy is not None else (None, None)
            if x is None or y is None:
                continue
            x_anchor = int(x * anchor_scale)
            y_anchor = int(np.round(y * anchor_scale, 0))
            if x_anchor < 0 or x_anchor >= 32 or y_anchor < 0 or y_anchor >= 32:
                continue
            x_anchorized = x_anchor / anchor_scale
            y_anchorized = y_anchor / anchor_scale
            dist_to_anchor = np.sqrt((x_anchorized - x) ** 2 + (y_anchorized - y) ** 2)
            instance_val = np.float32(lane_instance * 50.0)
            # print(y)
            if prev_x_anchor is not None:
                if y_anchorized != prev_y_anchor:
                    lane_added = True
                    lane_grid[y_anchor, x_anchor, 0:2] = [1, 0]
                    lane_grid[y_anchor, x_anchor, 2] = math.log(abs(x - x_anchorized) + 0.0001)
                    lane_grid[y_anchor, x_anchor, 3] = instance_val
                    # lane_coordinates.append([int(x), int(y_anchorized)])
                    prev_x_anchor = x_anchorized
                    prev_y_anchor = y_anchorized
                    prev_dist_to_anchor = dist_to_anchor
                    continue
                elif dist_to_anchor < prev_dist_to_anchor:
                    lane_grid[int(prev_y_anchor * anchor_scale), int(prev_x_anchor * anchor_scale), 0:2] = [0, 1]
                    lane_grid[int(prev_y_anchor * anchor_scale), int(prev_x_anchor * anchor_scale), 2] = 0.0001
                    lane_grid[int(prev_y_anchor * anchor_scale), int(prev_x_anchor * anchor_scale), 3] = 0
                    lane_grid[y_anchor, x_anchor, 0:2] = [1, 0]
                    lane_grid[y_anchor, x_anchor, 2] = math.log(abs(x - x_anchorized) + 0.0001)
                    lane_grid[y_anchor, x_anchor, 3] = instance_val
                    # lane_coordinates[-1] = [int(x), int(y_anchorized)]
                    prev_dist_to_anchor = dist_to_anchor
                    lane_added = True
                    continue
                if abs(x_anchorized - prev_x_anchor) > 1 / anchor_scale:
                    lane_grid[y_anchor, x_anchor, 0:2] = [1, 0]
                    lane_grid[y_anchor, x_anchor, 2] = math.log(abs(x - x_anchorized) + 0.0001)
                    lane_grid[y_anchor, x_anchor, 3] = instance_val
                    # lane_coordinates.append([int(x_anchorized) + int(0.5/anchor_scale), int(y_anchorized)])
                    prev_x_anchor = x_anchorized
                    prev_y_anchor = y_anchorized
                    prev_dist_to_anchor = dist_to_anchor
                    lane_added = True

            else:
                prev_y_anchor = y_anchorized
                prev_x_anchor = x_anchorized
                prev_dist_to_anchor = dist_to_anchor
                lane_grid[y_anchor, x_anchor, 0:2] = [1, 0]
                lane_grid[y_anchor, x_anchor, 2] = math.log(x - x_anchorized + 0.0001)
                lane_added = True
                # lane_coordinates.append([int(x_anchorized), int(y_anchorized)])
        lane_instance += 1
        # lane_coordinates.append([int(x), int(y)])
        # tlanes_bev.append(lane_coordinates)
    # n_lanes = []
    # for ln in lanes:
    #     n_lane = [(x, y) for x, y in zip(ln[0], ln[1])]
    #     n_lanes.append(n_lane)
    # return image, n_lanes
    # lane_grid[:, :, 2] = np.float32(lane_grid[:, :, 2])
    cut_bev_image = np.float32(cut_bev_image) * (1.0 / 255.0)
    if np.count_nonzero(lane_grid[:, :, 0]) < 4 or not lane_added:
        return None, None
    if not np.any(lane_grid[:, :, 0]):
        return None, None
    return cut_bev_image, lane_grid


def _resize_transformed_labels(label, previous_shape, cutoff_shape, resize_shape, cutoffs):
    """
    Resize the transformed label to the desired shape
    :param label: label to resize
    """
    c_width, c_height = cutoff_shape[:2]
    r_width, r_height = resize_shape[:2]
    p_width, p_height = previous_shape[:2]
    x_l_cutoff, x_r_cutoff, y_u_cutoff, y_d_cutoff = cutoffs

    x_l_cutoff = x_l_cutoff if x_l_cutoff >= 0 else p_width + x_l_cutoff
    x_r_cutoff = x_r_cutoff if x_r_cutoff > 0 else p_width + x_r_cutoff
    y_u_cutoff = y_u_cutoff if y_u_cutoff >= 0 else p_height + y_u_cutoff
    y_d_cutoff = y_d_cutoff if y_d_cutoff > 0 else p_height + y_d_cutoff

    if x_l_cutoff < label[0] < x_r_cutoff and y_u_cutoff < label[1] < y_d_cutoff:
        label = (label[0] - x_l_cutoff, label[1] - y_u_cutoff)
    else:
        return None

    label = (label[0] * r_width // c_width,
             label[1] * r_height // c_height)
    return label


if __name__ == '__main__':
    anchor_scale = (32 / 256)
    for lanes, image, day_of_recording in get_cvat_lanes('../annotations_2.xml', dataset_path):
        bev_image, lanes_bev = to_bev_lanes_and_labels(lanes, image, day_of_recording)
        for x in range(32):
            for y in range(32):
                if lanes_bev[y, x, 0] == 0:
                    continue
                offset = lanes_bev[y, x, 2]
                color = COLORS[int(lanes_bev[y, x, 3] / 50) % 4]
                cv2.circle(bev_image, (int(x / anchor_scale + tf.exp(offset)), int(y / anchor_scale)), 2, color, -1)
        # paint 32x32 grid
        for i in range(0, bev_image.shape[1], int(bev_image.shape[1] / 32)):
            cv2.line(bev_image, (i, 0), (i, image.shape[0]), (125, 125, 125), 1)
        for i in range(0, bev_image.shape[0], int(bev_image.shape[0] / 32)):
            cv2.line(bev_image, (0, i), (image.shape[1], i), (125, 125, 125), 1)
        cv2.imshow('bev_image', bev_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
