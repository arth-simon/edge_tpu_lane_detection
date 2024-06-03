"""cvat_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import sys
import cv2
import os
import math
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

cnt_unique = 0

imagecnt = 0

CVAT_READER = True
class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cvat_dataset dataset."""

    VERSION = tfds.core.Version('1.4.5')
    RELEASE_NOTES = {
        '1.4.5': 'Potential fix for test set',
    }

    DRIVE = "C:" if os.name == "nt" else "/mnt/c"

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            # builder=self,

            description="CVat Lane Dataset",

            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(dtype=tf.float32, shape=(None, None, 3), encoding="bytes"),
                "label": tfds.features.Sequence(
                    tfds.features.Tensor(shape=(None, None), dtype=tf.float32, encoding="bytes")),
                # "h_samples": tfds.features.Sequence(tfds.features.Tensor(shape=(None,), dtype=tf.int32)),
                # "augmentation_index": tf.int32,
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = f"{self.DRIVE}/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/"
        return {
            'train': self._generate_examples(os.path.join(path, "train_set"),
                                             'train_set.json',
                                             [0.0, -15.0, 15.0, -30.0, 30.0]),
            'test': self._generate_examples(os.path.join(path, "test_set"), 'test_set.json', None),
        }

    def _generate_examples(self, path, label_data_name, augmentation_deg):
        """Yields examples."""
        # DONE(cvat_dataset): Yields (key, example) tuples from the dataset
        global cnt_unique
        with open(
                f"{self.DRIVE}/Users/inf21034/PycharmProjects/edge_tpu_lane_detection/add_ins/cvat_config2.json") as json_file:
            config = json.load(json_file)
        net_input_img_size = config["model_info"]["input_image_size"]
        x_anchors = config["model_info"]["x_anchors"]
        y_anchors = config["model_info"]["y_anchors"]
        max_lane_count = config["model_info"]["max_lane_count"]
        ground_img_size = config["model_info"]["input_image_size"]
        dates = list(config["perspective_info"].keys())
        perspective_infos = {}
        for date in dates:
            H_list, map_x_list, map_y_list = create_map(config, date, augmentation_deg=augmentation_deg)
            H_list = tf.constant(H_list)
            map_x_list = tf.constant(map_x_list)
            map_y_list = tf.constant(map_y_list)
            perspective_infos[date] = (H_list, map_x_list, map_y_list)

        for image_ary, lanes_x_vals, lanes_y_vals, refIdx, day_of_recording in _data_reader(path, label_data_name,
                                                                                              augmentation_deg):
            # Generate a unique key for each example

            # if "12" not in day_of_recording:
            #     continue
            # if "2023-10-02" not in day_of_recording and "test" not in label_data_name:
            #     continue
            key = cnt_unique
            cnt_unique += 1
            H_list, map_x_list, map_y_list = perspective_infos[day_of_recording]
            cutoffs = config["perspective_info"][day_of_recording]["cutoffs"]
            img, label = _map_projection_data_generator(image_ary,
                                                        lanes_x_vals,
                                                        lanes_y_vals,
                                                        net_input_img_size,
                                                        x_anchors,
                                                        y_anchors,
                                                        max_lane_count,
                                                        H_list[refIdx],
                                                        map_x_list[refIdx],
                                                        map_y_list[refIdx],
                                                        ground_img_size,
                                                        cutoffs
                                                        )
            if img is None or label is None:
                continue
            yield key, {
                "image": img,
                "label": label,
            }
        # for f in path.glob('*.jpeg'):
        #   yield 'key', {
        #       'image': f,
        #       'label': 'yes',
        #   }


def _data_reader(dataset_path,
                 label_data_name,
                 augmentation_deg):
    print("Load data from ", dataset_path, label_data_name)
    label_data_path = os.path.join(dataset_path, label_data_name)
    if not os.path.exists(label_data_path):
        print("Label file doesn't exist, path : ", label_data_path)
        sys.exit(0)

    count = 0

    # load data
    with open(label_data_path, 'r') as reader:
        for line in reader.readlines():
            raw_label = json.loads(line)
            image_path = os.path.join(str(dataset_path), raw_label["raw_file"])
            day_of_recording = raw_label["raw_file"][0:10]
            label_lanes = raw_label["lanes"]
            label_h_samples = raw_label["h_samples"]

            # read image
            with Image.open(image_path) as image:
                image_ary = np.asarray(image)

            # enable this for small dataset test
            # if count >=32:
            #     break
            count += 1

            if augmentation_deg is None:
                yield image_ary, label_lanes, label_h_samples, 0, day_of_recording
            else:
                for refIdx in range(len(augmentation_deg)):
                    yield image_ary, label_lanes, label_h_samples, refIdx, day_of_recording


def _data_reader_cvat(dataset_path: str,
                      label_data_name: str,
                      augmentation_deg: list[float]):
    print("Load data from ", dataset_path, label_data_name)
    label_data_path = os.path.join(dataset_path, label_data_name)
    if not os.path.exists(label_data_path):
        print("Label file doesn't exist, path : ", label_data_path)
        sys.exit(0)

    tree = ET.parse(label_data_path)
    root = tree.getroot()

    for image in root.findall('image'):
        y_coordinates = []
        x_coordinates = []
        day_of_recording = image.attrib['name'][0:10]
        image_path = str(os.path.join(str(dataset_path), str(image.attrib['name'])))
        with Image.open(image_path) as img:
            image_ary = np.asarray(img)
        for polyline in image.findall('polyline'):
            points = polyline.attrib['points'].split(";")
            for point in points:
                point = point.split(',')
                x_coordinates.append(int(point[0]))
                y_coordinates.append(int(point[1]))

        if augmentation_deg is None:
            yield image_ary, x_coordinates, y_coordinates, 0
        else:
            for refIdx in range(len(augmentation_deg)):
                yield image_ary, x_coordinates, y_coordinates, refIdx, day_of_recording


def create_map(config, day_of_recording: str,
               augmentation_deg=None):
    transformation_settings = config["perspective_info"].get(day_of_recording, None)
    if transformation_settings is None:
        print("No transformation settings found for day of recording: ", day_of_recording, file=sys.stderr)
        return None
    src_img_size = transformation_settings["image_size"]
    ground_size = config["model_info"]["input_image_size"]

    w, h = src_img_size
    gw, gh = ground_size

    # calc homography (TuSimple fake)
    imgP = [transformation_settings["image_p0"],
            transformation_settings["image_p1"],
            transformation_settings["image_p2"],
            transformation_settings["image_p3"]]
    groundP = [transformation_settings["ground_p0"],
               transformation_settings["ground_p1"],
               transformation_settings["ground_p2"],
               transformation_settings["ground_p3"]]
    ground_scale_width = config["model_info"]["ground_scale_width"]
    ground_scale_height = config["model_info"]["ground_scale_height"]

    # We only use one perspective matrix for image transform, therefore, all images
    # at dataset must have same size, or the perspective transormation may fail.
    # In default, we assume the camera image input size is 1280x720, so the following
    # step will resize the image point for size fitting.
    # for i in range(len(imgP)):
    #     imgP[i][0] *= w / 1280.0
    #     imgP[i][1] *= h / 720.0

    # Scale the ground points, we assume the camera position is center of perspectived image,
    # as shown at following codes :
    #     (perspectived image with ground_size)
    #     ################################
    #     #             +y               #
    #     #              ^               #
    #     #              |               #
    #     #              |               #
    #     #              |               #
    #     #          p0 --- p1           #
    #     #          |      |            #
    #     #          p3 --- p2           #
    #     #              |               #
    #     # -x ----------C------------+x #
    #     ################################
    #
    # for i in range(len(groundP)):
    #     groundP[i][0] = groundP[i][0] * gw / w  # ground_scale_width + gw / 2.0
    #     groundP[i][1] = groundP[i][1] * gh / h  # gh - groundP[i][1] * ground_scale_height

    list_H = []
    list_map_x = []
    list_map_y = []

    groud_center = tuple(np.average(groundP, axis=0))
    if augmentation_deg is None:
        augmentation_deg = [0.0]

    for deg in augmentation_deg:
        R = cv2.getRotationMatrix2D(groud_center, deg, 1.0)
        rotate_groupP = []
        for gp in groundP:
            pp = np.matmul(R, [[gp[0]], [gp[1]], [1.0]])
            rotate_groupP.append([pp[0], pp[1]])

        # H, _ = cv2.findHomography(np.float32(imgP), np.float32(rotate_groupP))
        H = cv2.getPerspectiveTransform(np.float32(imgP), np.float32(rotate_groupP))
        _, invH = cv2.invert(H)

        map_x = np.zeros((gh, gw), dtype=np.float32)
        map_y = np.zeros((gh, gw), dtype=np.float32)

        for gy in range(gh):
            for gx in range(gw):
                nx, ny, nz = np.matmul(invH, [[gx], [gy], [1.0]])
                nx /= nz
                ny /= nz
                if 0 <= nx < gw and 0 <= ny < gh:
                    map_x[gy][gx] = nx
                    map_y[gy][gx] = ny
                else:
                    map_x[gy][gx] = -1
                    map_y[gy][gx] = -1

        list_H.append(H)
        list_map_x.append(map_x)
        list_map_y.append(map_y)

    return list_H, list_map_x, list_map_y


def _map_projection_data_generator(src_image,
                                   lanes_x_vals,
                                   label_h_samples,
                                   net_input_img_size,
                                   x_anchors,
                                   y_anchors,
                                   max_lane_count,
                                   H,
                                   map_x,
                                   map_y,
                                   groundSize,
                                   cutoffs):
    # transform image by perspective matrix
    height, width = src_image.shape[:2]
    # if width != 1280 or height != 720:
    #     src_image = cv2.resize(src_image, (1280, 720))
    # src_img_cpy = src_image.copy()
    # for l in label_lanes:
    #     for sz in range(len(label_h_samples)):
    #         cv2.circle(src_img_cpy, (int(l[sz]), int(label_h_samples[sz])), 2, (0, 255, 0), -1)

    # bev_img = cv2.remap(src_image, np.array(map_x), np.array(map_y),
    #                  interpolation=cv2.INTER_NEAREST,
    #                  borderValue=(125, 125, 125))

    # Multiplying the transformation matrix with an identity matrix in numpy transforms the "Eager Tensor" to a numpy array
    # cv2.warpPerspective() requires a numpy array as input and will fail on an "Eager Tensor"
    bev_img = cv2.warpPerspective(src_image, np.matmul(H, np.eye(3, 3)), (1280, 960), flags=cv2.INTER_LINEAR, )

    l_c, r_c, u_c, d_c = cutoffs
    bev_img = bev_img[u_c:d_c, l_c:r_c]
    bev_img = cv2.resize(bev_img, (net_input_img_size[1], net_input_img_size[0]))

    #################################
    # t_height, t_width = bev_img.shape[:2]
    grid_size = 32
    # for x in range(0, t_width, t_width // grid_size):
    #     cv2.line(bev_img, (x, 0), (x, t_height), (255, 0, 0), 1)  # Blue lines
    #
    # for y in range(0, t_height, t_height // grid_size):
    #     cv2.line(bev_img, (0, y), (t_width, y), (255, 0, 0), 1)  # Blue lines

    #################################

    bev_imgf = np.float32(bev_img) * (1.0 / 255.0)
    # cv2.imwrite("test.jpg", bev_img)

    # create label for class
    class_list = {'background': [0, 1],
                  'lane_marking': [1, 0]}
    class_count = len(class_list)  # [background, road]

    # create label for slice id mapping from  ground x anchor, and y anchor
    #   [y anchors,
    #    x anchors,
    #    class count + x offset]
    #
    class_count = 2
    offset_dim = 1
    instance_label_dim = 1
    label = np.zeros((y_anchors, x_anchors, class_count + offset_dim + instance_label_dim), dtype=np.float32)
    label_exists = np.zeros((y_anchors, x_anchors, class_count + offset_dim), dtype=np.float32)
    class_idx = 0
    x_offset_idx = class_count
    instance_label_idx = class_count + offset_dim

    # init values
    label[:, :, class_idx:class_idx + class_count] = class_list['background']
    label[:, :, x_offset_idx] = 0.0001

    # transform "h_samples" & "lanes" to desired format
    anchor_scale_x = float(x_anchors) / float(groundSize[1])
    anchor_scale_y = float(y_anchors) / float(groundSize[0])

    # bool to indicate if any lane has been added. if no lane has been added, return none
    lane_added = False
    # calculate anchor offsets
    for laneIdx in range(min(len(lanes_x_vals), max_lane_count)):
        lane_x_vals = lanes_x_vals[laneIdx]

        prev_bev_x = None
        prev_bev_y = None
        prev_x_anchor = None
        prev_y_anchor = None
        for idx in range(len(lane_x_vals)):
            source_y = label_h_samples[idx]
            source_x = lane_x_vals[idx]

            if source_x < 0:
                continue

            # do perspective transform at source_x, source_y
            bev_x, bev_y, gz = np.matmul(H, [[source_x], [source_y], [1.0]])
            if gz == 0:
                continue

            # conver to anchor coordinate(grid)
            bev_x = int(bev_x / gz)
            bev_y = int(bev_y / gz)
            resized = _resize_transformed_labels((bev_x, bev_y),
                                                 (width, height),
                                                 (400, 400),
                                                 (net_input_img_size[0], net_input_img_size[1]),
                                                 cutoffs=cutoffs)
            if resized is None:
                continue
            bev_x, bev_y = resized
            cv2.circle(bev_img, (int(bev_x), int(bev_y)), 2, (0, 255, 0), -1)
            if bev_x < 0 or bev_y < 0 or bev_x >= (groundSize[1] - 1) or bev_y >= (groundSize[0] - 1):
                continue

            x_anchor: int = int(bev_x * anchor_scale_x)
            y_anchor: int = int(bev_y * anchor_scale_y)

            if x_anchor < 0 or y_anchor < 0 or x_anchor >= (x_anchors - 1) or y_anchor >= (y_anchors - 1):
                continue

            instance_label_value = (laneIdx + 1.0) * 50
            label[y_anchor][x_anchor][class_idx:class_idx + class_count] = class_list['lane_marking']

            # do line interpolation to padding label data for perspectived coordinate.
            if prev_bev_x is None:
                prev_bev_x = bev_x
                prev_bev_y = bev_y
                prev_x_anchor = x_anchor
                prev_y_anchor = y_anchor
            else:
                if abs(y_anchor - prev_y_anchor) <= 1:
                    # skip if the point is already added; might happen due to interpolation or if lanes intersect
                    if label_exists[y_anchor][x_anchor][x_offset_idx] > 0:
                        continue
                    offset = bev_x - (x_anchor / anchor_scale_x)
                    label[y_anchor][x_anchor][x_offset_idx] += math.log(offset + 0.0001)
                    label[y_anchor][x_anchor][instance_label_idx] = instance_label_value
                    label_exists[y_anchor][x_anchor][x_offset_idx] = 1
                    lane_added = True
                else:
                    previous_point = np.array([prev_bev_x, prev_bev_y])
                    current_point = np.array([bev_x, bev_y])
                    prev2curr_distance = float(np.linalg.norm(previous_point - current_point))

                    normalized_direction_vec = (previous_point - current_point) / prev2curr_distance

                    inter_len = min(max(int(abs(prev_bev_y - bev_y)), 1), 10)
                    for source_y in range(inter_len):
                        interpolated_point = (current_point + normalized_direction_vec *
                                              (float(source_y) / float(inter_len)) * prev2curr_distance)

                        x_anchor = np.int32(interpolated_point[0] * anchor_scale_x)
                        y_anchor = np.int32(interpolated_point[1] * anchor_scale_y)

                        if label_exists[y_anchor][x_anchor][x_offset_idx] > 0:
                            continue

                        offset = interpolated_point[0] - (x_anchor / anchor_scale_x)
                        label[y_anchor][x_anchor][x_offset_idx] += math.log(offset + 0.0001)
                        label[y_anchor][x_anchor][class_idx:class_idx + class_count] = class_list['lane_marking']
                        label[y_anchor][x_anchor][instance_label_idx] = instance_label_value
                        label_exists[y_anchor][x_anchor][x_offset_idx] = 1
                        lane_added = True

                prev_bev_x = bev_x
                prev_bev_y = bev_y
                prev_x_anchor = x_anchor
                prev_y_anchor = y_anchor
    # global imagecnt
    # with open(f"images/outpt_imgs/{imagecnt:03d}.jpg", "wb") as f:
    #     f.write(cv2.imencode('.jpg', bev_img)[1])
    # imagecnt += 1
    if not lane_added:
        return None, None
    return bev_imgf, label


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
