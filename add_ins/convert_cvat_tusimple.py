import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
import cv2
import argparse
import json

# Path: scripts/convert_cvat_tusimple.py

"""
TODO: 
    - [ ] Randextrapolation
    - [ ] lane cut-off
"""


class CVATDataset:
    def __init__(self):
        self.images = []
        self.min_y = -1
        self.max_y = 0

    def add_image(self, image):
        self.images.append(image)

    def to_tusimple(self):
        for image in self.images:
            if not image.annotations:
                print("No annotations for image")
                continue
            min_y, max_y = image.get_minmax_y()
            if min_y < self.min_y or self.min_y < 0:
                self.min_y = min_y
            if max_y > self.max_y:
                self.max_y = max_y
        steps = (self.max_y - self.min_y) // 10
        y_samples = np.linspace(self.min_y, self.max_y, steps + 1, dtype=np.int32)
        for image in self.images:
            image.to_tusimple(y_samples)

    def write_to_json(self, out_path):
        with open(out_path, 'w') as f:
            for image in self.images:
                if not image.annotations:
                    continue
                f.write(str(image) + '\n')


class CVATImage:
    def __init__(self, image_id, image_width, image_height, image_path):
        self.image_id = image_id
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.min_y = -1
        self.max_y = 0
        self.annotations: list[CVATAnnotation] = []

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def get_minmax_y(self):
        for annotation in self.annotations:
            if annotation.min_y < self.min_y or self.min_y < 0:
                self.min_y = annotation.min_y
            if annotation.max_y > self.max_y:
                self.max_y = annotation.max_y
        # Auf 10 runden; für alle gelabelten Bilder gleich (wie in TUSimple)
        self.max_y = self.max_y + 10 - self.max_y % 10
        self.min_y = self.min_y - self.min_y % 10
        return self.min_y, self.max_y

    def to_tusimple(self, y_samples: np.ndarray):
        self.annotations = sorted(self.annotations, key=lambda x: x.label, reverse=True)
        for annotation in self.annotations:
            annotation.to_tusimple(y_samples)

    def __str__(self):
        json_img = {
            "lanes": [lane.x_val.tolist() for lane in self.annotations],
            "h_samples": self.annotations[0].y_samples.tolist(),
            "raw_file": self.image_path,
        }
        return json.dumps(json_img)

    def scale_image(self, image, scale):
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


class CVATAnnotation:
    def __init__(self, label, points_str, image_width, image_height):
        self.label = label  # "name of lane (left_lane, right_lane or similar); currently unused
        self.max_y: int = 0
        self.min_y: int = -1
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.points_str: str = points_str
        self.points: [(int, int)] = []
        self.y_samples = None
        self.x_val: np.ndarray | None = None
        self.extract_points()

    def scale_points(self, scale):
        for k, point in enumerate(self.points):
            point[0] = int(point[0] * scale / self.image_width)
            point[1] = int(point[1] * scale / self.image_height)
            self.points[k] = point

    def add_border_point(self, point):
        x, y = point
        # Distances to each edge
        smallest_distance = 0
        distances = {
            "top": y,
            "bottom": self.image_height - y,
            "left": x,
            "right": self.image_width - x
        }

        # Find the minimum distance and corresponding edge
        closest_edge = min(distances, key=distances.get)
        closest_distance = distances[closest_edge]
        if closest_distance == 0:
            return None
        elif closest_edge == "bottom":
            return x, self.image_height
        elif closest_edge == "left":
            return 0, y
        elif closest_edge == "right":
            return self.image_width, y
        return None

    def extract_points(self):
        points = self.points_str.split(';')
        for point in points:
            x, y = point.split(',')
            if int(float(y)) > self.max_y:
                self.max_y = int(float(y))
            if int(float(y)) < self.min_y or self.min_y < 0:
                self.min_y = int(float(y))
            self.points.append((int(float(x)), int(float(y))))

        border_point = self.add_border_point(self.points[0])
        if border_point:
            self.points = [border_point] + self.points

    def check_next_point(self, point, next_point, k):
        cnt = 2
        while abs(next_point[1] - point[1]) < 10:
            # if next_point[1] // 10 < point[1] // 10:
            #     next_point[1] = next_point[1] - next_point[1] % 10
            #     point[1] = point[1] - point[1] % 10
            # else:
            #     next_point[1] = next_point[1] - next_point[1] % 10
            #     point[1] = point[1] + 10 - point[1] % 10
            # left_idx, right_idx = sorted([np.searchsorted(y_samples, point[1]), np.searchsorted(y_samples, next_point[1])])
            # x_samples[left_idx:right_idx + 1] = np.array([point, next_point]).T[0]
            # continue
            self.points[k + cnt - 1] = None
            next_point = self.points[k + cnt] if k + cnt < len(self.points) else None
            cnt += 1
            if next_point is None:
                break
        return next_point

    @staticmethod
    def round_to_ten_and_sort(point: list[int], next_point: list[int]) -> list[list[int]]:
        """
        :param point: current point
        :param next_point: next point
        :return: left_y, right_y
        Rounds the y values of the points to the previous multiple of 10 and sorts them.
        """
        p, np = [point[0]], [next_point[0]]
        np.append(next_point[1] - next_point[1] % 10)
        p.append(point[1] - point[1] % 10)
        return sorted([p, np], key=lambda x: x[0])

    def generate_x_samples(self, left: tuple[int, int], right: tuple[int, int]) -> tuple[np.ndarray, int, int]:
        """
        :param left: left point
        :param right: right point (the property of being 'left' and 'right' is not used here)
        :return: x_samples, left_idx, right_idx
        Interpolates between left and right point and returns the x_samples, left_idx and right_idx.
        Creates a linear function x = y_1 * y + y_0 and evaluates it for all y in y_samples.
        """
        loc_y_samples = np.linspace(min(left[1], right[1]), max(left[1], right[1]), abs(right[1] - left[1]) // 10 + 1,
                                    dtype=np.int32)
        loc_y_samples = loc_y_samples[loc_y_samples >= self.min_y]
        y_1, y_0 = np.polyfit(np.array([left[1], right[1]]), np.array([left[0], right[0]]), 1)
        left_idx, right_idx = sorted([np.searchsorted(self.y_samples, loc_y_samples[0]),
                                      np.searchsorted(self.y_samples, loc_y_samples[-1])])
        return np.polyval([y_1, y_0], loc_y_samples), left_idx, right_idx

    def to_tusimple(self, y_samples: np.ndarray):
        """
        :param y_samples: y values to be sampled
        :return: None
        Converts the CVATAnnotation to a tusimple annotation using linear interpolation.
        It will always interpolate between two points and then round the y values to the previous multiple of 10.

        """
        self.cut_off_lane()
        x_samples = np.zeros_like(y_samples, dtype=np.int32)
        x_samples = x_samples - 2
        self.y_samples = y_samples
        for k, point in enumerate(self.points):
            if k == len(self.points) - 1:
                break
            if not point:
                continue
            point = [int(point[0]), int(point[1])]
            next_point = self.points[k + 1]
            next_point = [int(next_point[0]), int(next_point[1])]
            next_point = self.check_next_point(point, next_point, k)
            if next_point is None:
                break
            # linker und rechter punkt bestimmen+
            left, right = self.round_to_ten_and_sort(point, next_point)
            # x in Abhängigkeit von y
            min_x, max_x = sorted([point[0], next_point[0]])
            local_x_samples, left_idx, right_idx = self.generate_x_samples(left, right)
            # make local x samples integers
            x_samples[left_idx:right_idx + 1] = local_x_samples.astype(np.int32)
        self.x_val = x_samples

    def cut_off_lane(self):
        # 1. Finde den Index der Punkte die in y-werten zurückgehen
        # 2. Schneide dann die obere hälfte ab
        smallest_loc_y = self.max_y + 1
        local_points = []
        for k, point in enumerate(self.points):
            if point[1] < smallest_loc_y:
                smallest_loc_y = point[1]
                local_points.append(point)
        self.points = local_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True, help='The path to the CVAT annotation file')
    parser.add_argument('--image_path', required=False, help='The path to the images')
    args = parser.parse_args()
    annotation_path = args.annotation_path
    annotation_folder = os.path.dirname(annotation_path)
    image_path = args.image_path

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    # get all images with annotations
    dataset = CVATDataset()
    # image_list = []
    for image in root.findall('image'):
        image_id = image.attrib['id']
        image_width = image.attrib['width']
        image_height = image.attrib['height']
        image_path = image.attrib['name']
        cvat_image = CVATImage(image_id, image_width, image_height, image_path)
        for polyline in image.findall('polyline'):
            label = polyline.attrib['label']
            points = polyline.attrib['points']
            annotation = CVATAnnotation(label, points, image_width, image_height)
            cvat_image.add_annotation(annotation)
        dataset.add_image(cvat_image)
    dataset.to_tusimple()
    dataset.write_to_json(os.path.join(annotation_folder, 'tusimple1.json'))


if __name__ == "__main__":
    main()
