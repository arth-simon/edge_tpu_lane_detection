import os
import json
import time
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from typing import List, Optional


# --------------------------------------------------------------------------------------------------------------

class EdgeTPUInference:
    def __init__(self, base_path: str, model_config_path: str, debug: bool = False, post_process: bool = True):
        self.config = json.load(open(model_config_path, 'r'))
        self.debug = debug
        self.enable_post_process = post_process

        self.original_shape: Optional[List[int]] = None

        # Load the model onto the Edge TPU
        model_path = os.path.join(base_path, "models", self.config["model_info"]["tflite_model_name"])
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get index of inputs and outputs, Model input information
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1:3]
        self.roi = self.config["perspective_info"]["cutoffs"]

        _, _, _, self.max_instance_count = common.output_tensor(self.interpreter, 0).shape
        _, self.y_anchors, self.x_anchors, _ = common.output_tensor(self.interpreter, 0).shape

    def preprocess_image(self, image):
        """
            Preprocess the image (cutoff and resize)
            :param image: Image as numpy matrix to be preprocessed
        """
        # mach des hier gerne
        # cutoffs is list with 4 values: x1, y1, x2, y2
        if self.original_shape is None:
            self.original_shape = image.shape
        xl, xr, yu, yd = self.roi

        image = image[yu:yd, xl:xr]

        image = cv2.resize(image, self.input_size)
        # Normalize or other preprocessing steps if required
        return image

    def predict(self, image: np.ndarray):
        image_prep = self.preprocess_image(image)

        start_time = time.time()
        # Preprocess and run inference
        common.set_input(self.interpreter, image_prep)
        self.interpreter.invoke()

        print("Inference took: ", time.time() - start_time)

        # Post-process and extract results
        instance = common.output_tensor(self.interpreter, 0)
        offsets = common.output_tensor(self.interpreter, 1)
        # anchor_axis = common.output_tensor(self.interpreter, 2)



        lanes = self.postprocess(instance, offsets, anchor_axis)
        # only keep the int(config["max_lane_count"]) lanes with most points
        max_lane_count = self.config["model_info"]["max_lane_count"]
        # get the indices of the x lanes with the most points
        lanes_indices_with_most_points = np.argsort([len(lane) for lane in lanes])[::-1][:max_lane_count]
        lanes = [lanes[i] for i in lanes_indices_with_most_points]

        # # check which lane is left, center or right by checking each first coordinate and sorting from low to high
        # lanes = sorted(lanes, key=lambda x: x[0][0] if len(x) > 0 else 0)

        print(lanes)
        return lanes

    def postprocess(self, instance, offsets, anchor_axis):
        COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        lanes = []
        if not self.enable_post_process:
            for instanceIdx in range(self.max_instance_count):
                current_lane = []
                for dy in range(self.y_anchors):
                    for dx in range(self.x_anchors):
                        instance_prob = instance[0, dy, dx, instanceIdx]
                        offset = offsets[0, dy, dx, 0]
                        gx = anchor_axis[0, dy, dx, 0] + offset
                        gy = anchor_axis[0, dy, dx, 1]
                        current_lane.append(self.prediction_to_coordinates((gx, gy)))
                lanes.append(current_lane)

        else:
            # Convert to numpy arrays if not already
            instance_np = np.array(instance)
            offsets_np = np.array(offsets)
            anchor_axis_np = np.array(anchor_axis)

            anchor_x_axis = anchor_axis_np[:, :, :, 0]
            anchor_y_axis = anchor_axis_np[:, :, :, 1]
            anchor_x_axis = np.expand_dims(anchor_x_axis, axis=-1)
            anchor_y_axis = np.expand_dims(anchor_y_axis, axis=-1)

            # create 0/1 mask by instance
            instance_mask = np.where(instance_np > 0.5, 1.0, 0.0)

            # Mux x anchors and offsets by instance
            anchor_x_axis += offsets_np
            anchor_x_axis *= instance_mask

            # Calculate mean of x axis
            sum_of_instance_row = np.sum(instance_mask, axis=2)
            sum_of_x_axis = np.sum(anchor_x_axis, axis=2)
            mean_of_x_axis = np.divide(sum_of_x_axis, sum_of_instance_row, out=np.zeros_like(sum_of_x_axis),
                                       where=sum_of_instance_row != 0)
            mean_of_x_axis = np.expand_dims(mean_of_x_axis, axis=2)
            mean_of_x_axis = np.tile(mean_of_x_axis, [1, 1, self.x_anchors, 1])

            # Create mask for threshold
            X_VARIANCE_THRESHOLD = 10.0
            diff_of_axis_x = np.abs(anchor_x_axis - mean_of_x_axis)
            mask_of_mean_offset = np.where(diff_of_axis_x < X_VARIANCE_THRESHOLD, 1.0, 0.0)

            # Apply threshold
            instance_mask *= mask_of_mean_offset
            anchor_x_axis *= mask_of_mean_offset
            anchor_y_axis *= mask_of_mean_offset

            # Average anchors by row
            sum_of_x_axis = np.sum(anchor_x_axis, axis=2)
            mean_of_x_axis = np.divide(sum_of_x_axis, sum_of_instance_row, out=np.zeros_like(sum_of_x_axis),
                                       where=sum_of_instance_row != 0)

            sum_of_y_axis = np.sum(anchor_y_axis, axis=2)
            mean_of_y_axis = np.divide(sum_of_y_axis, sum_of_instance_row, out=np.zeros_like(sum_of_y_axis),
                                       where=sum_of_instance_row != 0)

            # Rendering
            for instanceIdx in range(self.max_instance_count):
                current_lane = []
                for dy in range(self.y_anchors):
                    instance_prob = sum_of_instance_row[0, dy, instanceIdx]
                    gx = mean_of_x_axis[0, dy, instanceIdx]
                    gy = mean_of_y_axis[0, dy, instanceIdx]
                    if instance_prob > 0.5:
                        current_lane.append(self.prediction_to_coordinates((gx, gy)))
                lanes.append(current_lane)
        return lanes

    def prediction_to_coordinates(self, label):
        """
        Transform back labels and to the desired format
        """

        xl, xr, yu, yd = self.roi
        roi_width = xr - xl
        roi_height = yd - yu
        original_height, original_width, _ = self.original_shape
        if xl < label[0] < xr and yu < label[1] < yd:
            label = (label[0] - xl, label[1] - yu)
        label = (label[0] * original_width // roi_width,
                 label[1] * original_width // roi_height)
        return label


# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # run inference on single picture
    path_of_image = r"C:\Users\bensc\Downloads\50bilder\1702994123576254273.jpg"

    # Load the model
    model = EdgeTPUInference(base_path=r"C:\Users\bensc\Projects\edge_tpu_ld\edge_tpu_lane_detection",
                             model_config_path=r"C:\Users\bensc\Projects\edge_tpu_ld\edge_tpu_lane_detection\config\cvat_config.json")

    # Load the image
    image = cv2.imread(path_of_image)

    # Run the inference and measure the time
    start_time = time.time()
    result = model.predict(image)
    print("Inference took: ", time.time() - start_time)

    # display the result
    for lane in result:
        for coord in lane:
            cv2.circle(image, coord, 5, (255, 0, 0), -1)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
