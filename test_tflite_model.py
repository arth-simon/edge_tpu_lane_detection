import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import cv2
import datasets
import math
import datasets.cvat_dataset
import tensorflow_datasets as tfds
from datasets import TusimpleLane
from eval import LaneDetectionEval
from tqdm import tqdm
import pickle
#from pycoral.utils import edgetpu
import atexit
import time

VISUALIZE = False

def del_interpreter(interpreter):
    del interpreter
    return


def augment_image(image_label, seed):
    image, label = image_label
    image = tf.image.stateless_random_brightness(image, 0.15, seed=seed)
    image = tf.image.stateless_random_contrast(image, 0.2, 0.85, seed=seed)
    # image = tf.image.random_flip_left_right(image)
    return image, label


# --------------------------------------------------------------------------------------------------------------
def tflite_image_test(tflite_model_quant_file,
                      dataset,
                      with_post_process=False,
                      mtx=None):
    # load model from saved model
    # interpreter = pycoral.utils.edgetpu.make_interpreter(tflite_model_quant_file)
    # interpreter = edgetpu.make_interpreter(tflite_model_quant_file)
    # atexit.register(del_interpreter, interpreter)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    #                                   experimental_delegates=[tf.lite.experimental.load_delegate(
    #                                       "edgetpu.dll")]
    #                                   )
    interpreter.allocate_tensors()
    rng = tf.random.Generator.from_seed(time.time(), alg='philox')

    def f(image, label):
        seed = rng.make_seeds(1)[:, 0]
        image, label = augment_image((image, label), seed)
        return image, label

    # dataset = dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # get index of inputs and outputs
    input_index = interpreter.get_input_details()[0]
    output_index_instance = interpreter.get_output_details()[0]
    output_index_offsets = interpreter.get_output_details()[1]
    # output_index_anchor_axis = interpreter.get_output_details()[2]

    # get part of data from output tensor
    _, _, _, max_instance_count = output_index_instance['shape']
    # _, y_anchors, x_anchors, _ = output_index_anchor_axis['shape']
    y_anchors, x_anchors = 32, 32

    files = []
    with open("../../source/IMG_ROOTS/1280x960_ROSBAGS/tusimple.json", "r") as inf:
        files = [json.loads(i)["raw_file"] for i in inf if '15-35-18' in i]

    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    total_correct, total_predicted, total_ground_truth = 0, 0, 0
    precision_by_frame = []
    recall_by_frame = []
    f1_by_frame = []

    anchor_axis = pickle.load(open("anchor_axis.pkl", "rb"))

    for k, elem in tqdm(enumerate(dataset), desc="evaluating...",
                        total=len(dataset)):  # tqdm(enumerate(files), desc="creating images", total=len(files)):
        test_img = elem[0]  # np.zeros((1, 256, 256, 3), dtype=np.uint8)
        # tmp_img = cv2.imread("../../source/IMG_ROOTS/1280x960_ROSBAGS/images/" + elem) # elem[0]
        test_label = elem[1]  # np.zeros((1, 32, 32, 1), dtype=np.float32)
        # t_label = test_label[0]
        # if mtx is not None:
        #     tmp_img = cv2.warpPerspective(tmp_img, mtx, (1280, 960))

        # l_c, r_c, u_c, d_c = [430, 870, 440, 880]
        # tmp_img = tmp_img[u_c:d_c, l_c:r_c]
        # test_img[0] = cv2.resize(tmp_img, (256, 256))

        if input_index['dtype'] == np.uint8:
            test_img = np.uint8(test_img * 255)
        # inference
        interpreter.set_tensor(input_index["index"], test_img)
        interpreter.invoke()

        instance = interpreter.get_tensor(output_index_instance["index"])
        offsets = interpreter.get_tensor(output_index_offsets["index"])
        # anchor_axis = interpreter.get_tensor(output_index_anchor_axis["index"])

        #print(f"instance: {instance.shape}, offsets: {offsets.shape}, anchor_axis: {anchor_axis.shape}")

        correct, predicted, ground_truth = LaneDetectionEval.evaluate_predictions(
            (instance, offsets, anchor_axis),
            test_label)
        total_correct += correct
        total_predicted += predicted
        total_ground_truth += ground_truth
        precision_by_frame.append(total_correct / total_predicted if total_predicted > 0 else 0)
        recall_by_frame.append(total_correct / total_ground_truth if total_ground_truth > 0 else 0)
        f1_by_frame.append(
            2 * precision_by_frame[-1] * recall_by_frame[-1] / (precision_by_frame[-1] + recall_by_frame[-1])
            if precision_by_frame[-1] + recall_by_frame[-1] > 0 else 0)

        if not VISUALIZE:
            continue
        # convert image to gray 
        main_img = cv2.cvtColor(test_img[0], cv2.COLOR_BGR2GRAY)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
        #print(np.median(offsets[0, :, :, 0]))
        # sys.exit(0)
        # post processing
        if not with_post_process:
            # rendering
            for instanceIdx in range(max_instance_count):
                for dy in range(y_anchors):
                    for dx in range(x_anchors):
                        instance_prob = instance[0, dy, dx, instanceIdx]
                        offset = offsets[0, dy, dx, 0]  # tf.exp(offsets[0, dy, dx, 0] * 0.008) #
                        gx = anchor_axis[0, dy, dx, 0] + offset
                        gy = anchor_axis[0, dy, dx, 1]

                        if instance_prob > 0.5:  # 0.5:
                            cv2.circle(main_img, (int(gx), int(gy)), 2, tuple(int(x) for x in COLORS[instanceIdx]))
        else:
            instance = tf.convert_to_tensor(instance)
            offsets = tf.convert_to_tensor(offsets)
            anchor_axis = tf.convert_to_tensor(anchor_axis)

            offsets = tf.exp(offsets)

            anchor_x_axis = anchor_axis[:, :, :, 0]
            anchor_y_axis = anchor_axis[:, :, :, 1]
            anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=-1)
            anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=-1)

            instance = tf.where(instance > 0.5, 1.0, 0.0)

            anchor_x_axis = tf.add(anchor_x_axis, offsets)
            anchor_x_axis = tf.multiply(anchor_x_axis, instance)

            sum_of_instance_row = tf.reduce_sum(instance, axis=2)
            sum_of_x_axis = tf.reduce_sum(anchor_x_axis, axis=2)
            mean_of_x_axis = tf.math.divide_no_nan(sum_of_x_axis, sum_of_instance_row)
            mean_of_x_axis = tf.expand_dims(mean_of_x_axis, axis=2)
            mean_of_x_axis = tf.tile(mean_of_x_axis, [1, 1, x_anchors, 1])

            X_VARIANCE_THRESHOLD = 10.0
            diff_of_axis_x = tf.abs(tf.subtract(anchor_x_axis, mean_of_x_axis))
            mask_of_mean_offset = tf.where(diff_of_axis_x < X_VARIANCE_THRESHOLD, 1.0, 0.0)

            instance = tf.multiply(mask_of_mean_offset, instance)
            anchor_x_axis = tf.multiply(mask_of_mean_offset, anchor_x_axis)
            anchor_y_axis = tf.multiply(mask_of_mean_offset, anchor_y_axis)

            sum_of_instance_row = tf.reduce_sum(instance, axis=2)
            sum_of_x_axis = tf.reduce_sum(anchor_x_axis, axis=2)
            mean_of_x_axis = tf.math.divide_no_nan(sum_of_x_axis, sum_of_instance_row)

            sum_of_y_axis = tf.reduce_sum(anchor_y_axis, axis=2)
            mean_of_y_axis = tf.math.divide_no_nan(sum_of_y_axis, sum_of_instance_row)

            lanes = [[] for _ in range(max_instance_count)]
            for instanceIdx in range(max_instance_count):
                instance_mask = sum_of_instance_row[0, :, instanceIdx] > 0.5
                gx_filtered = tf.boolean_mask(mean_of_x_axis[0, :, instanceIdx], instance_mask)
                gy_filtered = tf.boolean_mask(mean_of_y_axis[0, :, instanceIdx], instance_mask)

                if len(gx_filtered) > 0:
                    gx_list = gx_filtered.numpy()
                    gy_list = gy_filtered.numpy()

                    current_lane = np.stack((gx_list, gy_list), axis=-1)
                    lanes[instanceIdx] = [(coord) for coord in current_lane]
            for lanes, color in zip(lanes, COLORS):
                for i in range(len(lanes) - 1):
                    cv2.circle(main_img, (int(lanes[i][0]), int(lanes[i][1])), 2, color)

        # redering output image
        target_szie = (1000, 1000)
        main_img = cv2.resize(main_img, target_szie)

        inv_dx = 1.0 / float(x_anchors)
        inv_dy = 1.0 / float(y_anchors)
        for dy in range(y_anchors):
            for dx in range(x_anchors):
                px = (inv_dx * dx) * target_szie[0]
                py = (inv_dy * dy) * target_szie[1]
                # also draw points for ground truth data but in different strength
                cv2.line(main_img, (int(px), 0), (int(px), target_szie[1]), (125, 125, 125))
                cv2.line(main_img, (0, int(py)), (target_szie[0], int(py)), (125, 125, 125))
        for y in range(y_anchors):
            for x in range(x_anchors):
                if test_label[0, y, x, 0] == 0:
                    continue
                offset = test_label[0, y, x, 2]
                gx = anchor_axis[0, y, x, 0] + tf.exp(offset) - 0.0001
                gy = anchor_axis[0, y, x, 1]
                cv2.circle(main_img, (int(gx * target_szie[0] / 256), int(gy * target_szie[0] / 256)), 2, (125, 125, 255), 2, 2)

            # for i in range(len(label) - 1):
            #    cv2.circle(main_img, (int(label[i][0] + label[i][0][2]), int(label[i][1])), 2, (255, 255, 255))
        cv2.imshow("result", main_img)
        cv2.waitKey(0)
        # print(f"writing image: {k:03d}")
        # cv2.imwrite(f"images/outpt_imgs/frame{k:04d}.jpg", main_img)
        # plt.figure(figsize = (8,8))
        # plt.imshow(main_img)
        # plt.show()
    return precision_by_frame, recall_by_frame, f1_by_frame


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # read configs
    with open('add_ins/cvat_config2.json', 'r') as inf:
        config = json.load(inf)

    net_input_img_size = config["model_info"]["input_image_size"]
    x_anchors = config["model_info"]["x_anchors"]
    y_anchors = config["model_info"]["y_anchors"]
    max_lane_count = config["model_info"]["max_lane_count"]
    checkpoint_path = config["model_info"]["checkpoint_path"]
    tflite_model_name = config["model_info"]["tflite_model_name"]

    persp_info = config["perspective_info"]["2023-12-19"]

    src_points = [(persp_info[f"image_p{i}"][0], persp_info[f"image_p{i}"][1]) for i in range(4)]
    dst_points = [(persp_info[f"ground_p{i}"][0], persp_info[f"ground_p{i}"][1]) for i in range(4)]
    mtx = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32),
                                      np.array(dst_points, dtype=np.float32))

    if not os.path.exists(tflite_model_name):
        print("tlite model doesn't exist, please run \"generate_tflite_nidel.py\" first to convert tflite model.")
        sys.exit(0)

    # enable memory growth to prevent out of memory when training
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # set path of training data
    train_dataset_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/train_set"
    train_label_set = ["train_set.json"]
    """["label_data_0313.json",
                       "label_data_0531.json",
                       "label_data_0601.json"]"""
    test_dataset_path = "C:/Users/inf21034/source/IMG_ROOTS/1280x960_CVATROOT/test_set"
    test_label_set = ["test_set.json"]

    # valid_batches = datasets.TusimpleLane(test_dataset_path,
    #                                       test_label_set,
    #                                       config,
    #                                       augmentation=False).get_pipe()
    valid_batches = tfds.load('cvat_dataset:1.4.10', split='validation', shuffle_files=False, as_supervised=True)
    # TusimpleLane(test_dataset_path, test_label_set, config).get_pipe()
    # tfds.load('cvat_dataset', split='test', shuffle_files=True, as_supervised=True)
    valid_batches = valid_batches.batch(1)
    hyper_batch = "dyn_aug2"

    print("---------------------------------------------------")
    print("Load model as TF-Lite and test")
    print("---------------------------------------------------")
    precision, recall, f1 = tflite_image_test(tflite_model_name, valid_batches, True, mtx)
    print("---------------------------------------------------")
    print(f"{precision[-1]=}, {recall[-1]=}, {f1[-1]=}")
    sys.exit(0)
    # accs_t, accs_fp, accs_fn = tflite_image_test(tflite_model_name, valid_batches, True, mtx)
    plt.plot(range(len(precision)), precision, 'm--')
    plt.xlabel("frames")
    plt.ylabel("Precision")
    plt.title("Average Precision of lane detection")
    plt.savefig(f"images/precision_{hyper_batch}.png")
    plt.clf()
    plt.plot(range(len(recall)), recall, 'm--')
    plt.xlabel("frames")
    plt.ylabel("Recall")
    plt.title("Average Recall of lane detection")
    plt.savefig(f"images/recall_{hyper_batch}.png")
    plt.clf()
    plt.plot(range(len(f1)), f1, 'm--')
    plt.xlabel("frames")
    plt.ylabel("F1")
    plt.title("Durchschnittlicher F1-Score der Fahrspurerkennung")
    plt.savefig(f"images/F1_{hyper_batch}.png")
    plt.clf()
