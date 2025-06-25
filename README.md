# Lane Detector Fork for SmartRollerz DHBW - Lane Detection on Coral Edge TPU

Hello,
this repository is part of the bigger DHBW Smart Rollerz project (https://dhbw-smartrollerz.org/). We work on a autonmous model car for a interdisciplinary university competition. 

Ben Schlauch and Alwin Zomotor try to bring Edge TPU Lane Detection with this architecture to the SmartRollerz car. 



## Contents
- [Lane Detector implementation in Keras](#lane-detector-implementation-in-keras)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Dependencies](#dependencies)
  - [How to use it](#how-to-use-it)
    - [Training](#training-and-dataset-creation)
    - [TF-Lite model convertion](#tf-lite-model-convertion)
    - [Test the model](#test-the-model)
    - [TF-lite Hexagon Delegate test (Snapdragon 835/Hexagon 682)](#tf-lite-hexagon-delegate-test-snapdragon-835hexagon-682)
    - [TODO](#todo)
    - [References](#references)


## Overview
- Create a model for multi-lane detctor that runs on Coral Edge TPU
- Use CVAT or TUSimple labeled images for training
- Uses Birdseyeview
- Over 40 FPS on coral tpu inncluding postprocessing on CPU
- "master" branch for training, "ROS" branch for usage on the competition model car inside a ROS system

<b>The main network architecture</b>:
![](images/model_arch.png) 

- The input of model is perspective image, and the outputs are anchor offset, class prob and instance data called embeddings
- We split the input image as multiple anchors (n x n):
  - Each anchor responsible for data precting only if lane cross to it.
  - Predcited data would be offset of x, class prob and embeddings.
  - Each lane will have unique embedding index in one image for instance segmentation. see [link](https://arxiv.org/abs/1708.02551) for more details about  embeddings.

- Our model is created by:
  -  resnet block based model as backbone
  -  3 branchs as raw output for training:
     - <b>x_cls</b> : Class probability at each anchor that the lane or background.
     - <b>x_offsets</b> :  Offsets at each anchor (only x offset be cared)
     - <b>x_embeddings</b> : embedding data for instance segmentation.
  - <b>OutputMuxer</b> : A data muxter to mux raw outputs.

## Dependencies
- Tensorflow 2.4.0-dev20200815
- numpy
- opencv-python
- matplotlib

## How to use it

### Training and dataset creation:

Tested only on **Ubuntu 24.04**.

---

#### Requirements

- **Access to NAS** for training data, ROS bags, and configuration files
- **Using a virtual Python-Environment is recommended**

---

#### Setup steps

##### 1. Clone the repository

```bash
git clone https://github.com/Ben-schlch/edge_tpu_lane_detection.git
cd edge_tpu_lane_detection
```

---

##### 2. Prepare directory structure

Inside the project directory, create the following:

```
edge_tpu_lane_detection/
└── format_root/
    ├── images/
    ├── train_set/
    └── test_set/
```

---

##### 3. Download required files from NAS

From: `Team Folder/Daten/Lane Detection/edge_tpu_lane_detection_files`:

- `train_set1.xml`
- `test_set1.xml`
- `extract_rosbag.py`
- `distortion_cpy.py`
- `distortion_config.pkl`
- `calib_2023_12-19.png`

Also required:

- At least **one ROS bag file (.bag)** from `Team Folder/Daten/ROS Bags/ROS1/`  
  → Example destination directory: `edge_tpu_lane_detection/rosbags/`

---

##### 4. Organize files

- `train_set1.xml` → `format_root/train_set/`
- `test_set1.xml` → `format_root/test_set/`
- `distortion_config.pkl` → `rosbags/`
- Remaining files (except ROS-Bags from Step 3) → root project directory `edge_tpu_lane_detection/`

---

##### 5. Install Python 3.10

Follow the guide:
👉 [Installing Python 3.10 with Ubuntu 24.04](https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb)

**Important:**

- **Step 4 (symbolic link)** skip
- **Step 6** skip
- **Step 7** run **outside the project directory**  
  → Or: Add `venv/` into `.gitignore` file.

---

##### 6. Activate Virtual Environment

```bash
source venv/bin/activate
```

---

##### 7. Install Dependencies

```bash
pip install -r requirements.txt
pip install rosbags
```

---

##### 8. Adapt `distortion_cpy.py`

Edit the following lines:

- **Line 17:** Update path to `distortion_config.pkl`  
- **Line 18:** Update path to `calib_2023_12-19.png`  
- **Line 75:** Comment out the line  
- Before **Line 81** insert with path to `calib_2023_12-19.png`:

```python
self._folder_path = '<YourPath>/calib_2023-12-19.png'
images = glob.glob(self._folder_path)
```

---

##### 9. Extract images from ROS-Bags

Run from project root:

```bash
python extract_rosbag.py --bag "rosbags_Ordner/*" --output "format_root/images" --step_size 10
```

---

##### 10. Create symlinks

Run from `edge_tpu_lane_detection/add_ins/`:

```bash
python create_symlinks.py "/<YourPath>/edge_tpu_lane_detection/format_root/" "<YourPath>/edge_tpu_lane_detection/format_root/images/"
```

Expected directory structure:

```
format_root/
├── images/
│   └── 2023-12-19/
├── train_set/
│   ├── train_set1.xml
│   ├── <symlink>_2023-12-19/
├── test_set/
│   ├── test_set1.xml
└── └── <symlink>_2023-12-19/
```

#### **Note:** The `TuSimple_dataset_path` field in `config.json` currently has no effect on the training pipeline based on my understanding.  
However, reviewing or adjusting it could be beneficial for future extensions or related projects.

---

##### 11. Start Training

Run from the root directory:

```bash
python train_tflite.py
```

---

#### Project directory overview

```
edge_tpu_lane_detection/
├── add_ins/
│   └── create_symlinks.py
├── format_root/
│   ├── images/
│   ├── train_set/
│   │   ├── train_set1.xml
│   │   └── <symlink>_2023-12-19/
│   └── test_set/
│       ├── test_set1.xml
│       └── <symlink>_2023-12-19/
├── rosbags/
│   ├── *.bag
│   └── distortion_config.pkl
├── extract_rosbag.py
├── distortion_cpy.py
├── calib_2023_12-19.png
├── train_tflite.py
└── requirements.txt
```

### TF-Lite model convertion and Edge TPU Compiler
Once the training finish, we must convert model as TF-Lite model for mobile platform. Please run <b>generate_tflite_model.py</b> to convert model, the converted model is named same as element <b>"tflite_model_name"</b> at config.json.

    > python3 generate_tflite_model.py

Once the TFlite model is created, we need to use the edge tpu compiler to generate a model compatibel with the Coral. Please note you need to set the experimental and undocumented "a" flag because of the subgraphs 

    > edgetpu_compiler -a [modelpath...]


### Test the model
<b>test_tflite_model.py</b> is used to load and test converted tf-lite at previous step, this program will inference tf-lite model and rendering the result of inferencing.

    > python3 test_tflite_model.py

There is also a file to test the edge tpu model

Postprocessing looks like this:
![](images/post_process_at_test.png) 


### Final results
On our proprietary dataset, our latest model for the Edge Tpu performed with XXX. 

### TODO
-..........



### References
- Please check out the original repository. https://github.com/ML-Cai/LaneDetector
- We wrote a 80 page very detailed documentation of our project and also in depth about this architecture and our optimizations and struggles. For more information, please message me. 



