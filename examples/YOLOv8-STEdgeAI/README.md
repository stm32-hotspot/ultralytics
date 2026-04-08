# YOLO for STM32

This repository provides a collection of pre-trained and quantized yolov8, yolo11, yolo26 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.

## Benefits ✨
- Offers a set of models compatible with STM32 platforms and stm32ai-modelzoo.
- Offers a step by step guide on how to export quantization friendly yolo26 onnx models to be used with stm32ai-modelzoo-services and deployed on STM32N6 Discovery Kit.
- Offers a quantization friendly pose estimation model (fixed on the latest version of Ultralytics)
- A step by step guide on how to use AiRunner to evaluate yolov8 models on STM32N6. [link](tutorials/How_to_use_AiRunner_to_evaluate_yolov8_on_STM32N6.md)
- A guide on how to deploy the gesture detection model on STM32N6. [link](tutorials/how_to_deploy_hand_gesture_model.md)

## Notice
- If You combine this software (“Software”) with other software from STMicroelectronics ("ST Software"), to generate a software or software package ("Combined Software"), for instance for use in or in combination with STM32 products, You must comply with the license terms under which ST distributed such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any distribution or non-internal use of the Combined Software. You are responsible for compliance with applicable license terms for any Software You use, and as such, You must limit your use of this software and any Combined Software accordingly.

## Available YOLO Models


| Models                                                      | Task                 | Input Resolution  | Format                         | Input Type      | Output Type           |
|-------------------------------------------------------------|----------------------|-------------------|--------------------------------|-----------------|-----------------------|
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 416x416x3         | per channel int8               | uint8           | float                 |
| [YOLO11n](stedgeai_models/object_detection/yolo11)          | person_detection     | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/gesture_detection/)               | gesture detection    | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/gesture_detection/)               | gesture detection    | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 256x256x3         | per tensor int8                | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 192x192x3         | per channel int8               | uint8           | float                 |
| [YOLO11n](stedgeai_models/pose_estimation/yolo11)           | pose_estimation      | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLO11n](stedgeai_models/pose_estimation/yolo11)           | pose_estimation      | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/segmentation/)                    | segmentation         | 256x256x3         | per channel int8               | int8            | int8                  |
| [YOLOv8n](stedgeai_models/segmentation/)                    | segmentation         | 320x320x3         | per channel int8               | int8            | int8                  |
| [YOLO11n](stedgeai_models/segmentation/yolo11)              | segmentation         | 256x256x3         | per channel int8               | int8            | int8                  |

# YOLO Ultralytics to STM32N6 Deployment Guide

This README explains how to train a YOLO26n model using the ST Ultralytics fork, export it to ONNX format, quantize it using STM32AI Model Zoo Services, and deploy it on an STM32N6 board.

## A. Train and Export in ST Ultralytics Fork

### 1) Clone ST Ultralytics fork

```bash
git clone https://github.com/stm32-hotspot/ultralytics.git
```

### 2) Navigate to repository

```bash
cd ultralytics
```



### 3) Install Ultralytics

Create a Python 3.12 environment
```bash
python -m venv yolo-env
```

Activate the environment:

- On Windows: yolo-env\Scripts\activate
- On Unix or MacOS: source yolo-env/bin/activate

then install the package in editable mode to be able to use the yolo command line tool from anywhere in the environment, and to reflect any changes made to the code without needing to reinstall.

```bash
pip install -e .
```


### 4) Go to YOLOv8-STEdgeAI example folder

```bash
cd examples/YOLOv8-STEdgeAI
```


### 5) Install ONNX dependencies for export and inference

```bash
pip install onnx==1.16.1 onnxruntime==1.20.1 tqdm
```

### 6) Download dataset
We will use a small dataset as an example, based on the COCO 2017 validation dataset for this tutorial. You can change the dataset by updating the paths in the next steps accordingly.

We provide a script to download and prepare the dataset for training and evaluation. Run the following command from the current directory:

```bash
python download_dataset.py
```

### 7) Convert dataset to YOLO format
As required by the current version of the training pipeline, we need to convert the COCO annotations to YOLO format. Run the following command from the current directory:

```bash
python convert_dataset.py --coco_images_dir datasets/coco/images/val --coco_annotations_file datasets/coco/annotations/instances_val2017.json
```

### 8) Train model
In this example we will train a YOLO26n model for 3 epochs you can adjust the number of epochs as needed, with an image size of 256. You can adjust these parameters as needed, but make sure to keep the image size consistent across training, export, and evaluation steps.

```bash
yolo train model=yolo26n.pt data=dataset.yaml epochs=3 imgsz=256
```

### 9) Export to ONNX
Now we will export the trained model to ONNX format, which is compatible with STM32AI Model Zoo Services. Make sure to keep the opset version consistent with the one specified in the quantization configuration in the next steps.

Update the model path in the command below if the model is saved in a different location.
```bash
yolo export model=../../runs/detect/train/weights/best.pt format=onnx end2end=False imgsz=256 simplify=True opset=17
```

### 10) Evaluate exported ONNX model
Update the model path in the command below if the model is saved in a different location. This step is important to verify that the exported ONNX model has good accuracy before proceeding with quantization and deployment. Make sure to keep the image size and dataset paths consistent with the previous steps.
```bash
yolo val task=detect model=..\..\runs\detect\train\weights\best.onnx imgsz=256 data=dataset.yaml
```

## B. Quantize and Deploy with STM32AI Model Zoo Services

### 11) Clone STM32AI Model Zoo Services
make sure to clone the project in the same location as the ST Ultralytics fork for easier path management, but this is not mandatory. You can clone it anywhere and update the paths in the next steps accordingly.

```bash
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git --depth 1
```

### 12) Navigate to repository

```bash
cd stm32ai-modelzoo-services
```

### 13) Initialize and update submodules

```bash
git submodule update --init --recursive
```

If this step fails, rerun it from the repository root and verify network/proxy access.

### 14) Create a dedicated environment and install requirements

```bash
conda create -n st_zoo python=3.12.9
conda activate st_zoo
pip install -r requirements.txt
```

### 15) Navigate to object detection use case

```bash
cd object_detection
```

### 16) Convert dataset to TFS format
```bash
cd ./datasets/dataset_create_tfs
```
- update dataset_config.yaml with the coco dataset paths as follows and make sure to keep the paths consistent with the ones used in the previous steps in the ST Ultralytics fork, you can also adjust the max_detections and exclude_unlabeled_images settings as needed.

```yaml
dataset:
  dataset_name: coco_80_val
  training_path: ../../../../ultralytics/examples/YOLOv8-STEdgeAI/datasets/coco/images/val
  validation_path: ../../../../ultralytics/examples/YOLOv8-STEdgeAI/datasets/coco/images/val
  test_path: ../../../../ultralytics/examples/YOLOv8-STEdgeAI/datasets/coco/images/val

settings:
  max_detections: 20
  exclude_unlabeled_images: True # If set to False, images without ground truths will be included in the dataset.
  
hydra:
  run:
    dir: outputs/${now:%Y_%m_%d_%H_%M_%S}

```

### 17) Navigate back to object_detection and run dataset conversion script

```bash
cd ../..
```

### 18) Update user_config.yaml to quantize and evaluate exported ONNX model

```yaml

operation_mode: chain_eqe

model:
   model_type: yolo26n
   model_path: tuto/ultralytics/runs/detect/train/weights/best.onnx

dataset:
  dataset_name: coco
  format: tfs
  class_names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
                    'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
  test_path: ../../../../ultralytics/examples/ST-YOLO/datasets/coco/images/val
  quantization_path: ../../../../ultralytics/examples/ST-YOLO/datasets/coco/images/val
  quantization_split: 0.001

preprocessing:
   rescaling:
      scale: 1/255
      offset: 0
   resizing:
      aspect_ratio: fit
      interpolation: nearest
   color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: False #True   # Plot precision versus recall curves. Default is False.
  max_detection_boxes: 100

quantization:
  quantizer: onnx_quantizer
  target_opset: 17
  granularity: per_channel #per_channel
  quantization_type: PTQ
  quantization_input_type: float 
  quantization_output_type: float
  export_dir: quantized_models

mlflow:
   uri: ./tf/src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```

- run the quantization and evaluation pipeline:

```bash
python stm32ai_main.py
```

### 19) Deploy exported ONNX model on N6

- update user_config.yaml to deploy the onnx model on the N6 as follows, and make sure to update the model_path with the path to the quantized model generated from the previous step:

```yaml
operation_mode: deployment

model:
  model_type: yolo26n
  model_path: tf/src/experiments_outputs/%Y_%m_%d_%H_%M_%S}/quantized_models/best_quant_qdq_pc.onnx # update with the path to the quantized model generated from the previous step

dataset:
  class_names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
                    'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

preprocessing:
  resizing:
    aspect_ratio: crop
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.5
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  max_detection_boxes: 10

tools:
  stedgeai:
    optimization: balanced
    on_cloud: False
    path_to_stedgeai: C:/ST/STEdgeAI/4.0/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_1.17.0/STM32CubeIDE/stm32cubeide.exe

deployment:
  c_project_path: ../application_code/object_detection/STM32N6/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK

mlflow:
   uri: ./tf/src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

- verify the board setup follwing this tutorial: https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_DEPLOYMENT_STM32N6.md#3-deployment

- run:

```bash
python stm32ai_main.py
```

## Notes

- Replace placeholder paths (for example, model_path values) with your real local paths before running.
- Keep ONNX opset aligned between export and quantization settings.
- Use the same class order everywhere (training, evaluation, quantization, deployment).


