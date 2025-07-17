# YOLOv8 for STM32

This repository provides a collection of pre-trained and quantized yolov8 and yolo11 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.

## Benefits ✨
- Offers a set of models compatible with STM32 platforms and stm32ai-modelzoo.
- Offers a quantization friendly pose estimation model (fixed on the latest version of Ultralytics)
- A step by step guide on how to use AiRunner to evaluate yolov8 models on STM32N6.

## Notice
- If You combine this software (“Software”) with other software from STMicroelectronics ("ST Software"), to generate a software or software package ("Combined Software"), for instance for use in or in combination with STM32 products, You must comply with the license terms under which ST distributed such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any distribution or non-internal use of the Combined Software. You are responsible for compliance with applicable license terms for any Software You use, and as such, You must limit your use of this software and any Combined Software accordingly.

## Available YOLOv8 Models


| Models                                                      | Task                 | Input Resolution  | Format                         | Input Type      | Output Type           |
|-------------------------------------------------------------|----------------------|-------------------|--------------------------------|-----------------|-----------------------|
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 416x416x3         | per channel int8               | uint8           | float                 |
| [YOLO11n](stedgeai_models/object_detection/yolo11)          | person_detection     | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLO11n](stedgeai_models/object_detection/yolo11)          | person_detection     | 320x320x3         | per channel int8               | uint8           | float                 |
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

## Exporting quantization friendly YOLOv8 Pose Models

This fork offers a quantization friendly Yolov8 pose model, you can export the pose model by following these steps:

Note that the latest versions of Ultralytics are now equivalent to this fork
- Install YOLOv8 by cloning this Ultralytics GitHub repository. After cloning, navigate into the directory and install the package in editable mode -e using pip.
```bash
git clone https://github.com/stm32-hotspot/ultralytics.git

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

- Run the following code to export the quantization friendly pose model:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model
# or load a custom trained model
#model = YOLO('pytorch_models/yolov8n-pose.pt')

# Export the model
model.export(format='saved_model', simplify = True, imgsz = 256)
```

- Please note that the exported pose model cannot be evaluated with Utralytics scripts as the model output is normalized and in float.
- Once you generated the quantization friendly Yolov8 pose model in saved model format, you can now quantize it following this [stm32ai-modelzoo-services tutorial](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/tutorials/scripts/yolov8_quantization)
- Evaluation and inference scripts using Yolov8 pose models are available in [stm32ai-modelzoo-services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)

## Hand gesture detection model training
The gesture detection models have been trained on the Hagrid dataset:
- 19 classes (corresponding to the 18 initial class of Hagrid v1 and including the no gesture class)
- ST data augmentation (~200 k images)
    - Zoom on gestures to have better detection in short distance (10 %)
    - Multi-gestures mosaic to improve detection when several gestures are in the field (10%)
    - Background images (without gesture, 10%)

You can use the yaml file [user_config_yolov8n_hagrid_gesture_deploy.yaml](stedgeai_models/gesture_detection/user_config_yolov8n_hagrid_gesture_deploy.yaml) to deploy the model with 320x320x3 resolution with [STM32 model zoo services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/object_detection).

Once properly installed and configured, use the following command to deploy on a STM32N6 Discovery Kit:

```bash
python stm32ai_main.py --config-path . --config-name user_config_yolov8n_hand_gesture.yaml
```


## Deployment and management on STM32 boards

To efficiently deploy Yolov8 models on STM32 boards you can follow these tutorials:
-	[Object Detection](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/deployment/doc/tuto/How_to_deploy_yolov8_yolov5_object_detection.md)
-	[Pose Estimation](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/pose_estimation/deployment/doc/tuto/How_to_deploy_yolov8_pose_estimation.md)
-	[Instance Segmentation](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/instance_segmentation/deployment/doc/tuto/How_to_deploy_yolov8_instance_segmentation.md)

## Validation on target with AiRunner and STM32N6
You can run validation with yolov8 models directly on your STM32N6 board following this step by step guide:
- [How to use AiRunner to evaluate yolov8 on STM32N6](tutorials/How_to_use_AiRunner_to_evaluate_yolov8_on_STM32N6.md)
