# YOLOv8 for STM32

This repository provides a collection of pre-trained and quantized yolov8 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.

## Benefits ✨

- Offers a set of models compatible with STM32 platforms and stm32ai-modelzoo.
- Offers a quantization friendly pose estimation model.

## Notice

- If You combine this software (“Software”) with other software from STMicroelectronics ("ST Software"), to generate a software or software package ("Combined Software"), for instance for use in or in combination with STM32 products, You must comply with the license terms under which ST distributed such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any distribution or non-internal use of the Combined Software. You are responsible for compliance with applicable license terms for any Software You use, and as such, You must limit your use of this software and any Combined Software accordingly.


## Available YOLOv8 Models


| Models                                                      | Task                 | Input Resolution  | Format                         | Input Type      | Output Type           |
|-------------------------------------------------------------|----------------------|-------------------|--------------------------------|-----------------|-----------------------|
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 256x256x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 320x320x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/object_detection/)                | person_detection     | 416x416x3         | per channel int8               | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 256x256x3         | per tensor int8                | uint8           | float                 |
| [YOLOv8n](stedgeai_models/pose_estimation/)                 | pose_estimation      | 256x256x3         | per channel int8               | uint8           | float                 |


## Exporting quantization friendly YOLOv8 Pose Models

This fork offers a quantization freindly Yolov8 pose model, you can export the pose model by following these steps:

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

- Please note that the exported pose model cannot be evaluated with Utralytics scripts as the model output is normalized.
- Once you generated the quantization friendly Yolov8 pose model in saved model format, you can now quantize it following this [stm32ai-modelzoo tutorial](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/tutorials/scripts/yolov8_quantization)
- Evaluation and inference scripts using Yolov8 pose models are available in [stm32ai-modelzoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)