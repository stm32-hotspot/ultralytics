# YOLOv8 for STM32

This repository provides a collection of pre-trained and quantized yolov8 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.

## Benefits ✨

- Offers a set of models compatible with STM32 platforms and stm32ai-modelzoo.


## Notice
 
- If You combine this software (“Software”) with other software from STMicroelectronics ("ST Software"), to generate a software or software package ("Combined Software"), for instance for use in or in combination with STM32 products, You must comply with the license terms under which ST distributed such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any distribution or non-internal use of the Combined Software. You are responsible for compliance with applicable license terms for any Software You use, and as such, You must limit your use of this software and any Combined Software accordingly.


## Available YOLOv8 Models


| Models                                                                                                      | Task                 | Input Resolution  | Format	                      | Input Type      | Output Type           |
|-------------------------------------------------------------------------------------------------------------|----------------------|-------------------|--------------------------------|-----------------|-----------------------|
| [YOLOv8n](stedgeai_models/object_detection/yolov8n_256_quant_pc_uf_od_coco-person-st.tflite)                | person_detection     | 256x256x3         | per channel int8               | uint8           |float                  |
| [YOLOv8n](stedgeai_models/object_detection/yolov8n_320_quant_pc_uf_od_coco-person-st.tflite)                | person_detection     | 320x320x3         | per channel int8               | uint8           |float                  |
| [YOLOv8n](stedgeai_models/object_detection/yolov8n_416_quant_pc_uf_od_coco-person-st.tflite)                | person_detection     | 416x416x3         | per channel int8               | uint8           |float                  |


- Quantization, evaluation and deployment of Yolov8 models will be soon available in [stm32ai-modelzoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
