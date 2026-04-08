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
