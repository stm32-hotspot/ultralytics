# How to use AiRunner to evaluate a Yolov8n model on the STM32N6


## Overview

`stm_ai_runner` is a Python module providing an unified inference interface for the different stedgeai runtime: X86 or STM32.  
It allows to use the generated c-model from an user Python script like a `tf.lite.Interpreter` to perform an inference. According the capabilities of the associated run-time,profiling data are also reported (execution time per layer,...).  
For advanced usage, the user has the possibility to register a callback allowing to have the intermediate tensor/feature values before and/or after the execution of a node.  

It allows to easily run the exact same model from a Python evaluation environment either running the reference on:  
* the host runtime: tensorflow lite runtime or onnx runtime 
* the stedgeai runtime for standard STM32
* the stedgeai runtime running on the target (for standard STM32 or for the STM32N6 with NPU)

For more information about the AI Runner, please refer to the [online documentation](https://stedgeai-dc.st.com/assets/embedded-docs/ai_runner_python_module.html).  

This how to is dedicated to the STM32N6 case comparing execution of a tflite quantized model using tflite runtime on host and the model running on the STM32N6 NPU.  

## Pre-requisite

Ultralytics environment installed from source: https://docs.ultralytics.com/quickstart/#__tabbed_1_3  

Clone the ultralytics repository:  

```
git clone https://github.com/ultralytics/ultralytics

```

Navigate to the cloned directory:  

```
cd ultralytics

```

Install the package in editable mode for development  

```
pip install -e .
```

> [!WARNING]  
> If you installed Ultralytics environment directly through pip install, the files to modfiy will be located in your environment package location.  
> For instance, for a user on Windows using a miniconda environment, the scripts will be located in C:/Users/name/AppData/Local/miniconda3/envs/yolov8/Lib/site-packages/ultralytics/  

By default, Ultralytics requirements do not install the packages required to export to onnx or tensorflow lite.
When exporting for the first time, it will either use pre-installed packages or do an auto update installing the latest versions which then causes compatibility issues.
To ensure compatibility, you need to install (or downgrade) the versions of tensorflow, onnx and onnxruntime following below requirements:
Use a python 3.9 environment (for the tflite_support package dependency)
Tensorflow version between 2.8.3 and 2.15.1
ONNX version between 1.12.0 and 1.15.0
ONNX runtime version between 1.13 and 1.18.1
```
	pip install tensorflow==2.15.1
	pip install tf_keras==2.15.1
	pip install onnx==1.15.0
	pip install onnxruntime==1.18.1
```
Other packages can be installed through the auto update procedure when doing the first export command.

[ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) package with NPU extension (STEdgeAI-NPU) installed and ability to run the validation example on the STM32N6 Discovery board.   

## AI Runner installation

The AI Runner is a module handling the communication to the board.  
The AI Runner module is part of the ST Edge AI Core package as it is used for the validation feature.  
Once installed you can find it within the installation directory under scripts/ai_runner as stm_ai_runner.   
For instance the windows default installation folder for the v2 version of ST Edge AI Core is:  
C:/ST/STEdgeAI/2.0  
and the AI Runnner module is then   
C:/ST/STEdgeAI/2.0/scripts/ai_runner/stm_ai_runner   

The dependencies are minimal and it requires only:  
numpy  
protobuf  
pyserial==3.4  
tqdm  

In Ultralytics environmennt install the stm_ai_runner dependencies:  
```
cd C:/ST/STEdgeAI/2.0/scripts/ai_runner/stm_ai_runner
pip install -r requirements.txt
```

Set the environment variable `PYTHONPATH` to tell python where to find the module:   
```
$ export PYTHONPATH=<path/to/stm_ai_runner>:$PYTHONPATH
```

## Generating the Yolov8n tflite quantized model

Let's start from the reference Yolov8n object detector pre-trained on COCO 2017 80 classes.   

The first step is to export and quantize the model in a 192x192x3 resolution.  
To experiment we will use the default export to tflite int8 feature of Ultralytics through the Python interface:   

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# Export the model
model.export(format="tflite", imgsz=192, int8=True)
```

> [!NOTE]  
> For yolov5, use the yolov5nu version instead of the yolov5 that is using the same output shape than yolov8n.  

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolov5nu.pt")  # load an official model

# Export the model
model.export(format="tflite", imgsz=192, int8=True)
```

The default "coco8" small dataset will be downloaded and used as calibration dataset (otherwise a yaml file shall provide the link to the dataset).  
We ran some experimentation and even if the dataset is small, the qunantization on this dataset is efficient.  
The output of the export is in the generated yolov8n_saved_model directory.  

The two files of interest are:  
* yolov8n_integer_quant.tflite: the model quantized with input in float and output in float (ie the model first layer is a Quantize layer and the last layer a Dequantize layer)
* yolov8n_full_integer_quant.tflite: the model quantized with input in int8 and output in int8

The two models are equivalent except the input / output format.  

> [!WARNING]  
> For embedded device, the output of the camera are RGB values stored as uint8, it is therefore convenient to have models with uint8 input.  


ST embedded post processing is supporting float or int8 output.  

Currently, the deployment guidelines are for models with:  
* uint8 input, int8 output
* uint8 input, float output

It is why we are providing our own equivalent quantization scripts to be able to select the requested input / output format and per-channel quantization instead of per-tensor.  
Per-channel quantization is more efficient to maintain accuracy.   

> [!TIP] See the dedicated readme [How to deploy yolov8 yolov5 object detection](
https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/deployment/doc/tuto/How_to_deploy_yolov8_yolov5_object_detection.md)  

Ultralytics scripts are modified to support also these combinations of input/output.  


## Running a prediction with Ultralytics scripts

Utralytics scripts allows to run a prediction of the tflite model either from the PyTorch float model or the tflite quantized model.  

For the tflite model, the scripts support model with:  
* float input / float output: yolov8n_integer_quant.tflite
* int8 input / int8 output: yolov8n_full_integer_quant.tflite

Running a prediction of the float model:  

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("yolov8n.pt")

# Predict with the model
results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=192)
```

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("yolov8n_saved_model/yolov8n_full_integer_quant.tflite")

# Predict with the model
results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=192)
```

The result is stored in runs/predict.  


## Patching Ultralytics scripts to use the AI Runner

Be sure to have exported the path to the AI Runner module:  
Set the environment variable `PYTHONPATH` to tell python where to find the module.  

```
$ export PYTHONPATH=<path/to/STM32Cube_AI/scripts/ai_runner/stm_ai_runner>:$PYTHONPATH
```

Or copy the stm_ai_runner in the ultralytics directory.  

We will modify the ultralytics files to accept a new target stedgeai.  

For this only two files needs to be modified:  
* ultralytics/utils/checks.py: authorize a new target
* ultralytics/nn/autobackend.py: manage the new stedgeai runtime (on host or one target)

The following patch is based on Ultralytcis v8.2.39, you may need to slightly adapt to your version.  
Line to add are followed by `# add`, line to remove are commented and followed by `# remove`.  
For a full section, you can copy past from `# add start` to `# add end`.  

### Patch of check.py

Open ultralytics/utils/checks.py  

```
def check_file(file, suffix="", download=True, hard=True):
    """Search/download file (if necessary) and return path."""
        check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    # if ( # remove
    if file.startswith('stedgeai:'):  # stedgeai backend # add
        return file  # add
    elif (  # add
         not file
         or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
         or file.lower().startswith("grpc://")
```

### Patch of autobackend.py

Add stedgeai target to the list and specify that stedgeai is taking NHWC inputs as tflite:  

```
nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            ncnn,
            triton,
            stedgeai  # add
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        # nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH) # remove
        nhwc = stedgeai or coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)  # add
        stride = 32  # default stride
        model, metadata = None, None
```

Initialize the AI Runner for inference as for other target:  

```
        # NVIDIA Triton Inference Server
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
        # add start
        elif stedgeai:  # add
            LOGGER.info(f"Loading {w} for ST Edge AI inference...")
            from stm_ai_runner import AiRunner
            
            ai_runner_desc = w[len('stedgeai:'):]
            ai_runner_interpreter = AiRunner()
            if not ai_runner_interpreter.connect(ai_runner_desc):
                raise TypeError(f"model='{w}' unable to load the model")
            ai_runner_input_details = ai_runner_interpreter.get_inputs()  # inputs
            ai_runner_output_details = ai_runner_interpreter.get_outputs()  # outputs
            for detail in ai_runner_input_details:
                LOGGER.info(f" I: {detail.name} {detail.shape} {detail.dtype} {detail.scale} {detail.zero_point}")
            for detail in ai_runner_output_details:
                LOGGER.info(f" O: {detail.name} {detail.shape} {detail.dtype} {detail.scale} {detail.zero_point}")
        # add end
            
         # Any other format (unsupported)
         else:
             from ultralytics.engine.exporter import export_formats
```

Run the inference with the stedgeai for host or target:  

```
         # NVIDIA Triton Inference Server
         elif self.triton:
             im = im.cpu().numpy()  # torch to numpy
             y = self.model(im)
            
        # Add start
        # ST Edge AI Core engine
        elif self.stedgeai:
            # Align the output shape of stedgeai to align with tflite shape
            def reduce_shape(x):  # reduce shape (request by legacy API)
                old_shape = x.shape
                n_shape = [old_shape[0]]
                for v in x.shape[1:len(x.shape) - 1]:
                    if v != 1:
                        n_shape.append(v)
                n_shape.append(old_shape[-1])
                return x.reshape(n_shape)

            LOGGER.info(f"Running the ST Edge AI Core interpreter.. {w} {h}")

            im = im.cpu().numpy()  # nhwc format
            LOGGER.info(f" image                : {im.shape} {im.dtype} {im.min()} {im.max()} ")

            detail = self.ai_runner_input_details[0]
            # Manage files with int8 or uint8 inputs instead of float
            # Image is then quantized from float to integer
            is_int = detail.dtype == np.int8 or detail.dtype == np.uint8
            if is_int:
                scale, zero_point = detail.scale[0], detail.zero_point[0]
                LOGGER.info(f" .. quantize          : s={scale} zp={zero_point}")
                im = (im / scale + zero_point).astype(detail.dtype)  # quantize

            LOGGER.info(f" image pre-processed  : {im.shape} {im.dtype} {im.min()} {im.max()} ")
            preds, _ = self.ai_runner_interpreter.invoke(im)
            y = []
            for x, detail in zip(preds, self.ai_runner_output_details):
                LOGGER.info(f" prediction           : {x.shape} {x.dtype} {x.min()} {x.max()} ")
                x = reduce_shape(x)
                if detail.dtype in {np.int8, np.uint8}:
                    scale, zero_point = detail.scale[0], detail.zero_point[0]
                    x = (x.astype(np.float32) - zero_point) * scale  # dequant
                    LOGGER.info(f" .. de-quantize       : {x.shape} {x.dtype} {x.min()} {x.max()} s={scale} zp={zero_point}")

                if x.ndim == 3:
                    # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                    # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                    x[:, [0, 2]] *= w
                    x[:, [1, 3]] *= h
                    LOGGER.info(f" .. denormalize       : {x.shape} {x.dtype} {x.min()} {x.max()} w={w} h={h}")

                y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            LOGGER.info("") # add
        # Add end

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
```

Modification of the tensorflow lite section to support combination of inputs / outputs format:  

```
else:  # Lite or Edge TPU
                 details = self.input_details[0]
                # is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model # remove
                is_int = details["dtype"] in {np.int8, np.uint8, np.int16}  # is TFLite quantized int8 or int16 model # add
                if is_int:
                    scale, zero_point = details["quantization"]
                    LOGGER.info(f" .. quantize          : s={scale} zp={zero_point}") # add
                     im = (im / scale + zero_point).astype(details["dtype"])  # de-scale # add
                LOGGER.info(f" image pre-processed  : {im.shape} {im.dtype} {im.min()} {im.max()} ") # add
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    # if is_int: # remove
                    LOGGER.info(f" prediction           : {x.shape} {x.dtype} {x.min()} {x.max()} ") # add
                    if output["dtype"] in {np.int8, np.uint8}: # add
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        LOGGER.info(f" .. de-quantize       : {x.shape} {x.dtype} {x.min()} {x.max()} s={scale} zp={zero_point}") # add
                    if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                        LOGGER.info(f" .. denormalize       : {x.shape} {x.dtype} {x.min()} {x.max()} w={w} h={h}") # add
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            LOGGER.info("") # add
```

Patch the _model_type function to support stegdeai function:  

```
             url = urlsplit(p)
             triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}
 
        # return types + [triton] # remove
        stedgeai = p.startswith('stedgeai:') # add
        return types + [triton, stedgeai] # add
```


## Validating the model with ST Edge AI Core

Connect your STM32N6 board as usual to your PC in Dev mode (boot switches on the right).  
Go to the folder scripts/N6_scripts of the ST Edge AI Core installation.  
For instance on Windows: C:/ST/STEdgeAI/2.0/scripts/ai_runner/stm_ai_runner/scripts/N6_scripts.   
Copy the quantized model yolov8n_full_integer_quant.tflite in the models folder.  

Launch the model generation for N6:  

```
stedgeai generate --model ./models/yolov8n_integer_quant.tflite --target stm32n6 --st-neural-art profile_O3@user_neuralart.json
```

Build and load the validation application thanks to the scripts:  

```
python n6_loader.py
```

Run an evaluation on random data:  

```
stedgeai validate --model ./models/yolov8n_integer_quant.tflite --target stm32n6 --mode target --desc serial:921600
```

As usual you can check the generation and validation reports in the folder st_ai_output.   
The validation FW is now programmed on the N6 board.  

> [!WARNING]  
> Do not disconnect the board until the end of the experimentation, being in Dev Mode, the application is stored in RAM and not in external Flash.  
> Each time you want to run an evaluation with the AI Runner, be sur to run this step before.  


## Running a prediction with Ultralytics scripts & AI Runner

Running a prediction of the yolov8n_full_integer_quant model with tensorflow lite runtime on host:  

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("yolov8n_saved_model/yolov8n_full_integer_quant.tflite")

# Predict with the model
results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=192)
```

Running a prediction of the yolov8n_full_integer_quant model with stedge runtime on host:  

The runtime for standard STM32 is used as reference. It requires then to build it locally with the following command:  

```
stedgeai generate -m yolov8n_saved_model/yolov8n_full_integer_quant.tflite --target stm32 --dll
```

The host library will be built in st_ai_ws directory.  

Then run:  

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("stedgeai:st_ai_ws")

# Predict with the model
results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=192)
```

Running a prediction of the yolov8n_full_integer_quant model with stedge runtime on STM32N6:  

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("stedgeai:serial:921600")

# Predict with the model
results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=192)
```

The data with the expected format for the model will be sent from the PC to the board through the USB virtual com and model output sent back to be post-processed.  

Therefore the different outputs can be compared.  

> [!WARNING]  
> Due to hardware (like size of the accumulators) and runtime differences, bit exactness is impossible.  
> Even between two versions of tensorflow runtime, we observed significant differences or between type of OS (Linux / Windows).    
> However, similar performance are expected.  
> Be also aware that due to quantization the computed confidence level have steps rather than continous values.  

The Ultralytcis validation command is also working (with a batch of 1):  

```
from ultralytics import YOLO

# Load the float reference Pytorch model
model = YOLO("stedgeai:serial:921600")

# Predict with the model
results = model.val(data="coco.yaml", imgsz=192, batch=1, device="cpu")
```
> [!NOTE]  
> The time bottleneck time to run a model on the N6 board is due to the virtual com speed baudrate (921600) and therefore on the image size.  
> It is recommended to use subset of the test set to validate on target.  

## License

AiRunner is licensed under the BSD 3-Clause license.  
Ultralytics scripts are lcinesed under the AGPL-3.0 license.  

## Copyright

Copyright (c) 2024 STMicroelectronics.  
All rights reserved.  

