# TPU
TPU CORAL TESTING
# OpenCV camera detect human face with Coral

## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.

2.  Clone or Update this Git repo onto your computer or Dev Board:

    ```
    git clone https://github.com/AndyHUI711/TPU.git
   
    git pull
    ```
3.  Go to TPU file

    ```
    cd TPU
    ```

4.  Install the OpenCV libraries:

    ```
    bash install_requirements.sh
    ```
5. Install h5py lib:
    ```
    pip3 install --upgrade pip
    pip3 install h5py
    ```
 6. Install tqdm lib:
    ```
    sudo pip3 install tqdm
    ```
 7. Install yaml lib:
    ```
    sudo pip3 install pyyaml
    ```
    
## Run the detection  (SSD models)

```
python3 detect.py
```

By default, this uses the ```mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.

## Run PPE detection (YOLOV5)
```
python3 detect_yolo.py -m all_models/yolov5_ppe/best_int8_edgetpu.tflite --stream 
```
