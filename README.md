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

2.  Clone this Git repo onto your computer or Dev Board:

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
    
## Run the detection  (SSD models)

```
python3 detect.py
```

By default, this uses the ```mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.

