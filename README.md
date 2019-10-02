# Environment
Python Version `3.5.1` or higher

```{bash}
cd cnn_runtime_prediction
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run Prediction
Inside of `cnn_runtime_prediction.py` the following variables should be set:

* `clock_speed`: clock speed of target plattform in Hz
* `path_to_cnn_json_file`: path to the JSON file which contains the CNN description
* `model_prefix`: path to the directory in which all model files are stored
* `path_to_*_model_file`: path to a specific model file

```{bash}
python3 cnn_runtime_prediction.py
```

# Models

## Raspberry Pi 3B+
### Files:
```
data/models/convolution_model_raspberrypi3bplus.joblib
data/models/fully_connected_model_raspberrypi3bplus.joblib
data/models/lrn_model_raspberrypi3bplus.joblib
data/models/max_pooling_model_raspberrypi3bplus.joblib
data/models/relu_model_raspberrypi3bplus.joblib
data/models/softmax_model_raspberrypi3bplus.joblib
```

### Measurements
 * Software: [Pico-CNN](https://github.com/ekut-es/pico-cnn)
 * OS: Linux raspberrypi 4.9.80-v7+ #1098 SMP Fri Mar 9 19:11:42 GMT 2018 armv7l GNU/Linux, Raspbian 9 Stretch
 * Compiler: gcc (Raspbian 6.3.0-18+rpi1+deb9u1) 6.3.0 20170516
 * Optimizations: `-O0`
 * `clock_speed = 1.4e9`

## Raspberry Pi 2B v1.1
### Files:
```
data/models/convolution_model_raspberrypi2bv11.joblib
data/models/fully_connected_model_raspberrypi2bv11.joblib
data/models/lrn_model_raspberrypi2bv11.joblib
data/models/max_pooling_model_raspberrypi2bv11.joblib
data/models/relu_model_raspberrypi2bv11.joblib
data/models/softmax_model_raspberrypi2bv11.joblib
```

### Measurements
 * Software: [Pico-CNN](https://github.com/ekut-es/pico-cnn)
 * OS: Linux raspberrypi 4.9.80-v7+ #1098 SMP Fri Mar 9 19:11:42 GMT 2018 armv7l GNU/Linux, Raspbian 9 Stretch
 * Compiler: gcc (Raspbian 6.3.0-18+rpi1+deb9u1) 6.3.0 20170516
 * Optimizations: `-O0`
 * `clock_speed = 900e6`

## ODROID-XU4
### Files:
```
data/models/convolution_model_odroidxu4.joblib
data/models/fully_connected_model_odroidxu4.joblib
data/models/lrn_model_odroidxu4.joblib
data/models/max_pooling_model_odroidxu4.joblib
data/models/relu_model_odroidxu4.joblib
data/models/softmax_model_odroidxu4.joblib
```

### Measurements
 * Software: [Pico-CNN](https://github.com/ekut-es/pico-cnn)
 * OS: Linux odroidxu4 4.14.85-152 #1 SMP PREEMPT Mon Dec 3 03:00:02 -02 2018 armv7l GNU/Linux, Ubuntu 18.04.1 LTS
 * Compiler: gcc (Ubuntu/Linaro 7.4.0-1ubuntu1~18.04.1) 7.4.0
 * Optimizations: `-O0`
 * `clock_speed = 2.0e9`
