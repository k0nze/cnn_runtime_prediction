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
 * OS: Linux raspberrypi 4.9.80-v7+ #1098 SMP Fri Mar 9 19:11:42 GMT 2018 armv7l GNU/Linux, Raspbian
 * Compiler: gcc (Raspbian 6.3.0-18+rpi1+deb9u1) 6.3.0 20170516
 * Optimizations: `-O0`
