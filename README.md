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
