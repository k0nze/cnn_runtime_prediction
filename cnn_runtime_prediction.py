import joblib
import numpy as np
import json

# clock speed in Hz
clock_speed = 1.4e9

path_to_cnn_json_file = "./data/json/alexnet.json"

model_prefix = "./data/models/"

path_to_convolution_model_file =     model_prefix + "convolution_model.joblib"
path_to_max_pooling_model_file =     model_prefix + "max_pooling_model.joblib"
path_to_lrn_model_file =             model_prefix + "lrn_model.joblib"
path_to_fully_connected_model_file = model_prefix + "fully_connected_model.joblib"
path_to_relu_model_file =            model_prefix + "relu_model.joblib"
path_to_softmax_model_file =         model_prefix + "softmax_model.joblib"


# load prediction models
convolution_model = joblib.load(path_to_convolution_model_file)
max_pooling_model = joblib.load(path_to_max_pooling_model_file)
lrn_model = joblib.load(path_to_lrn_model_file)
fully_connected_model = joblib.load(path_to_fully_connected_model_file)
relu_model = joblib.load(path_to_relu_model_file)
softmax_model = joblib.load(path_to_softmax_model_file)


def convolution_num_mac_ops(Iw, Ih, Ic, n, s, p, Oc):
    Iw_p = Iw + p * 2 * int(n / 2)
    Ih_p = Ih + p * 2 * int(n / 2)

    return (((Iw_p - n) / s) + 1) * (((Ih_p - n) / s) + 1) * n * n * Ic * Oc

def max_pooling_num_comp_ops(Iw, Ih, Ic, n, s):
    return (((Iw - n) / s) + 1) * (((Ih - n) / s) + 1) * n * n * Ic


def lrn_num_mul_div_ops(Iw, Ih, Ic, n):
    return Iw * Ih * Ic * (n + 4)


def fully_connected_num_mac_ops(Iw, Ow):
    return Iw * Ow


# read CNN json description
with open(path_to_cnn_json_file) as cnn_json_file:
    cnn = json.load(cnn_json_file)

layer_names = sorted(cnn)

predicted_cpu_cycles = 0

# predict cpu cycles for each layer of the CNN
for layer_name in layer_names:

    print(layer_name)

    # get layer_kind from layer_name
    layer_kind = layer_name.split('_', 1)[1]

    layer_parameter = []

    if layer_kind == "convolution":

        layer_parameter = [
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_channels'],
            cnn[layer_name]['kernel_size'],
            cnn[layer_name]['stride'],
            cnn[layer_name]['padding'],
            cnn[layer_name]['output_channels']
        ]

        num_mac_ops = convolution_num_mac_ops(
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_channels'],
            cnn[layer_name]['kernel_size'],
            cnn[layer_name]['stride'],
            cnn[layer_name]['padding'],
            cnn[layer_name]['output_channels']
        )

        layer_parameter.append(num_mac_ops)

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += convolution_model.predict(x)[0]

    elif layer_kind == "max_pooling":

        layer_parameter = [
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_channels'],
            cnn[layer_name]['kernel_size'],
            cnn[layer_name]['stride']
        ]

        num_comp_ops = max_pooling_num_comp_ops(
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_channels'],
            cnn[layer_name]['kernel_size'],
            cnn[layer_name]['stride']
        )

        layer_parameter.append(num_comp_ops)

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += max_pooling_model.predict(x)[0]

    elif layer_kind == "lrn":

        layer_parameter = [
            cnn[layer_name]['input_width'],
            cnn[layer_name]['input_channels']
        ]

        num_mul_div_ops = lrn_num_mul_div_ops(cnn[layer_name]['input_width'], cnn[layer_name]['input_width'], cnn[layer_name]['input_channels'], 5)

        layer_parameter.append(num_mul_div_ops)

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += lrn_model.predict(x)[0]

    elif layer_kind == "fully_connected":

        layer_parameter = [
            cnn[layer_name]['input_size'],
            cnn[layer_name]['output_size']
        ]

        num_mac_ops = fully_connected_num_mac_ops(cnn[layer_name]['input_size'], cnn[layer_name]['output_size'])

        layer_parameter.append(num_mac_ops)

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += fully_connected_model.predict(x)[0]

    elif layer_kind == "relu":

        layer_parameter = [cnn[layer_name]['input_size']]

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += relu_model.predict(x)[0]

    elif layer_kind == "softmax":

        layer_parameter = [cnn[layer_name]['input_size']]

        x = np.array(layer_parameter).reshape(1, -1)

        predicted_cpu_cycles += softmax_model.predict(x)[0]

    print(layer_parameter)

run_time = predicted_cpu_cycles/clock_speed

print()
print("Predicted CPU cycles: " + str(int(predicted_cpu_cycles)))
print("Predicted Runtime:    " + str(round(run_time, 5)) + " s")
