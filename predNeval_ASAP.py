import tensorflow as tf
from utils.ASAP_utils import load_test_data, load_pretrained_model, predict_model, evaluate_model


# Use GPU
physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')
if (len(physical_device) >= 1):
    if (tf.config.experimental.get_memory_growth(physical_device[0]) == 1):
        tf.config.experimental.set_memory_growth(physical_device[0],True)


# Params for Config
params_config = {'nb_inputs' : 56,
                'in_seq_len' : 100,
                }

# Params for Model
params_model = {'cell_multiheadatt' : 16*4,
                'num_head_multiheadatt' : 4,
                'pruning_stat' : True,
                'cell_lstm' : 20,
                'cell_dense' : 20,
                }


if __name__ == "__main__":
    '''
    Prediction and evaluation of ASAP model

    Choose mode:
     - "pred": predict with ASAP
     - "eval": evaluate objectively by loading precomputed predictions of ASAP
     - "predNeval": predict and evaluate  objectively the predictions of ASAP
    '''
    mode = "predNeval"
    data_path = '../data'

    xij_test = load_test_data(data_path)
    model = load_pretrained_model(params_config, params_model)

    if(mode=="pred"):
        pred_dict = predict_model(model, xij_test, params_config, save=True)
    elif(mode=="eval" or mode=="predNeval"):
        if(mode=="eval"):
            saved_prediction = True
        else:
            saved_prediction = False
        [rmse_mean, ks_mean, dtw_mean_pred, dtw_mean_gt] = evaluate_model(model, xij_test, params_config, data_path, saved_prediction=saved_prediction)
        print("Objective evaluation results:")
        print("RMSE:", rmse_mean)
        print("KS test:", ks_mean)
        print("DTW of ASAP predictions(¬PA&PB):", dtw_mean_pred)
        print("DTW of Ground Truth(PA&PB):", dtw_mean_gt)
    else:
        print("Error: mode is either pred or eval")

