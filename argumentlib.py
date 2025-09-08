import argparse
import configparser
import logging
import sys
import os

from pathlib import Path

logger = logging.getLogger(__name__)

formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')

parser = argparse.ArgumentParser(description='Multi-task MRI Arguments')

os.chdir(sys.path[0])

def parse_args():
    print("parse_args gets executed")
    parser.add_argument('--gpu_id', type=str, default=0, help='Which GPU to use. If None, use CPU')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers used')

    ### training args ####
    parser.add_argument('--save_name', type=str, default='itermask3d_flair', help='saving file name')
    parser.add_argument('--epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--save_epoch', type=int, default=20, help='save the model every x number of epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_data_path', type=str, default='./dataset/ADNI',help='Train dataset path in which the data is stored')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size at train time')
    parser.add_argument('--train_modality', type=str, default='flair', help='input modality')
    parser.add_argument('--train_file_name_txt', type=str, default=None,
                        help='optional, the list of all the selected training subject names stored in a txt file. If none, then all the files in the directory will be used.')
    parser.add_argument('--drop_learning_rate_epoch', type=int, default=300, help='at which epoch the learning rate starts to drop')
    parser.add_argument('--drop_learning_rate_value', type=float, default=1e-4, help='learning rate drop rate')

    ### testing args ###
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size at test time')
    parser.add_argument('--test_modality', type=str, default='flair', help='input modality')
    parser.add_argument('--test_file_name_txt', type=str, default=None,
                        help='The list of all the selected training subject names stored in a txt file.')
    parser.add_argument('--test_data_path', type=str, default='./dataset/BRATS', help='Test dataset path in which the data is stored')
    parser.add_argument('--load_model_path', type=str, default=None, help='the model to load for testing')
    parser.add_argument('--gamma', type=float, default=0.01, help='The first derivative value gamma that is used to decide the threshold for shrinking')
    parser.add_argument('--testing_task', type=str, default='detection',
                        help='choose from detection or segmentation')
    parser.add_argument('--fit_funtion_y_limit', type=float, default=10, help='the fit function only fit points below certain limit')
    parser.add_argument('--fit_function_plot', type=bool, default=False, help='If you want to plot the fit function to see how it performs')
    parser.add_argument('--detection_score', type=str, default='final_mask_size', help='for detection, when computing the metric, how to define the anomaly score, final_mask_size or mean_error')
    parser.add_argument('--test_data_path2', type=str, default='', help='for detection, if second dataset is needed, the path of the second dataset')
    parser.add_argument('--shrinking_start_mask', type=str, default='brain_mask', help='whether the shrinking process should be initialized from the entire brain mask or from the optimal thresholded mask, choose from brain_mask or best_threshold_mask')
    parser.add_argument('--test_file_name_txt2', type=str, default=None,
                        help='optional, the list of all the selected training subject names stored in a txt file. If none, then all the files in the directory will be used.')
    parser.add_argument('--synthetic_anomaly', type=bool, default=True, help='whether or not to use synthetic anomaly detection')
    parser.add_argument('--synthetic_anomaly_type', type=str, default='spike',
                        help='If the task is synthetic anomalies, here are ways of generating some, choose from gaussian_noise, top_chunk, middle_chunk, bias_field, spike, ghosting.')
    args = parser.parse_args()


    # Create the necessary folders
    output_path = Path('./output') / args.save_name
    output_model_path = Path(output_path) / 'models'
    # Create all necessary folders, if they don't exist
    for f in [output_path, output_model_path]:
        os.makedirs(f, exist_ok=True)

    # Set logger
    file_handler = logging.FileHandler(output_path / 'config.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Log arguments
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args



def parse_config():
    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform = str

    # Log config
    for section in config.sections():
        logger.info("Section: %s", section)
        for options in config.options(section):
            logger.info("x %s:::%s:::%s", options,
                        config.get(section, options), str(type(options)))

    return config


args = parse_args()
config = parse_config()


