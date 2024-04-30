import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from types import SimpleNamespace
import torch
import scipy

PROJECT_ROOT = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(PROJECT_ROOT)

from src.utils import log
from scripts.inference_ranking_model import inference_ranking_model


print(torch.cuda.is_available())

# # Avery Brook Side
# runs = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac')) if 'AVERY_BROOK_SIDE' in f]
# for run in runs:
#     # Get # annotations
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         line = f.readline()
#         # get arg after --train-data-file in the line
#         train_pairs_f = line.split('--train-data-file')[1].split('--')[0].strip()
#         num_annot = len(pd.read_csv(train_pairs_f))
#         test_pairs_f = line.split('--test-data-file')[1].split('--')[0].strip()
#         test_pairs = pd.read_csv(test_pairs_f)
#         test_pairs['timestamp_1'] = pd.to_datetime(test_pairs['timestamp_1'])
#         test_pairs['timestamp_2'] = pd.to_datetime(test_pairs['timestamp_2'])
#         earliest_test_ts = min(test_pairs['timestamp_1'].min(), test_pairs['timestamp_2'].min())
        
#     # Get best epoch
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         for line in f:
#             if 'num_annot' in line:
#                 num_annot = int(line.split('=')[1])
#                 break
#     metrics_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'metrics_per_epoch*.pkl')
#     metrics_f = glob.glob(metrics_f_pattern)[0]
#     metrics = pickle.load(open(metrics_f, 'rb'))
#     best_epoch = np.argmin(metrics['val_loss'])
#     best_ckpt_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'checkpoints', 'epoch{}_*.ckpt'.format(best_epoch))
#     best_ckpt = glob.glob(best_ckpt_pattern)[0]

#     # Run inference
#     print('Running inference for {}'.format(run))
#     # make parent dir if not exists
#     os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'), exist_ok=True)
#     args = SimpleNamespace(
#         exp_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference')),
#         inference_data_file='../../../data/Streamflow/fpe_stations/Avery Brook_Side_01171000/FLOW_CFS/images.csv',
#         inference_image_root_dir='../../../data/Streamflow/fpe_stations/Avery Brook_Side_01171000/FLOW_CFS',
#         output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'),
#         train_output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run),
#         batch_size=64,
#         augment=True,
#         normalize=True,
#         gpu=2,
#         ckpt_path=best_ckpt,
#         col_label="value"
#     )
#     args.logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))

#     # get image sample mean and std from training experiment logs
#     exp_log_file = os.path.join(args.train_output_dir, "run.logs")
#     with open(exp_log_file, "r") as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             if "Computed image channelwise means" in line:
#                 args.img_sample_mean = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )
#             if "Computed image channelwise stdevs" in line:
#                 args.img_sample_std = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )

#     inference_ranking_model(args)

#     # Compute Kendall's tau on test time period only
#     inference_results_f = os.path.join(
#         args.exp_dir,
#         "inference_results_" + os.path.basename(args.inference_data_file),
#     )
#     inference_results = pd.read_csv(inference_results_f)
#     inference_results['timestamp'] = pd.to_datetime(inference_results['timestamp'])
#     inference_results = inference_results[inference_results['timestamp'] >= earliest_test_ts]
#     inference_results = inference_results.sort_values(by=['timestamp'])
#     # drop nan values in value column
#     inference_results = inference_results.dropna(subset=['value'])
#     true_value = inference_results['value'].values
#     pred_value = inference_results['scores'].values
#     tau, pvalue = scipy.stats.kendalltau(true_value, pred_value)
#     print('Kendall\'s tau for {} with {} annotations is {} (p={})'.format(run, num_annot, tau, pvalue))
    
# # West Branch Swift River
# runs = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac')) if 'WEST_BRANCH_SWIFT_RIVER' in f]
# for run in runs:
#     print('Running inference for {}'.format(run))
#     if run in [
#         'train_ranking_model_WEST_BRANCH_SWIFT_RIVER_1', 
#         'train_ranking_model_WEST_BRANCH_SWIFT_RIVER_2',
#         'train_ranking_model_WEST_BRANCH_SWIFT_RIVER_7',
#         'train_ranking_model_WEST_BRANCH_SWIFT_RIVER_9',
#         ]:
#         continue

#     # Get # annotations
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         line = f.readline()
#         # get arg after --train-data-file in the line
#         train_pairs_f = line.split('--train-data-file')[1].split('--')[0].strip()
#         num_annot = len(pd.read_csv(train_pairs_f))
#         test_pairs_f = line.split('--test-data-file')[1].split('--')[0].strip()
#         test_pairs = pd.read_csv(test_pairs_f)
#         test_pairs['timestamp_1'] = pd.to_datetime(test_pairs['timestamp_1'])
#         test_pairs['timestamp_2'] = pd.to_datetime(test_pairs['timestamp_2'])
#         earliest_test_ts = min(test_pairs['timestamp_1'].min(), test_pairs['timestamp_2'].min())
        
#     # Get best epoch
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         for line in f:
#             if 'num_annot' in line:
#                 num_annot = int(line.split('=')[1])
#                 break
#     metrics_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'metrics_per_epoch*.pkl')
#     print(metrics_f_pattern)
#     metrics_f = glob.glob(metrics_f_pattern)[0]
#     metrics = pickle.load(open(metrics_f, 'rb'))
#     best_epoch = np.argmin(metrics['val_loss'])
#     best_ckpt_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'checkpoints', 'epoch{}_*.ckpt'.format(best_epoch))
#     best_ckpt = glob.glob(best_ckpt_pattern)[0]

#     # Run inference
#     # print('Running inference for {}'.format(run))
#     # make parent dir if not exists
#     os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'), exist_ok=True)
#     args = SimpleNamespace(
#         exp_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference')),
#         inference_data_file='../../../data/Streamflow/fpe_stations/West Branch Swift River_01174565/FLOW_CFS/images.csv',
#         inference_image_root_dir='../../../data/Streamflow/fpe_stations/West Branch Swift River_01174565/FLOW_CFS',
#         output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'),
#         train_output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run),
#         batch_size=64,
#         augment=True,
#         normalize=True,
#         gpu=2,
#         ckpt_path=best_ckpt,
#         col_label="value"
#     )
#     args.logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))

#     # get image sample mean and std from training experiment logs
#     exp_log_file = os.path.join(args.train_output_dir, "run.logs")
#     with open(exp_log_file, "r") as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             if "Computed image channelwise means" in line:
#                 args.img_sample_mean = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )
#             if "Computed image channelwise stdevs" in line:
#                 args.img_sample_std = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )

#     inference_ranking_model(args)

#     # Compute Kendall's tau on test time period only
#     inference_results_f = os.path.join(
#         args.exp_dir,
#         "inference_results_" + os.path.basename(args.inference_data_file),
#     )
#     inference_results = pd.read_csv(inference_results_f)
#     inference_results['timestamp'] = pd.to_datetime(inference_results['timestamp'])
#     inference_results = inference_results[inference_results['timestamp'] >= earliest_test_ts]
#     inference_results = inference_results.sort_values(by=['timestamp'])
#     # drop nan values in value column
#     inference_results = inference_results.dropna(subset=['value'])
#     true_value = inference_results['value'].values
#     pred_value = inference_results['scores'].values
#     tau, pvalue = scipy.stats.kendalltau(true_value, pred_value)
#     print('Kendall\'s tau for {} with {} annotations is {} (p={})'.format(run, num_annot, tau, pvalue))


# # West Whately
# runs = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac')) if 'WEST_WHATELY' in f]
# for run in runs:
#     if run != 'train_ranking_model_WEST_WHATELY_5':
#         continue
#     print('Running inference for {}'.format(run))
#     # Get # annotations
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         line = f.readline()
#         # get arg after --train-data-file in the line
#         train_pairs_f = line.split('--train-data-file')[1].split('--')[0].strip()
#         num_annot = len(pd.read_csv(train_pairs_f))
#         test_pairs_f = line.split('--test-data-file')[1].split('--')[0].strip()
#         test_pairs = pd.read_csv(test_pairs_f)
#         test_pairs['timestamp_1'] = pd.to_datetime(test_pairs['timestamp_1'])
#         test_pairs['timestamp_2'] = pd.to_datetime(test_pairs['timestamp_2'])
#         earliest_test_ts = min(test_pairs['timestamp_1'].min(), test_pairs['timestamp_2'].min())
        
#     # Get best epoch
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         for line in f:
#             if 'num_annot' in line:
#                 num_annot = int(line.split('=')[1])
#                 break
#     metrics_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'metrics_per_epoch*.pkl')
#     print(metrics_f_pattern)
#     metrics_f = glob.glob(metrics_f_pattern)[0]
#     metrics = pickle.load(open(metrics_f, 'rb'))
#     best_epoch = np.argmin(metrics['val_loss'])
#     best_ckpt_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'checkpoints', 'epoch{}_*.ckpt'.format(best_epoch))
#     best_ckpt = glob.glob(best_ckpt_pattern)[0]

#     # Run inference
#     # print('Running inference for {}'.format(run))
#     # make parent dir if not exists
#     os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'), exist_ok=True)
#     args = SimpleNamespace(
#         exp_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference')),
#         inference_data_file='../../../data/Streamflow/fpe_stations/West Whately_01171005/FLOW_CFS/images.csv',
#         inference_image_root_dir='../../../data/Streamflow/fpe_stations/West Whately_01171005/FLOW_CFS',
#         output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'),
#         train_output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run),
#         batch_size=64,
#         augment=True,
#         normalize=True,
#         gpu=2,
#         ckpt_path=best_ckpt,
#         col_label="value"
#     )
#     args.logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))

#     # get image sample mean and std from training experiment logs
#     exp_log_file = os.path.join(args.train_output_dir, "run.logs")
#     with open(exp_log_file, "r") as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             if "Computed image channelwise means" in line:
#                 args.img_sample_mean = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )
#             if "Computed image channelwise stdevs" in line:
#                 args.img_sample_std = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )

#     inference_ranking_model(args)

#     # Compute Kendall's tau on test time period only
#     inference_results_f = os.path.join(
#         args.exp_dir,
#         "inference_results_" + os.path.basename(args.inference_data_file),
#     )
#     inference_results = pd.read_csv(inference_results_f)
#     inference_results['timestamp'] = pd.to_datetime(inference_results['timestamp'])
#     inference_results = inference_results[inference_results['timestamp'] >= earliest_test_ts]
#     inference_results = inference_results.sort_values(by=['timestamp'])
#     # drop nan values in value column
#     inference_results = inference_results.dropna(subset=['value'])
#     true_value = inference_results['value'].values
#     pred_value = inference_results['scores'].values
#     tau, pvalue = scipy.stats.kendalltau(true_value, pred_value)
#     print('Kendall\'s tau for {} with {} annotations is {} (p={})'.format(run, num_annot, tau, pvalue))


# # West Brook Lower
# runs = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac')) if 'WEST_BROOK_LOWER' in f]
# for run in runs:
#     if run in [
#         'train_ranking_model_WEST_BROOK_LOWER_1',
#         # 'train_ranking_model_WEST_BROOK_LOWER_2'
#         'train_ranking_model_WEST_BROOK_LOWER_3',
#         'train_ranking_model_WEST_BROOK_LOWER_4',
#         'train_ranking_model_WEST_BROOK_LOWER_5',
#         ]:
#         continue
#     print('Running inference for {}'.format(run))

#     # Get # annotations
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         line = f.readline()
#         # get arg after --train-data-file in the line
#         train_pairs_f = line.split('--train-data-file')[1].split('--')[0].strip()
#         num_annot = len(pd.read_csv(train_pairs_f))
#         test_pairs_f = line.split('--test-data-file')[1].split('--')[0].strip()
#         test_pairs = pd.read_csv(test_pairs_f)
#         test_pairs['timestamp_1'] = pd.to_datetime(test_pairs['timestamp_1'])
#         test_pairs['timestamp_2'] = pd.to_datetime(test_pairs['timestamp_2'])
#         earliest_test_ts = min(test_pairs['timestamp_1'].min(), test_pairs['timestamp_2'].min())
        
#     # Get best epoch
#     params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
#     params_f = glob.glob(params_f_pattern)[0]
#     with open(params_f, 'r') as f:
#         for line in f:
#             if 'num_annot' in line:
#                 num_annot = int(line.split('=')[1])
#                 break
#     metrics_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'metrics_per_epoch*.pkl')
#     print(metrics_f_pattern)
#     metrics_f = glob.glob(metrics_f_pattern)[0]
#     metrics = pickle.load(open(metrics_f, 'rb'))
#     best_epoch = np.argmin(metrics['val_loss'])
#     best_ckpt_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'checkpoints', 'epoch{}_*.ckpt'.format(best_epoch))
#     best_ckpt = glob.glob(best_ckpt_pattern)[0]

#     # Run inference
#     # print('Running inference for {}'.format(run))
#     # make parent dir if not exists
#     os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'), exist_ok=True)
#     args = SimpleNamespace(
#         exp_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference')),
#         inference_data_file='../../../data/Streamflow/fpe_stations/West Brook Lower_01171090/FLOW_CFS/images.csv',
#         inference_image_root_dir='../../../data/Streamflow/fpe_stations/West Brook Lower_01171090/FLOW_CFS',
#         output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'),
#         train_output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run),
#         batch_size=64,
#         augment=True,
#         normalize=True,
#         gpu=2,
#         ckpt_path=best_ckpt,
#         col_label="value"
#     )
#     args.logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))

#     # get image sample mean and std from training experiment logs
#     exp_log_file = os.path.join(args.train_output_dir, "run.logs")
#     with open(exp_log_file, "r") as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             if "Computed image channelwise means" in line:
#                 args.img_sample_mean = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )
#             if "Computed image channelwise stdevs" in line:
#                 args.img_sample_std = np.array(
#                     [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
#                 )

#     inference_ranking_model(args)

#     # Compute Kendall's tau on test time period only
#     inference_results_f = os.path.join(
#         args.exp_dir,
#         "inference_results_" + os.path.basename(args.inference_data_file),
#     )
#     inference_results = pd.read_csv(inference_results_f)
#     inference_results['timestamp'] = pd.to_datetime(inference_results['timestamp'])
#     inference_results = inference_results[inference_results['timestamp'] >= earliest_test_ts]
#     inference_results = inference_results.sort_values(by=['timestamp'])
#     # drop nan values in value column
#     inference_results = inference_results.dropna(subset=['value'])
#     true_value = inference_results['value'].values
#     pred_value = inference_results['scores'].values
#     tau, pvalue = scipy.stats.kendalltau(true_value, pred_value)
#     print('Kendall\'s tau for {} with {} annotations is {} (p={})'.format(run, num_annot, tau, pvalue))


# West Brook Upper
runs = [f for f in os.listdir(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac')) if 'WEST_BROOK_UPPER' in f]
for run in runs:
    if run in [
        'train_ranking_model_WEST_BROOK_UPPER_1',
    ]:
        continue
    print('Running inference for {}'.format(run))

    # Get # annotations
    params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
    params_f = glob.glob(params_f_pattern)[0]
    with open(params_f, 'r') as f:
        line = f.readline()
        # get arg after --train-data-file in the line
        train_pairs_f = line.split('--train-data-file')[1].split('--')[0].strip()
        num_annot = len(pd.read_csv(train_pairs_f))
        test_pairs_f = line.split('--test-data-file')[1].split('--')[0].strip()
        test_pairs = pd.read_csv(test_pairs_f)
        test_pairs['timestamp_1'] = pd.to_datetime(test_pairs['timestamp_1'])
        test_pairs['timestamp_2'] = pd.to_datetime(test_pairs['timestamp_2'])
        earliest_test_ts = min(test_pairs['timestamp_1'].min(), test_pairs['timestamp_2'].min())
        
    # Get best epoch
    params_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'params.txt')
    params_f = glob.glob(params_f_pattern)[0]
    with open(params_f, 'r') as f:
        for line in f:
            if 'num_annot' in line:
                num_annot = int(line.split('=')[1])
                break
    metrics_f_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'metrics_per_epoch*.pkl')
    print(metrics_f_pattern)
    metrics_f = glob.glob(metrics_f_pattern)[0]
    metrics = pickle.load(open(metrics_f, 'rb'))
    best_epoch = np.argmin(metrics['val_loss'])
    best_ckpt_pattern = os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, 'checkpoints', 'epoch{}_*.ckpt'.format(best_epoch))
    best_ckpt = glob.glob(best_ckpt_pattern)[0]

    # Run inference
    # print('Running inference for {}'.format(run))
    # make parent dir if not exists
    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'), exist_ok=True)
    args = SimpleNamespace(
        exp_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference')),
        inference_data_file='../../../data/Streamflow/fpe_stations/West Brook Upper_01171030/FLOW_CFS/images.csv',
        inference_image_root_dir='../../../data/Streamflow/fpe_stations/West Brook Upper_01171030/FLOW_CFS',
        output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run, run.replace('train', 'inference'), f'epoch{best_epoch}_inference'),
        train_output_dir=os.path.join(PROJECT_ROOT, 'results', 'vary_annot_frac', run),
        batch_size=64,
        augment=True,
        normalize=True,
        gpu=2,
        ckpt_path=best_ckpt,
        col_label="value"
    )
    args.logger = log(log_file=os.path.join(args.exp_dir, "run.logs"))

    # get image sample mean and std from training experiment logs
    exp_log_file = os.path.join(args.train_output_dir, "run.logs")
    with open(exp_log_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if "Computed image channelwise means" in line:
                args.img_sample_mean = np.array(
                    [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
                )
            if "Computed image channelwise stdevs" in line:
                args.img_sample_std = np.array(
                    [float(x) for x in line.split(":")[-1].strip()[1:-1].split()]
                )

    inference_ranking_model(args)

    # Compute Kendall's tau on test time period only
    inference_results_f = os.path.join(
        args.exp_dir,
        "inference_results_" + os.path.basename(args.inference_data_file),
    )
    inference_results = pd.read_csv(inference_results_f)
    inference_results['timestamp'] = pd.to_datetime(inference_results['timestamp'])
    inference_results = inference_results[inference_results['timestamp'] >= earliest_test_ts]
    inference_results = inference_results.sort_values(by=['timestamp'])
    # drop nan values in value column
    inference_results = inference_results.dropna(subset=['value'])
    true_value = inference_results['value'].values
    pred_value = inference_results['scores'].values
    tau, pvalue = scipy.stats.kendalltau(true_value, pred_value)
    print('Kendall\'s tau for {} with {} annotations is {} (p={})'.format(run, num_annot, tau, pvalue))