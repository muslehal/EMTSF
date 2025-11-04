
import sys
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import numpy as np
import pandas as pd
import time
import random
import argparse
import datetime
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# Import other modules first
from src.learner import Learner

from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls


from models import EMTSF, model_b, model_c, Model,model_d#, model_a


# Argument Parsing
parser = argparse.ArgumentParser()


# IntegratedModel   model1 model2 dlinear
parser.add_argument('--dset', type=str, default='ettm1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64    , help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')

parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=1, help='whether to use time features or not')

# Patch
parser.add_argument('--patch_len', type=int, default=32, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

# Model args 
parser.add_argument('--n_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
#parser.add_argument('--d_ff', type=in720t, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')

# Optimization args
parser.add_argument('--n_epochs', type=int, default=2, help='number of training epochs')


parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=2, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

parser.add_argument('--n1',type=int,default=256,help='First Embedded representation')#256
parser.add_argument('--n2',type=int,default=256,help='Second Embedded representation')
parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')#256
parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
parser.add_argument('--e_fact', type=int, default=2, help='expand factor parameter of Mamba')
parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')

parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token') 
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--momentum', type=float, default=0.1, help='momentum')
parser.add_argument('--dp_rank', type=int,default = 8)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--merge_size',type=int,default = 2)
parser.add_argument('--task_name', type=str, default='short_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually') 
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
#parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--label_len', type=int, default=48, help='start token length')

parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')

parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')


# Dataset and dataloader
parser.add_argument('--features2', type=int, default=7, help='Each prediction includes 7 features')



parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

# Early Stopping arguments are already added
parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped')
parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum change in the monitored metric to qualify as an improvement')
parser.add_argument('--monitor', type=str, default='valid_loss', help='Metric to be monitored for early stopping (e.g., val_loss, val_accuracy)')

# easy config modification
parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
#parser.add_argument('--data-path', type=str, help='path to dataset')
#parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--pretrained',
                    help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used (deprecated!)')
parser.add_argument('--output', default='output', type=str, metavar='PATH',
                    help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# distributed training
parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

# for acceleration
parser.add_argument('--fused_window_process', action='store_true',
                    help='Fused window shift & window partition, similar for reversed part.')
parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
parser.add_argument('--optim', type=str,
                    help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
parser.add_argument('--hidden', type=int, default=7,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--save_path', type=str, default='saved_models', help='Path to save the model')
parser.add_argument('--save_model_name', type=str, default='model.pth', help='Model name to save')


args, unparsed = parser.parse_known_args()

# Set default values if not provided
if args.save_path is None:
    args.save_path = 'saved_models/' + args.dset
if args.save_model_name is None:
    args.save_model_name = f"{args.model_name2}_cw{args.context_points}_tw{args.target_points}_patch{args.patch_len}_stride{args.stride}_epochs{args.n_epochs}_model{args.model_id}"

# Ensure the save directory exists
os.makedirs(args.save_path, exist_ok=True)

configs = args  # Assuming you use args directly as configs
def get_model(c_in, args, model_type='model_a', freeze_experts=True):
    model_type = model_type.lower()

    if model_type == 'model_a':
        model = Model(args, c_in)
    elif model_type == 'model_b':
        model = model_b(args, c_in)
    elif model_type == 'model_c':
        model = model_c(args, c_in)
    elif model_type == 'model_d':
        model = model_d(args, c_in)
    elif model_type == 'emtsf':
        model = EMTSF(args, c_in, num_experts_a=1, num_experts_b=1, num_experts_c=1, freeze_experts=freeze_experts)
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model


def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)

def find_lr():
    # Get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)


    # Get loss
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')
    # loss_func = combined_loss
    
    # Get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # Define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)  # cbs=cbs                      

    # Fit the data to the model
    return learn.lr_finder()

def train_func(model_type='EMTSF', lr=args.lr):
    # Get dataloader
    dls = get_dls(args)
    print('in out', dls.vars, dls.c, dls.len)



    # Get model
    if model_type == 'EMTSF':
        # Initially freeze experts
        model = get_model(dls.vars, args, model_type=model_type, freeze_experts=True)
    else:
        model = get_model(dls.vars, args, model_type=model_type)
    print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Get loss function
    #loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')


    # Get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
        SaveModelCB(monitor='valid_loss', fname=args.save_model_name, path=args.save_path)
    ]

    
    # Define learner
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[mse, mae]
    )

    # Training loop
    if model_type == 'EMTSF':
        # Train with frozen experts
        learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)

        # Unfreeze experts
        #print('Fine-tuning the entire model')
        learn.unfreeze()
        # Optionally, set a lower learning rate for fine-tuning
        lr_finetune = lr / 10
        learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr_finetune, pct_start=0.2)
    else:
        # Fit the data to the model
        learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)

    torch.save(model.state_dict(), os.path.join(args.save_path, args.save_model_name + f"{model_type}.pth"))

    
def test_func(model_type='model_c'):
     #Define the weight path (hardcoded for now, ensure it's correct)
    weight_path = os.path.join(args.save_path, args.save_model_name + f"{model_type}.pth")#'.pth') # os.path.join(args.save_path, args.save_model_name + '.pth')# "saved_models/weather/target_96/model_weather_epochs20_context512_target96.pth"
    #weight_path = os.path.join(args.save_path, args.save_model_name + ".pth")
    #weight_path = "saved_models/illness/target_24/model_illness_epochs100_context96_target24.pth" #"saved_models/weather/target_720/model_weather_epochs20_context512_target720model_c.pth"
    # Get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, model_type=model_type)
    
    # Load the model weights
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # Get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    learn = Learner(dls, model, cbs=cbs)

    # Run the test
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])

    # Extract predictions, targets, and scores
    predictions, targets, scores = out

    # Check the type of predictions and targets
    print(f"Type of predictions: {type(predictions)}")
    print(f"Type of targets: {type(targets)}")

    # Directly use predictions and targets as NumPy arrays
    predictions_np = predictions if isinstance(predictions, np.ndarray) else predictions.cpu().numpy()
    targets_np = targets if isinstance(targets, np.ndarray) else targets.cpu().numpy()

    # Flatten the arrays if necessary or reshape as needed
    predictions_flat = predictions_np.reshape(-1, predictions_np.shape[-1])
    targets_flat = targets_np.reshape(-1, targets_np.shape[-1])

    # Combine predictions and targets
    results = np.concatenate((predictions_flat, targets_flat), axis=1)

    # Create a DataFrame
    num_features = predictions_np.shape[-1]
    columns = [f'Predicted_Feature_{i}' for i in range(num_features)] + \
              [f'Actual_Feature_{i}' for i in range(num_features)]
    df = pd.DataFrame(results, columns=columns)

    # Extract scores using list indices
    mse_score = scores[0]  # Assuming mse is the first score
    mae_score = scores[1]  # Assuming mae is the second score

    # Add scores to the DataFrame
    df['MSE'] = mse_score
    df['MAE'] = mae_score

    # # Save to CSV
    # csv_filename = os.path.join(args.save_path, f"{args.save_model_name}_test_results.csv")
    # df.to_csv(csv_filename, index=False)
    # print(f"Testing results and metrics saved to {csv_filename}")

    return out


def plot_feature_actual_vs_predicted(actual, predicted, feature_idx):
    """
    Plot the actual vs predicted values for a specific feature for the first sequence.

    Parameters:
    - actual (np.array or torch.Tensor): Array of actual values.
    - predicted (np.array or torch.Tensor): Array of predicted values.
    - feature_idx (int): Index of the feature to plot.
    """
   
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()

    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Select the first sequence for the given feature index
    actual_feature = actual[0, :, feature_idx]
    predicted_feature = predicted[0, :, feature_idx]

    # Plot the first sequence
    plt.figure(figsize=(10, 6))
    plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')
    plt.title(f"Actual vs Predicted for Feature {feature_idx}, Sequence 0")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    if args.is_train:
        # Train model_a
        suggested_lr = find_lr()
        print('Suggested LR for model_a:', suggested_lr)
        def count_parameters(model_a): # count number of trainable parameters in the model
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_func(model_type='model_a', lr=suggested_lr)

        #train model_b
        suggested_lr = find_lr()
        print('Suggested LR for model_b:', suggested_lr)
        def count_parameters(model_b): # count number of trainable parameters in the model
             return sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_func(model_type='model_b', lr=suggested_lr)
        
        # Train model_c
        suggested_lr = find_lr()
        print('Suggested LR for model_c:', suggested_lr)
        def count_parameters(model_c): # count number of trainable parameters in the model
             return sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_func(model_type='model_c', lr=suggested_lr)

        # Train model_b
        suggested_lr = find_lr()
        print('Suggested LR for model_d:', suggested_lr)
        def count_parameters(model_d): # count number of trainable parameters in the model
             return sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_func(model_type='model_d', lr=suggested_lr)


       # Now train the ensemble model (EMTSF)
        suggested_lr = find_lr()
        print('Suggested LR for EMTSF:', suggested_lr)
        def count_parameters(EMTSF): # count number of trainable parameters in the model
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_func(model_type='EMTSF', lr=suggested_lr)
    else:
        # Testing mode
        out = test_func(model_type='model_b')
       
        #out = test_func(model_type='EMTSF')
        print('Score:', out[2])

        print('Shape:', out[0].shape)
        
        # for feature_idx in range(7):  # Adjust based on the number of features
        #     plot_feature_actual_vs_predicted(out[1], out[0], feature_idx)

    print('----------- Complete! -----------')









