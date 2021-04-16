'''
Same as eval_hierarchal.py but the first stage is using the blind model
And the second stage is using the content based grader
'''

'''
Evaluation Approach (Always using model ensembles)
Use A1-C2 models to get grade predictions
Choose a threshold k
For all datapoints with predictions > k
Get new prediction using B2-C2 models
Caluclate MSE for all predictions
Repeat for all k to get a plot of MSE vs threshold k
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep_blind_content import get_data
import sys
import os
import argparse
from tools import AverageMeter, get_default_device, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg
from models import BERTGrader
import statistics
import matplotlib.pyplot as plt
import numpy as np
from eval_ensemble import eval
from eval_hierarchal import get_ensemble_preds, apply_hierarchal

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELSA', type=str, help='Blind trained .th models for first stage separated by space')
    commandLineParser.add_argument('MODELSB', type=str, help='Content trained .th models for second stage separated by space')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('TEST_PROMPTS', type=str, help='test data prompts mlf')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")

    args = commandLineParser.parse_args()
    model_pathsA = args.MODELSA
    model_pathsA = model_pathsA.split()
    model_pathsB = args.MODELSB
    model_pathsB = model_pathsB.split()
    test_data_file = args.TEST_DATA
    test_grades_files = args.TEST_GRADES
    test_prompts_mlf = args.TEST_PROMPTS
    batch_size = args.B

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_hierarchal_blind_content.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data as tensors
    input_ids_content, input_ids_blind, mask_content, mask_blind, token_ids_content, token_ids_blind, labels = get_data(test_data_file, test_grades_files, test_prompts_mlf, 0.0)
    test_ds_content = TensorDataset(input_ids_content, mask_content, token_ids_content, labels)
    test_dl_content = DataLoader(test_ds_content, batch_size=batch_size)
    test_ds_blind = TensorDataset(input_ids_blind, mask_blind, token_ids_blind, labels)
    test_dl_blind = DataLoader(test_ds_blind, batch_size=batch_size)

    # Load the models
    modelsA = []
    for model_path in model_pathsA:
        model = BERTGrader()
        model.load_state_dict(torch.load(model_path))
        modelsA.append(model)

    modelsB = []
    for model_path in model_pathsB:
        model = BERTGrader()
        model.load_state_dict(torch.load(model_path))
        modelsB.append(model)

    targets = None
    all_predsA = []
    all_predsB = []

    for model in modelsA:
        preds, targets = eval(test_dl_blind, model)
        all_predsA.append(preds)

    for model in modelsB:
        preds, targets = eval(test_dl_content, model)
        all_predsB.append(preds)

    predsA = get_ensemble_preds(all_predsA)
    predsB = get_ensemble_preds(all_predsB)

    ks = []
    rmses = []
    rmses_ref = []
    ref = calculate_mse(torch.FloatTensor(predsA), torch.FloatTensor(targets)).item()
    ref = ref ** 0.5

    for k in np.linspace(0, 6, 60):
        preds = apply_hierarchal(predsA, predsB, thresh=k)
        mse = calculate_mse(torch.FloatTensor(preds), torch.FloatTensor(targets)).item()
        rmse = mse**0.5
        ks.append(k)
        rmses.append(rmse)
        rmses_ref.append(ref)

    # Plot
    filename = 'rmse_vs_k.png'
    plt.plot(ks, rmses_ref, label="Baseline")
    plt.plot(ks, rmses, label="Hierarchical")
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
