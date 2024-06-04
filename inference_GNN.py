from data_load.data_load import JARVIS_dataloader
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from model.edieggc import EDiEGGC
from tqdm import tqdm
from sklearn.metrics import r2_score
import pickle


def inference_run(args):
    # load data for inference, shuffle=False. no need target
    train_loader, valid_loader, test_loader, indices = JARVIS_dataloader(args)
    
    # load model
    if args.task == 'classification':
        with open(f'{args.inference_model.split('/')[0]}/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        args.label_list = list(label_encoder.classes_)
    model = EDiEGGC(args)
    model.load_state_dict(torch.load(args.inference_model))
    model.to(args.device)
    model.eval()

    # inference
    preds = []
    for g, lg, label in tqdm(train_loader):
        g = g.to(args.device)
        lg = lg.to(args.device)
        with torch.no_grad():
            pred, _, _, _, _ = model(g, lg)
        preds.append(pred.detach().cpu().numpy())
    if args.task == 'regression':
        preds = np.hstack(preds)
    elif args.task == 'classification':
        preds = np.vstack(preds)
        preds = np.argmax(preds, axis=1)
        preds = np.array(args.label_list)[preds]

    # save inference results
    df = pd.read_hdf(f'data_load/{args.dataset}_materials.h5', key=args.dataset)
    df[args.inference_target] = preds   
    df.to_hdf(f'data_load/{args.dataset}_materials.h5', key=args.dataset)


if __name__ == '__main__':
    # load parsers and original data to plot test data prediction results
    parser = argparse.ArgumentParser(description='prediction training')
    """Experiment setting."""
    parser.add_argument('--exp-name', type=str, default='', help="")
    parser.add_argument('--seed', type=int, default=42, help="")
    parser.add_argument('--inference_model', type=str, default='', help="Torch model state path")
    parser.add_argument('--inference_target', type=str, default='formation_energy_peratom', help="Target property for inference")
    parser.add_argument('--task', type=str, default='regression', help="Inference task", choices=['regression', 'classification'])
    parser.add_argument('--num-workers', type=int, default=6, help="")
    parser.add_argument('--dataset', type=str, default='oqmd', help="mp_3d_2020, dft_3d, oqmd, aflow2", choices=['mp_3d_2020', 'dft_3d', 'oqmd', 'aflow2'])
    parser.add_argument('--streaming', type=bool, default=False, help="")
    parser.add_argument('--shuffle', type=bool, default=False, help="Shuffle or not. for inference, shuffle=False")
    parser.add_argument('--target', type=str, default='pseudo target', help="To load dataloader")
    parser.add_argument('--epochs', type=int, default=300, help="")
    parser.add_argument('--num-train', type=int, default=1, help="")
    parser.add_argument('--num-valid', type=int, default=0, help="")
    parser.add_argument('--num-test', type=int, default=0, help="")
    parser.add_argument('--batch-size', type=int, default=64, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="")
    parser.add_argument('--weight-decay', type=float, default=0, help="")
    parser.add_argument('--max-norm', type=float, default=1000.0, help="")
    parser.add_argument('--scheduler', type=str, default='plateau', help="")
    parser.add_argument('--cutoff', type=float, default=5.0, help="")
    parser.add_argument('--device', type=str, default='cuda', help="cuda device")
    '''Model setting'''
    parser.add_argument('--embedding-type', type=str, default='cgcnn', help="")
    parser.add_argument('--alignn-layers', type=int, default=4, help="")
    parser.add_argument('--gcn-layers', type=int, default=4, help="")
    parser.add_argument('--atom-input-features', type=int, default=92, help="")
    parser.add_argument('--edge-input-features', type=int, default=80, help="")
    parser.add_argument('--triplet-input-features', type=int, default=40, help="")
    parser.add_argument('--embedding-features', type=int, default=64, help="")
    parser.add_argument('--hidden-features', type=int, default=256, help="")
    parser.add_argument('--output-features', type=int, default=1, help="")
    parser.add_argument('--link', type=str, default='identity', help="")
    args = parser.parse_args()

    inference_run(args)