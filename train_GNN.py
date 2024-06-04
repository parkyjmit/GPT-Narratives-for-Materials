import argparse
from trainer import train_run
import torch
import os
import logging
from datetime import datetime
from data_load.data_load import JARVIS_dataloader
import wandb


def main(args):
    args.device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    directory = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + args.exp_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(format='%(asctime)s %(message)s', filename=os.path.join(directory, f"{args.exp_name}.log"), level=logging.DEBUG)
    logger = logging.getLogger()
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    args.directory = directory
    logging.info(args)
    
    # wandb setting
    wandb.login()
    wandb.init(project='lightning_logs', name=directory, config=args)

    # Prepare dataset
    train_loader, valid_loader, test_loader, indices = JARVIS_dataloader(args)

    # Train model
    train_run(args, directory, train_loader, valid_loader, test_loader, indices[2], pretrain=False)
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prediction training')
    """Experiment setting."""
    parser.add_argument('--exp-name', type=str, default='oqmd_3d_total energy', help="")
    parser.add_argument('--task', type=str, default='regression', help="classification, regression")
    parser.add_argument('--num-workers', type=int, default=12, help="")
    parser.add_argument('--dataset', type=str, default='oqmd_3d', help="mp_3d_2020, dft_3d, oqmd_3d, aflow2")
    parser.add_argument('--streaming', type=bool, default=False, help="")
    parser.add_argument('--shuffle', type=bool, default=True, help="Shuffle or not. for inference, shuffle=False")
    parser.add_argument('--target', type=str, default='total energy', help="")
    parser.add_argument('--epochs', type=int, default=100, help="")
    parser.add_argument('--num-train', type=int, default=0.9, help="")
    parser.add_argument('--num-valid', type=int, default=0.05, help="")
    parser.add_argument('--num-test', type=int, default=0.05, help="")
    parser.add_argument('--batch-size', type=int, default=16, help="")
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

    # Learning
    main(args)
