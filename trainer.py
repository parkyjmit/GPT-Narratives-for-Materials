import os
import sys
import pickle
import logging
from tqdm import tqdm
import torch
from torch import optim
import numpy as np
from time import time

from model.edieggc import EDiEGGC
import wandb

torch.manual_seed(1234)
np.random.seed(1234)


def train_run(args, directory, train_loader, valid_loader, test_loader, test_indices, pretrain):
    # Prepare model
    model = EDiEGGC(args)
    model.to(args.device)

    logging.info('Train start')
    start_time = time()

    history_file = os.path.join(directory, 'history_' + args.exp_name + '.pickle')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif args.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = args.warmup_steps / (args.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif args.scheduler == "step":
        # pct_start = args.warmup_steps / (args.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )
    elif args.scheduler == "plateau":
        # pct_start = args.warmup_steps / (args.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=5,
        )
    early_stopping = EarlyStopping(patience=15)

    history = {'train': [], 'validation': [], 'test': []}
    history['test_indices'] = test_indices
    min_train_loss = torch.inf
    min_valid_loss = torch.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        # Train
        train_loss = evaluate_model(args, args.device, model, train_loader, optimizer, 'train') / args.num_train
        history['train'].append(train_loss)

        # Validation
        valid_loss = evaluate_model(args, args.device, model, valid_loader, optimizer, 'valid') / args.num_valid
        history['validation'].append(valid_loss)

        if args.scheduler == 'plateau':
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), os.path.join(directory, args.exp_name))
            min_train_loss = train_loss
            min_valid_loss = valid_loss
            best_epoch = epoch
            save_history(history, history_file)

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            logging.info(early_stopping.message)
            break

        logging.info(f'Epoch {epoch + 1}: Train RMSE: {train_loss:.5f}, Validation MAE: {valid_loss:.5f}  '
              f'Time elapsed: {(time() - start_time)/3600:.5f}')
        wandb.log({"Epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

    logging.info(f'Best result at epoch: {best_epoch}, '
          f'Train RMSE: {min_train_loss:.5f}, Validation MAE: {min_valid_loss:.5f}')

    end_time = time()
    # Test
    model.load_state_dict(torch.load(os.path.join(directory, args.exp_name)))
    test_loss = evaluate_model(args, args.device, model, test_loader, optimizer, 'test') / args.num_test
    
    history['test'].append(test_loss)
    save_history(history, history_file)
    logging.info(f'Test MAE: {test_loss:.5f} ')
    wandb.log({"test_loss": test_loss})
    logging.info(f'Time elapsed: {(end_time - start_time)/3600:.5f}')

def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


def evaluate_model(args, device, model, loader, optimizer, split):
    if split == 'train':
        running_loss = 0.0
        model.train()
        for g, lg, label in tqdm(loader):
            g = g.to(device)
            lg = lg.to(device)
            label = label.to(device)
            loss = calculate_loss(args, model, g, lg, label, optimizer, split)
            running_loss += loss * g.batch_size
        return running_loss
    else:
        running_loss = 0.0
        model.eval()

        for g, lg, label in tqdm(loader):
            g = g.to(device)
            lg = lg.to(device)
            label = label.to(device)
            with torch.no_grad():
                loss_g = calculate_loss(args, model, g, lg, label, optimizer, split)
                running_loss += loss_g * g.batch_size
        return running_loss


def calculate_loss(args, model, g, lg, label, optimizer, split):
    pred, _, _, _, _ = model(g, lg)
    if split == 'train':
        if args.task == 'regression':
            loss = torch.sqrt(torch.nn.functional.mse_loss(pred, label))
        elif args.task == 'classification':
            loss = torch.nn.functional.cross_entropy(pred, label.type(torch.int64))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
    else:
        if args.task == 'regression':
            loss = torch.nn.functional.l1_loss(pred, label)
        elif args.task == 'classification':
            loss = torch.nn.functional.cross_entropy(pred, label.type(torch.int64))
            # wandb.log({"roc": wandb.plot.roc_curve(label.cpu(), pred.cpu(), labels=args.label_list)})
    return loss.item()



class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.message = ''

    def __call__(self, val_loss):
        if val_loss != val_loss:
            self.early_stop = True
            self.message = 'Early stopping: NaN appear'
        elif self.best_score is None:
            self.best_score = val_loss
        elif self.best_score < val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.message = 'Early stopping: No progress'
        else:
            self.best_score = val_loss
            self.counter = 0