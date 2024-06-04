from torch.utils.data import DataLoader
from jarvis.core.graphs import compute_bond_cosines
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import dgl
import random
import logging
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed

# set random seed
torch.manual_seed(123)
np.random.seed(123)


def compute_d_u(edges):
    # r = edges.dst['pos'] - edges.src['pos']
    r = edges.data.pop('r')
    d = torch.norm(r, dim=1)
    u = r/d[:, None]
    return {'r': r, 'd': d, 'u': u}


def prepare_line_dgl(g):
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    lg.ndata.pop('r')
    lg.ndata.pop('d')
    lg.ndata.pop('u')
    return lg


def jatoms_to_graph(atoms):
    structure = Atoms.from_dict(dict(atoms))
    graph = Graph.atom_dgl_multigraph(structure, compute_line_graph=False, atom_features='cgcnn')
    graph.apply_edges(compute_d_u)
    return graph


class JARVISDataset(torch.utils.data.Dataset):
    '''
    Torch dataset for public database provided by JARVIS.
    '''
    def __init__(self, df, args):
        super().__init__()
        self.graphs = []
        self.graphs_line = []
        self.labels = []
        self.df = df
        self.streaming = args.streaming

        if not self.streaming:
            # load dataframe from csv and split into num_workers
            df_split = np.array_split(df, args.num_workers*4)

            # get graphs in parallel
            def get_graphs(split):
                '''
                split: pandas dataframe
                return: list of dgl.graphs
                '''
                return [jatoms_to_graph(d) for d in tqdm(split['atoms'])]
            pickle_list = Parallel(n_jobs=args.num_workers*4)(delayed(get_graphs)(split) for split in df_split)
            for g_list in pickle_list:
                self.graphs += g_list

            # build line graphs
            logging.info('Building line graphs')
            for i in tqdm(range(len(self.graphs))):
                lg = prepare_line_dgl(self.graphs[i])
                self.graphs_line.append(lg)
            
            del df_split
        
        # get labels according to task
        if args.task == 'classification':
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(df[args.target].values)
            args.label_list = list(label_encoder.classes_)
            with open(f'{args.directory}/label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
        elif args.task == 'regression':
            self.labels = df[args.target].values


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        if self.streaming:
            d = self.df.iloc[index]
            g = jatoms_to_graph(d['atoms'])
            lg = prepare_line_dgl(g)
        else:   
            g = self.graphs[index]
            lg = self.graphs_line[index]
        labels = self.labels[index]
        return g, lg, labels
    
    @staticmethod
    def collate(samples):
        g, lg, labels = map(list, zip(*samples))
        g = dgl.batch(g)
        lg = dgl.batch(lg)
        return g, lg, torch.tensor(labels, dtype=torch.float32)


def JARVIS_dataloader(args):
    '''
    Torch dataloader.
    :param args: arguments
    :return: graph dataloader
    '''

    # prepare DataFrame. Reason using hdf is to avoid string type in csv.
    df = pd.read_hdf(f'data_load/{args.dataset}_materials.h5', key=f'{args.dataset}')    
    df.dropna(subset=args.target, inplace=True)
    # df = df.iloc[:500]  # for debug

    args.num_train = int(len(df) * args.num_train)
    args.num_valid = int(len(df) * args.num_valid)
    args.num_test = int(len(df) * args.num_test)
    num = args.num_train + args.num_valid + args.num_test
    idx = list(range(num))
    if args.shuffle:
        random.seed(123)
        random.shuffle(idx)
    train_indices = idx[0:args.num_train]
    valid_indices = idx[args.num_train:args.num_train + args.num_valid]
    test_indices = idx[args.num_train + args.num_valid:args.num_train + args.num_valid + args.num_test]

    logging.info('Prepare train/validation/test data')
    data = JARVISDataset(df, args)
    collate_fn = data.collate
    del df

    train_data = torch.utils.data.Subset(data, train_indices)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    valid_data = torch.utils.data.Subset(data, valid_indices)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    test_data = torch.utils.data.Subset(data, test_indices)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, (train_indices, valid_indices, test_indices)