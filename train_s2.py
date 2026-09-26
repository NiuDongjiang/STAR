
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from graph_constructor import GraphDataset, collate_fn
from STAR_s2 import DTIPredictor

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
from tqdm import tqdm
from config.config_dict import *
from log.train_logger import *
from utils import *
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
import dgl
print(dgl.__version__)
print(dgl.backend.backend_name)
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
def c_index(y_true, y_pred):
    summ = 0
    pair = 0
    n = len(y_true)
    for i in range(1, n):
        for j in range(i):
            if y_true[i] > y_true[j]:
                pair += 1
                summ += (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                pair += 1
                summ += (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
    return summ / pair if pair != 0 else 0

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def SD(y_true, y_pred):
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_fit = lr.predict(y_pred)
    residual = y_true - y_fit
    return np.sqrt(np.sum(np.square(residual)) / (len(y_true) - 1))

# %%
def val(model, dataloader, device):
    model.eval()
    #tbar = tqdm(dataloader, total=len(dataloader), ncols=80)
    pred_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl= model(bg, device)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae_val = MAE(label, pred)
    corr_val = CORR(label, pred)
    cindex_val = c_index(label, pred)
    sd_val = SD(label, pred)
    model.train()

    return rmse, pr, mae_val, corr_val, cindex_val, sd_val

if __name__ == '__main__':
    cfg = 'TrainConfig'
    config = Config(cfg)
    args = config.get_config()
    print(args)
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")

    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'v2020-other-PL')
        valid_dir = os.path.join(data_root, 'v2020-other-PL')
        test_dir = os.path.join(data_root, 'v2020-other-PL')
        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))

        train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        test_set = GraphDataset(test_dir, test_df, graph_type=graph_type, create=False)
        val_set = GraphDataset(val_dir, val_df, graph_type=graph_type, create=False)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

        test2013_dir = os.path.join('/data/CASF-2013-updated/CASF-2013/coreset')
        test2016_dir = os.path.join('/data/CASF-2016/CASF-2016/coreset')
        testhiq_dir = os.path.join('/data/CSAR_HIQ')

        test2013_df = pd.read_csv(os.path.join('/data/test_2013.csv'))
        test2016_df = pd.read_csv(os.path.join('/data/test_2016.csv'))
        testhiq_df = pd.read_csv(os.path.join('/data/test_hiq.csv'))

        test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type='Graph', create=False)
        test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type='Graph', create=False)
        testhiq_set = GraphDataset(testhiq_dir, testhiq_df, graph_type='Graph', create=False)

        test2013_loader = DataLoader(test2013_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
        test2016_loader = DataLoader(test2016_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
        testhiq_loader = DataLoader(testhiq_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"test data: {len(test_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"testhiq data: {len(testhiq_set)}")
        print(torch.cuda.device_count())
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        activation_funcs = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, tau= 0.5, layer_num=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        # start training
        model.train()

        for epoch in range(epochs):
            #tbar = tqdm(train_loader, total=len(train_loader), ncols=100)
            for data in train_loader:
                bg, label = data
                bg = bg.to(device)
                label = label.to(device)

                pred_lp, pred_pl = model(bg, device)
                loss = (criterion(pred_lp, label) + criterion(pred_pl, label) + criterion(pred_lp, pred_pl)) / 3
                loss = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr, mae_val, corr_val, cindex_val, sd_val = val(model, val_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, test_rmse-%.4f, test_pr-%.4f, test_mae-%.4f, test_corr-%.4f, test_cindex-%.4f, test_sd-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr, mae_val, corr_val, cindex_val, sd_val)
            #logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, test_rmse-%.4f, test_pr-%.4f, test_mae-%.4f, test_corr-%.4f, test_cindex-%.4f, test_sd-%.4f" \
                          % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr, mae_val, corr_val, cindex_val, sd_val)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

            time.sleep(1)

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr, mae_val, corr_val, cindex_val, sd_val = val(model, test_loader, device)
        test2013_rmse, test2013_pr, mae_test2013, corr_test2013, cindex_test2013, sd_test2013 = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr, mae_test2016, corr_test2016, cindex_test2016, sd_test2016 = val(model, test2016_loader, device)
        testhiq_rmse, testhiq_pr, mae_testhiq, corr_testhiq, cindex_testhiq, sd_testhiq = val(model, testhiq_loader, device)

        # %%
