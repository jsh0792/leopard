import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utils.tools import EarlyStopping, get_logger
import os
from tqdm import tqdm
from utils.loss import absolute_loss_without_c, nll_loss
from utils.metric import c_index
import torch.nn as nn
import pandas as pd
from torchmetrics.classification import Accuracy, AUROC, F1Score
import gc

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, optimizer, args, cfg, device, fold,
                 train_data_loader, valid_data_loader=None, num_class=2, lr_scheduler = None):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.fold = fold
        self.epochs = args.epochs
        self.writer = SummaryWriter(args.save_dir + '/{}'.format(fold), flush_secs=15)  # tensorboard的保存路径
        self.model_path = os.path.join(args.save_dir, str(fold), 'model.pt')
        self.gc = 32
        self.lr_scheduler = lr_scheduler
        self.best_valid_cindex = 0.
        self.logger = get_logger('')

        self.debug_save = True  # debug

    def _train_epoch(self, epoch, early_stopping):
        self.model.train()
    
        all_risk_scores = np.zeros((len(self.data_loader)))
        all_censorships = np.zeros((len(self.data_loader)))
        all_event_times = np.zeros((len(self.data_loader)))

        train_loss, surv_loss_log = 0., 0.

        # for batch_idx, (path_features, Y_surv, event_time, if_event, case_id) in enumerate(self.data_loader, 1):
        for batch_idx, (path_features, Y_surv, event_time, if_event, case_id) in enumerate(tqdm(self.data_loader)):

            path_features = path_features[0].to(self.device)

            Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
            c = 1.0 - if_event.type(torch.FloatTensor).to(self.device)  # censorship+if_event=1
            event_time = event_time.to(self.device) 

            if self.cfg.Model.name == 'DSMIL':
                torch.cuda.empty_cache()
                max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = self.model(wsi=path_features, label=Y_surv)
                surv_loss = 0.5*nll_loss(hazards=hazards_i, S=S_i, Y=Y_surv, c=c) + 0.5*nll_loss(hazards=hazards_b, S=S_b, Y=Y_surv, c=c)
                loss = surv_loss / self.gc
                surv_loss_log += surv_loss.item()
                train_loss += loss.item()

                logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
                S = torch.cumprod(1 - logits, dim=1)

            # elif self.cfg.Model.name == 'GAT_custom' or self.cfg.Model.name == 'GAT':
            #     torch.cuda.empty_cache()
            #     risk = self.model(wsi=path_features, label=Y_surv)
            else:
                if self.cfg.Loss.kind == 'l1':
                    torch.cuda.empty_cache()
                    pred_time = self.model(wsi=path_features, label=Y_surv)
                    surv_loss = absolute_loss_without_c(event_time, pred_time, c)
                else:
                    torch.cuda.empty_cache()
                    surv_logits, hazards, S = self.model(wsi=path_features, label=Y_surv)
                    surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)

                loss = surv_loss / self.gc
                surv_loss_log += surv_loss.item()
                train_loss += loss.item()
    
            loss.backward()

            if (batch_idx + 1) % self.gc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.cfg.Loss.kind == 'l1':
                risk = -pred_time.detach().cpu().numpy()
            else:
                risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        train_loss /= len(self.data_loader)

        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train/surv_loss', surv_loss_log, epoch)
        self.writer.add_scalar('train/c_index', cindex, epoch)

        print("\n\n\n====\033[1;32mTraining\033[0m Statistics====")
        print('\033[1;34mTrain Loss\033[0m: \033[1;31m{:.4f}\033[0m'.format(train_loss))
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))

        self._valid_epoch(epoch, early_stopping)


    def _valid_epoch(self, epoch, early_stopping):

        self.model.eval()
        with torch.no_grad():   
    
            all_risk_scores = np.zeros((len(self.valid_data_loader)))
            all_censorships = np.zeros((len(self.valid_data_loader)))
            all_event_times = np.zeros((len(self.valid_data_loader)))
    
            val_loss, surv_loss_log = 0., 0.
    
            for batch_idx, (path_features, Y_surv, event_time, c, case_id) in enumerate(tqdm(self.valid_data_loader)): 

                path_features = path_features[0].to(self.device)

                Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
                c = 1.0 - c.type(torch.FloatTensor).to(self.device) # censor+event=1
                event_time = event_time.to(self.device) 
    
                if self.cfg.Model.name == 'DSMIL':
                    max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = self.model(wsi=path_features, label=Y_surv)
                    surv_loss = 0.5*nll_loss(hazards=hazards_i, S=S_i, Y=Y_surv, c=c) + 0.5*nll_loss(hazards=hazards_b, S=S_b, Y=Y_surv, c=c)
                    loss = surv_loss / self.gc
                    surv_loss_log += surv_loss.item()
                    val_loss += loss.item()
                    logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
                    S = torch.cumprod(1 - logits, dim=1)
                else:
                    if self.cfg.Loss.kind == 'l1':
                        pred_time = self.model(wsi=path_features, label=Y_surv)
                        surv_loss = absolute_loss_without_c(event_time, pred_time, c)
                    else:
                        surv_logits, hazards, S = self.model(wsi=path_features, label=Y_surv)
                        surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)
                    loss = surv_loss / self.gc
                    surv_loss_log += surv_loss.item()
                    val_loss += loss.item()
    
                if self.cfg.Loss.kind == 'l1':
                    risk = -pred_time.detach().cpu().numpy()
                else:
                    risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = c.item()
                all_event_times[batch_idx] = event_time

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        val_loss /= len(self.data_loader)

        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/surv_loss', surv_loss_log, epoch)
        self.writer.add_scalar('val/c_index', cindex, epoch)

        print("\n\n\n====\033[1;32mValid\033[0m Statistics====")
        print('\033[1;34mValid Loss\033[0m: \033[1;31m{:.4f}\033[0m'.format(val_loss))
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))

        if self.debug_save==False:
            return

        # ckpt_name是保存模型的路径的名称
        if cindex > self.best_valid_cindex :     ## debug
            self.best_valid_cindex = cindex
            model_path = os.path.join(self.args.save_dir, str(self.fold), 'epoch_'+str(epoch)+'_index_'+str(cindex)+'.pth'  )
            if self.args.multi_gpu_mode == 'DataParallel':
                torch.save(self.model.module.state_dict(), model_path)
            else:
                torch.save(self.model.state_dict(), model_path)
        metric = cindex
        early_stopping(epoch=epoch, metric=metric, models=self.model, ckpt_name=os.path.join(self.args.save_dir, str(self.fold)))   # 在验证阶段会累积score不改变的次数

    def _test_epoch(self, epoch, early_stopping):

        l_0403 = []
        l_0046 = []
        l_0400 = []

        self.model.eval()
        with torch.no_grad():   
    
            all_risk_scores = np.zeros((len(self.valid_data_loader)))
            all_censorships = np.zeros((len(self.valid_data_loader)))
            all_event_times = np.zeros((len(self.valid_data_loader)))
    
            val_loss, surv_loss_log = 0., 0.
    
            for batch_idx, (path_features, Y_surv, event_time, c, case_id) in enumerate(tqdm(self.valid_data_loader)): 
                path_features = path_features[0].to(self.device)
                Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
                c = 1.0 - c.type(torch.FloatTensor).to(self.device) # censor+event=1
                event_time = event_time.to(self.device) 
    
                if self.cfg.Model.name == 'DSMIL':
                    max_prediction, bag_prediction, hazards_i, S_i, hazards_b, S_b = self.model(wsi=path_features, label=Y_surv)
                    surv_loss = 0.5*nll_loss(hazards=hazards_i, S=S_i, Y=Y_surv, c=c) + 0.5*nll_loss(hazards=hazards_b, S=S_b, Y=Y_surv, c=c)
                    loss = surv_loss / self.gc
                    surv_loss_log += surv_loss.item()
                    val_loss += loss.item()
                    logits = 0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)
                    S = torch.cumprod(1 - logits, dim=1)
                else:
                    if self.cfg.Loss.kind == 'l1':
                        pred_time = self.model(wsi=path_features, label=Y_surv)
                        surv_loss = absolute_loss_without_c(event_time, pred_time, c)
                    else:
                        surv_logits, hazards, S = self.model(wsi=path_features, label=Y_surv)
                        surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)
                    loss = surv_loss / self.gc
                    surv_loss_log += surv_loss.item()
                    val_loss += loss.item()
    
                if self.cfg.Loss.kind == 'l1':
                    risk = -pred_time.detach().cpu().numpy()
                else:
                    risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                all_risk_scores[batch_idx] = risk
                all_censorships[batch_idx] = c.item()
                all_event_times[batch_idx] = event_time

                if '0403' in case_id[0]:
                    l_0403.append(risk[0])
                if '0046' in case_id[0]:
                    l_0046.append(risk[0])
                if '0400' in case_id[0]:
                    l_0400.append(risk[0])
            print(l_0403)
            print(l_0046)
            print(l_0400)
            # import statistics
            # print(statistics.mean(l_0403))
            # print(statistics.mean(l_0046))
            # print(statistics.mean(l_0400))

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        val_loss /= len(self.data_loader)

        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/surv_loss', surv_loss_log, epoch)
        self.writer.add_scalar('val/c_index', cindex, epoch)

        print("\n\n\n====\033[1;32mValid\033[0m Statistics====")
        print('\033[1;34mValid Loss\033[0m: \033[1;31m{:.4f}\033[0m'.format(val_loss))
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))

        if self.debug_save==False:
            return


    def train(self, fold=0):
        early_stopping = EarlyStopping(warmup=2, patience=5, stop_epoch=18, verbose = True, logger=self.logger, multi_gpus=self.args.multi_gpu_mode)

        for epoch in range(0, self.epochs):
            print(f'Epoch : {epoch}:')
            self._train_epoch(epoch, early_stopping)
            if self.cfg.Lr_scheduler._if:
                self.lr_scheduler.step()

            gc.collect()
            torch.cuda.empty_cache()

        cindex = {'ci': self.best_valid_cindex}
        result_file = os.path.join(self.args.save_dir, 'result.csv')
        summary_file = os.path.join(self.args.save_dir, 'summary.csv')
        result = {**cindex}
        df = pd.DataFrame.from_dict(result, orient='index').T
        if fold == 0:
            df.to_csv(result_file, mode='a', header=True, index=False)  # 临时修改
        else:
            df.to_csv(result_file, mode='a', header=False, index=False)
        if fold == 4:   # 临时修改
            df = pd.read_csv(result_file)
            result = {'ci_avg': df['ci'].mean(), 'ci_std': df['ci'].std()}
            df = pd.DataFrame.from_dict(result, orient='index').T
            df.to_csv(summary_file, index=False)


    ## debug
    def load_model_state(self, model, state_dict_path):
        state_dict = torch.load(state_dict_path)

        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            state_dict = new_state_dict
        model.load_state_dict(state_dict)

    ## debug
    def test(self, fold=0):
        # ckpt_name=os.path.join(self.args.save_dir, str(self.fold), 'mdoel.pt')
        # print(ckpt_name)

        ckpt_name = self.cfg.Data.model_path
        print(ckpt_name)
        if self.args.multi_gpu_mode == 'DataParallel':
            self.load_model_state(self.model.module, ckpt_name)
        else:
            self.load_model_state(self.model, ckpt_name)

        self.debug_save = False
        self._test_epoch(epoch=-1, early_stopping=None)
