#!/usr/bin/env python
import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    parser = argparse.ArgumentParser(description='Fused Spatial-Temporal Model Training')
    parser.add_argument('--work-dir', default='./work_dir/fused', help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='./runs/fused_model', help='prefix for saved model weights')
    parser.add_argument('--config', default='./config/train_fused.yml', help='path to the configuration file')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=True, help='if true, the classification score will be stored')
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=2, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--model', default='model.fused_model.FusedModel', help='the model will be used')
    parser.add_argument('--model-args', type=dict, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights to ignore')
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 50], nargs='+', help='the epoch where optimizer reduces the learning rate')
    parser.add_argument('--device', type=int, default=1, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=60, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--aux-loss-weight', type=float, default=0.5, help='weight for auxiliary losses')
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            weights = torch.load(self.arg.weights)
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        if type(self.arg.device) is list and len(self.arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch, after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(vars(self.arg), f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer in ['SGD', 'Adam']:
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=True):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # Forward pass
            output, spatial_aux, temporal_aux = self.model(data)
            loss_fused = self.loss(output, label)
            loss_spatial = self.loss(spatial_aux, label)
            loss_temporal = self.loss(temporal_aux, label)
            loss = loss_fused + self.arg.aux_loss_weight * (loss_spatial + loss_temporal)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output, 1)
            acc = torch.mean((predict_label == label).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)
            self.train_writer.add_scalar('loss_fused', loss_fused.item(), self.global_step)
            self.train_writer.add_scalar('loss_spatial', loss_spatial.item(), self.global_step)
            self.train_writer.add_scalar('loss_temporal', loss_temporal.item(), self.global_step)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log(f'\tMean training loss: {np.mean(loss_value):.4f}.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

        if save_model and ((epoch + 1) % self.arg.save_interval == 0 or epoch + 1 == self.arg.num_epoch):
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, f'{self.arg.model_saved_name}-{epoch+1}-{int(self.global_step)}.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file:
            f_w = open(wrong_file, 'w')
        if result_file:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log(f'Eval epoch: {epoch + 1}')
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False)
                    label = Variable(label.long().cuda(self.output_device), requires_grad=False)
                    output, _, _ = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.item())
                    _, predict_label = torch.max(output, 1)
                    right_num_total += torch.sum(predict_label == label).item()
                    total_num += label.size(0)
                    if wrong_file or result_file:
                        predict = predict_label.cpu().numpy().tolist()
                        true = label.cpu().numpy().tolist()
                        for i, (p, t) in enumerate(zip(predict, true)):
                            if result_file:
                                f_r.write(f'{p},{t}\n')
                            if p != t and wrong_file:
                                f_w.write(f'{index[i]},{p},{t}\n')
            score = np.concatenate(score_frag)
            accuracy = right_num_total / total_num
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                if self.arg.phase == 'train':
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(weights, f'{self.arg.model_saved_name}-best.pt')
            self.print_log(f'Accuracy: {accuracy:.4f} model: {self.arg.model_saved_name}')
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', np.mean(loss_value), self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss: {np.mean(loss_value):.4f}')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop{k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')
            if save_score:
                with open(f'{self.arg.work_dir}/epoch{epoch+1}_{ln}_score.pkl', 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-4:
                    break
                self.train(epoch, save_model=True)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log(f'Best accuracy: {self.best_acc:.4f} model_name: {self.arg.model_saved_name}')
        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
