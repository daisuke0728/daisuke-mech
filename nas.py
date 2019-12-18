import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model_search import Network
from architect import Architect

from argparse import Namespace

import torch.nn as nn
from torch.nn import Module
from torchvision.transforms import Compose

from task import Task

def train(args,train_queue, valid_queue, model, architect, criterion, optimizer, lr):

    print("train queue!!!!!!!!!!")
    print(train_queue)

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step,(input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.module.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(args,valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def nas(args:Namespace,task: Task, preprocess_func: Compose) -> Module:
    ''' Network Architecture Search method                                                                           
                                                                                                                     
    Given task and preprocess function, this method returns a model output by NAS.                                   
                                                                                                                     
    The implementation of DARTS is available at https://github.com/alphadl/darts.pytorch1.1                          
    '''

    # TODO: Replace model with the output by NAS                                                                     

    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    CLASSES = task.n_classes

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)                                                                                
    #gpus = [int(args.gpu)]                                                                                          
    gpus = [int(i) for i in args.gpu.split(',')]
    if len(gpus) == 1:
        torch.cuda.set_device(int(args.gpu))

    # cudnn.benchmark = True                                                                                         
    torch.manual_seed(args.seed)
    # cudnn.enabled=True                                                                                             
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = model.cuda()
    if len(gpus)>1:
        print("True")
        model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model = model.module

    arch_params = list(map(id, model.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params,
                           model.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        # model.parameters(),                                                                                        
        weight_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer = nn.DataParallel(optimizer, device_ids=gpus)

    if task.name == 'cifar100':
        #train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=preprocess_func)            
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)


    elif task.name=='cifar10':
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=preprocess_func)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.module, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training                                                                                                   
        train_acc, train_obj = train(args,train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)

        # validation                                                                                                 
        with torch.no_grad():
            valid_acc, valid_obj = infer(args,valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    # return a neural network model (torch.nn.Module)                                                                
    return model
