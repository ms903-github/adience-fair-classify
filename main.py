import argparse
import os
import pandas as pd
import sys
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
import math

from addict import Dict
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


from libs.functions import AverageMeter, ProgressMeter, accuracy
from libs.loader import load_pict, load_pict2
from libs.models import Classifier_resnet, Classifier, Discriminator
# from libs.transformer import MyTransformer


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for image classification with Flowers Recognition Dataset')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Add --resume option if you start training from checkpoint.'
    )

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, epoch, device):
    # 平均を計算してくれるクラス
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch)
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = sample[0]
        t = sample[1]

        x = x.to(device)
        t = t.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)
        loss = criterion(output, t)

        # measure accuracy and record loss
        acc1 = accuracy(output, t, topk=(1,))
        losses.update(math.sqrt(loss.item())*100, batch_size)
        top1.update(acc1[0].item(), batch_size)

        # keep predicted results and gts for calculate F1 Score
        _, pred = output.max(dim=1)
        gts += list(t.to("cpu").numpy())
        preds += list(pred.to("cpu").numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % 50 == 0:
            progress.display(i)

    # calculate F1 Score
    f1s = f1_score(gts, preds, average="macro")
    

    return losses.avg, top1.avg, f1s


def validate(val_loader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            x = sample[0]
            t = sample[1]
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)
            loss = criterion(output, t)

            # measure accuracy and record loss
            acc1 = accuracy(output, t, topk=(1,))
            losses.update(math.sqrt(loss.item())*100, batch_size)
            top1.update(acc1[0].item(), batch_size)

            # keep predicted results and gts for calculate F1 Score
            _, pred = output.max(dim=1)
            gts += list(t.to("cpu").numpy())
            preds += list(pred.to("cpu").numpy())

    f1s = f1_score(gts, preds, average="macro")
    
    return losses.avg, top1.avg, f1s

def train_adv(train_loader, model_g, model_h, model_d, criterion, optimizer_gh, optimizer_d, epoch, device, beta=1):
    model_g.to(device)
    model_h.to(device)
    model_d.to(device)
    # 平均を計算してくれるクラス
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_g = AverageMeter('Loss_g', ':.4e')
    top1_g = AverageMeter('Acc@1_g', ':6.2f')
    losses_d = AverageMeter('Loss_d', ':.4e')
    top1_d = AverageMeter('Acc@1_d', ':6.2f')

    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_g, top1_g, losses_d, top1_d],
        prefix="Epoch: [{}]".format(epoch)
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = sample[0]
        t = sample[1]
        g = sample[2]

        x = x.to(device)
        t = t.to(device)
        g = g.to(device)
        batch_size = x.shape[0]

        # train discriminator
        model_g.eval()
        model_d.train()
        feat = model_g(x)
        output = model_d(feat)
        loss_d = criterion(output, g)
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        acc1 = accuracy(output, g, topk=(1,))
        losses_d.update(loss_d.item(), batch_size)
        top1_d.update(acc1[0].item(), batch_size)

        # train generator
        model_g.train()
        model_h.train()
        model_d.eval()
        feat = model_g(x)
        output = model_h(feat)
        loss_g = criterion(output, t)
        # decieve discriminator
        output_d = model_d(model_g(x))
        adv_g = torch.LongTensor([0 if i == 1 else 1 for i in g]).to(device)
        loss_adv_g = criterion(output_d, adv_g)
        loss_g = loss_g + beta*loss_adv_g
        optimizer_gh.zero_grad()
        loss_g.backward()
        optimizer_gh.step()

        acc1 = accuracy(output, t, topk=(1,))
        losses_g.update(loss_g.item(), batch_size)
        top1_g.update(acc1[0].item(), batch_size)

        # keep predicted results and gts for calculate F1 Score
        _, pred = output.max(dim=1)
        gts += list(t.to("cpu").numpy())
        preds += list(pred.to("cpu").numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % 50 == 0:
            progress.display(i)

    # calculate F1 Score
    f1s = f1_score(gts, preds, average="macro")
    
    return losses_g.avg, losses_d.avg, top1_g.avg, top1_d.avg, f1s


def validate(val_loader, model, criterion, device, model_h=None, mode=None):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            x = sample[0]
            if mode == "d":
                t = sample[2]
            else:
                t = sample[1]
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            if model_h is not None:
                model_h.eval()
                output = model_h(model(x))
            else:
                output = model(x)
            loss = criterion(output, t)

            # measure accuracy and record loss
            acc1 = accuracy(output, t, topk=(1,))
            losses.update(math.sqrt(loss.item())*100, batch_size)
            top1.update(acc1[0].item(), batch_size)

            # keep predicted results and gts for calculate F1 Score
            _, pred = output.max(dim=1)
            gts += list(t.to("cpu").numpy())
            preds += list(pred.to("cpu").numpy())

    f1s = f1_score(gts, preds, average="macro")
    
    return losses.avg, top1.avg, f1s


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    if not os.path.exists(CONFIG.result_path):
        os.makedirs(CONFIG.result_path)
    shutil.copy(args.config, CONFIG.result_path)

    # cpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((CONFIG.width,CONFIG.height)),
             transforms.RandomResizedCrop(size=(CONFIG.width,CONFIG.height)),
             transforms.ToTensor()
             ])

    # Dataloader
    # if data are given in txt file format
    # train_data = load_pict(CONFIG.tr_path_data, transform=transform)
    # test_data = load_pict(CONFIG.te_path_data, transform=transform)
    # if data are given in directory format
    # train_data = load_pict(CONFIG.tr_data_path, transform=transform)
    train_data = load_pict2(CONFIG.tr_data_path, CONFIG.num_f_sample, CONFIG.num_m_sample, transform=transform)
    test_data = load_pict(CONFIG.te_data_path, transform=transform)
    test_f_data = load_pict(CONFIG.te_data_path, gen_mode="female", transform=transform)
    test_m_data = load_pict(CONFIG.te_data_path, gen_mode="male", transform=transform)

    train_loader = DataLoader(train_data, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True)
    val_f_loader = DataLoader(test_f_data, batch_size=1, shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True)
    val_m_loader = DataLoader(test_m_data, batch_size=1, shuffle=False, num_workers=CONFIG.num_workers, pin_memory=True)

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if CONFIG.adversarial:
        if CONFIG.model == 'resnet50':
            print('ResNet50 will be used as a model.')
            model_g = models.resnet50(pretrained=True).to(device)
        else:
            print('There is no model appropriate to your choice.')
            sys.exit(1)

        model_h = Classifier().to(device)
        model_d = Discriminator().to(device)
        optimizer_gh = optim.Adam(list(model_g.parameters()) + list(model_h.parameters()), lr=CONFIG.learning_rate)
        optimizer_d = optim.Adam(model_d.parameters(), lr=CONFIG.learning_rate)
        
        begin_epoch = 0

        # learning rate scheduler
        if CONFIG.scheduler == 'onplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_gh, 'min', patience=CONFIG.lr_patience
            )
        else:
            scheduler = None

        # create log
        best_acc1 = 0
        log = pd.DataFrame(
            columns=[
                'epoch', 'lr', 'train_loss', 'val_loss',
                'train_acc@1', 'val_acc@1', 'train_f1s', 'val_f1s', 
                'f_val_acc1', 'm_val_acc1', 
                'd_tr_loss', 'd_val_loss', 'd_tr_acc1', 'd_val_acc1', 'd_val_f1s'
            ]
        )

        # criterion for loss
        if CONFIG.class_weight:
            criterion = nn.CrossEntropyLoss(
                weight=get_class_weight(n_classes=n_classes).to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # train and validate model
        print('\n------------------------Start training------------------------\n')
        train_losses = []
        val_losses = []
        train_top1_accuracy = []
        val_top1_accuracy = []
        train_f1_score = []
        val_f1_score = []
        f_val_top1_acc = []
        f_val_f1_score = []
        m_val_top1_acc = []
        m_val_f1_score = []
        d_tr_losses = []
        d_val_losses = []
        d_tr_top1_acc = []
        d_val_top1_acc = []
        d_val_f1_score = []

    else:
        if CONFIG.model == 'resnet50':
            print('ResNet50 will be used as a model.')
            model = Classifier_resnet()
            for p in model.resnet.parameters():
                p.requires_grad = True
        else:
            print('There is no model appropriate to your choice.')
            sys.exit(1)

        # send the model to cuda/cpu
        model.to(device)

        if CONFIG.optimizer == 'Adam':
            print(CONFIG.optimizer + ' will be used as an optimizer.')
            optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
        elif CONFIG.optimizer == 'SGD':
            print(CONFIG.optimizer + ' will be used as an optimizer.')
            optimizer = optim.SGD(
                model.parameters(),
                lr=CONFIG.learning_rate,
                momentum=CONFIG.momentum,
                dampening=CONFIG.dampening,
                weight_decay=CONFIG.weight_decay,
                nesterov=CONFIG.nesterov
            )
        else:
            print(
                'There is no optimizer which suits to your option.'
                'You have to choose SGD or Adam as an optimizer in config.yaml')

        # learning rate scheduler
        if CONFIG.scheduler == 'onplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=CONFIG.lr_patience
            )
        else:
            scheduler = None

        begin_epoch = 0

        # create log
        best_acc1 = 0
        log = pd.DataFrame(
            columns=[
                'epoch', 'lr', 'train_loss', 'val_loss',
                'train_acc@1', 'val_acc@1', 'train_f1s', 'val_f1s', 
                'f_val_acc1', 'f_val_f1s', 'm_val_acc1', 'm_val_f1s'
            ]
        )

        # criterion for loss
        if CONFIG.class_weight:
            criterion = nn.CrossEntropyLoss(
                weight=get_class_weight(n_classes=n_classes).to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # train and validate model
        print('\n------------------------Start training------------------------\n')
        train_losses = []
        val_losses = []
        train_top1_accuracy = []
        val_top1_accuracy = []
        train_f1_score = []
        val_f1_score = []
        f_val_top1_acc = []
        f_val_f1_score = []
        m_val_top1_acc = []
        m_val_f1_score = []
        

    for epoch in range(begin_epoch, CONFIG.max_epoch):

        # training
        if CONFIG.adversarial:
            train_loss, train_loss_d, train_acc1, train_acc1_d, train_f1s = train_adv(
                train_loader, model_g, model_h, model_d, criterion, optimizer_gh, optimizer_d, epoch, device, beta=CONFIG.beta)
        else:
            train_loss, train_acc1, train_f1s = train(
                train_loader, model, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_top1_accuracy.append(train_acc1)
        train_f1_score.append(train_f1s)
        if CONFIG.adversarial:
            d_tr_losses.append(train_loss_d)
            d_tr_top1_acc.append(train_acc1_d)

        # validation
        if CONFIG.adversarial:
            val_loss, val_acc1, val_f1s = validate(
                val_loader, model_g, criterion, device, model_h=model_h)
            val_losses.append(val_loss)
            val_top1_accuracy.append(val_acc1)
            val_f1_score.append(val_f1s)
            d_val_loss, d_val_acc1, d_val_f1s = validate(
                val_loader, model_g, criterion, device, model_h=model_h, mode="d"
            )
            d_val_losses.append(d_val_loss)
            d_val_top1_acc.append(d_val_acc1)
            d_val_f1_score.append(d_val_f1s)    
            # save a model if top1 acc is higher than ever
            if best_acc1 < val_top1_accuracy[-1]:
                best_acc1 = val_top1_accuracy[-1]
                torch.save(
                    model_g.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_acc1_model_g.prm')
                )
                torch.save(
                    model_h.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_acc1_model_h.prm')
                )
            lr = optimizer_gh.param_groups[0]['lr']
        else:
            val_loss, val_acc1, val_f1s = validate(
                val_loader, model, criterion, device)
            val_losses.append(val_loss)
            val_top1_accuracy.append(val_acc1)
            val_f1_score.append(val_f1s)
            # save a model if top1 acc is higher than ever
            if best_acc1 < val_top1_accuracy[-1]:
                best_acc1 = val_top1_accuracy[-1]
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.result_path, 'best_acc1_model.prm')
                )
            lr = optimizer.param_groups[0]['lr']
        # scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            'epoch: {}(both)\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc1: {:.5f}\tval_f1s: {:.5f}'
            .format(epoch, lr, train_losses[-1],
                    val_losses[-1], val_top1_accuracy[-1], val_f1_score[-1])
        )
        # validation(female)
        if CONFIG.adversarial:
            f_val_loss, f_val_acc1, f_val_f1s = validate(
                val_f_loader, model_g, criterion, device, model_h=model_h)
        else:
            f_val_loss, f_val_acc1, f_val_f1s = validate(
                val_f_loader, model, criterion, device)

        f_val_top1_acc.append(f_val_acc1)
        f_val_f1_score.append(f_val_f1s)

        print(
            'epoch: {}(female)\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc1: {:.5f}\tval_f1s: {:.5f}'
            .format(epoch, lr, train_losses[-1],
                    f_val_loss, f_val_acc1, f_val_f1s)
        )
        # validation(male)
        if CONFIG.adversarial:
            m_val_loss, m_val_acc1, m_val_f1s = validate(
                val_m_loader, model_g, criterion, device, model_h=model_h)
        else:
            m_val_loss, m_val_acc1, m_val_f1s = validate(
                val_m_loader, model, criterion, device)

        m_val_top1_acc.append(m_val_acc1)
        m_val_f1_score.append(m_val_f1s)

        print(
            'epoch: {}(male)\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc1: {:.5f}\tval_f1s: {:.5f}'
            .format(epoch, lr, train_losses[-1],
                    m_val_loss, m_val_acc1, m_val_f1s)
        )

        if CONFIG.adversarial:
            print(
                'epoch: {}(discriminator)\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc1: {:.5f}\tval_f1s: {:.5f}'
                .format(epoch, lr, d_tr_losses[-1], d_val_losses[-1],
                        d_val_top1_acc[-1], d_val_f1_score[-1])
            )

    # write logs to dataframe and csv file
        if CONFIG.adversarial:
            tmp = pd.Series([
                epoch,
                lr,
                train_losses[-1],
                val_losses[-1],
                train_top1_accuracy[-1],
                val_top1_accuracy[-1],
                train_f1_score[-1],
                val_f1_score[-1],
                f_val_top1_acc[-1],
                m_val_top1_acc[-1],
                d_tr_losses[-1],
                d_val_losses[-1],
                d_tr_top1_acc[-1],
                d_val_top1_acc[-1],
                d_val_f1_score[-1]
                ], index=log.columns
            )

        else:
            tmp = pd.Series([
                epoch,
                lr,
                train_losses[-1],
                val_losses[-1],
                train_top1_accuracy[-1],
                val_top1_accuracy[-1],
                train_f1_score[-1],
                val_f1_score[-1],
                f_val_top1_acc[-1],
                f_val_f1_score[-1],
                m_val_top1_acc[-1],
                m_val_f1_score[-1]
            ], index=log.columns
            )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)



if __name__ == '__main__':
    main()
