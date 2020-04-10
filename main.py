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

from addict import Dict
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from libs.functions import AverageMeter, ProgressMeter, accuracy
from libs.loader import load_pict, load_pict2
from libs.models import Classifier_resnet
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
        losses.update(loss.item(), batch_size)
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
            x = sample['img']
            t = sample['class_id']
            x = x.to(device)
            t = t.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)
            loss = criterion(output, t)

            # measure accuracy and record loss
            acc1 = accuracy(output, t, topk=(1,))
            losses.update(loss.item(), batch_size)
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
    train_data = load_pict2(CONFIG.data_path, transform=transform)
    test_data = load_pict2(CONFIG.data_path, transform=transform, test=True)

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    # the number of classes
    n_classes = CONFIG.n_classes

    if CONFIG.model == 'resnet50':
        print('ResNet50 will be used as a model.')
        model = Classifier_resnet(n_classes)
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
            'train_acc@1', 'val_acc@1', 'train_f1s', 'val_f1s'
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

    for epoch in range(begin_epoch, CONFIG.max_epoch):

        # training
        train_loss, train_acc1, train_f1s = train(
            train_loader, model, criterion, optimizer, epoch, device)

        train_losses.append(train_loss)
        train_top1_accuracy.append(train_acc1)
        train_f1_score.append(train_f1s)

        # validation
        val_loss, val_acc1, val_f1s = validate(
            val_loader, model, criterion, device)

        val_losses.append(val_loss)
        val_top1_accuracy.append(val_acc1)
        val_f1_score.append(val_f1s)

        # scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # save a model if top1 acc is higher than ever
        if best_acc1 < val_top1_accuracy[-1]:
            best_acc1 = val_top1_accuracy[-1]
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG.result_path, 'best_acc1_model.prm')
            )

        # save checkpoint every epoch
        save_checkpoint(
            CONFIG.result_path, epoch, model, optimizer, best_acc1, scheduler)

        # save a model every 10 epoch
        if epoch % 10 == 0 and epoch != 0:
            save_checkpoint(
                CONFIG.result_path, epoch, model, optimizer,
                best_acc1, scheduler, add_epoch2name=True
            )

        
        # write logs to dataframe and csv file
        tmp = pd.Series([
            epoch,
            optimizer.param_groups[0]['lr'],
            train_losses[-1],
            val_losses[-1],
            train_top1_accuracy[-1],
            val_top1_accuracy[-1],
            train_f1_score[-1],
            val_f1_score[-1]
        ], index=log.columns
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        print(
            'epoch: {}\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc1: {:.5f}\tval_f1s: {:.5f}'
            .format(epoch, optimizer.param_groups[0]['lr'], train_losses[-1],
                    val_losses[-1], val_top1_accuracy[-1], val_f1_score[-1])
        )

    # save models
    torch.save(
        model.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))


if __name__ == '__main__':
    main()
