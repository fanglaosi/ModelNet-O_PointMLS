"""
python test.py --model pointMLP --msg 20220209053148-404
"""
import argparse
import csv
import os
import datetime
import pickle

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import progress_bar, IOStream
from ModelNet import ModelNet40, ModelNet_O
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


model_names = sorted(name for name in models.__dict__ if callable(models.__dict__[name]))

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', default='checkpoint/ModelNet_O',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='PointMLS_basic', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number')
    parser.add_argument('--test_path', default='test_score', help='path to save testing score')
    parser.add_argument('--val_path', default="data/ModelNet_O/*/test", help='path to testing data')
    parser.add_argument('--dataset', default='MNO', help='dataset')

    return parser.parse_args()

def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")

    args.checkpoint = os.path.join(args.checkpoint, args.model)

    if args.dataset == "MN40":
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=True)
    elif args.dataset == "MNO":
        test_loader = DataLoader(ModelNet_O(args.val_path, args.num_points), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=True)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])
    test_out = validate(net, test_loader, criterion, device, args)
    print(f"Vanilla out: {test_out}")


def validate(net, testloader, criterion, device, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []

    score = []

    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            logits, _ = net(data, 0.01)
            loss = criterion(logits, label)
            test_loss += loss.item()

            score.append(logits.detach().cpu().numpy())

            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    score = np.concatenate(score)

    if not os.path.exists(args.test_path):
        os.mkdir(args.test_path)
    with open('{}/{}_score.pkl'.format(args.test_path, args.model), 'wb') as f:
        pickle.dump(score, f)

    if args.model == "PointMLS_basic":
        with open('{}/{}_label.pkl'.format(args.test_path, args.model), 'wb') as f:
            pickle.dump(test_true, f)

    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }

def set_seed():
    torch.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)
    torch.cuda.manual_seed_all(2022)
    torch.cuda.manual_seed(2022)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(2022)

if __name__ == '__main__':
    set_seed()
    main()