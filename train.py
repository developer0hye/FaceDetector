import os
import time
import random
import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import dataset

import argparse
from model import *
import tools

parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--total_epoch', type=int, default=200,
                    help='total_epoch')
parser.add_argument('--dataset_root', default="train",
                    help='Location of dataset directory')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--gpu_ind', default=0, type=int,
                    help='To choose your gpu.')
parser.add_argument('--save_folder', default='./weights', type=str,
                    help='where you save weights')
parser.add_argument('--seed', default=21, type=int)

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, device):
    # set GPU
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    train_dataset = dataset.DatasetReader(dataset_path=args.dataset_root,
                                          model_input_size=(384, 384),
                                          use_augmentation=True)

    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=dataset.yolo_collate,
                                  pin_memory=True,
                                  drop_last=True)

    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's train OD network !")

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    iter_per_epoch = int(np.ceil(len(train_dataset)/args.batch_size))
    total_iter = iter_per_epoch * args.total_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_epoch, eta_min=1e-4)

    # loss counters
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    print('Training on:', args.dataset_root)
    print('The dataset size:', len(train_dataset))
    print("----------------------------------------------------------")

    # create batch iterator
    iteration = 0
    start_epoch = 0

# start training
    for epoch in range(start_epoch, args.total_epoch):
        model.train()

        if epoch == 10:
            for p in model.backbone.parameters():
                p.requires_grad = True

        for batch_imgs, batch_target_bboxes, inds in data_loader:
            iteration += 1

            batch_imgs = batch_imgs.to(device)

            input_size = batch_imgs.shape[2:]

            # forward
            t0 = time.time()

            batch_multi_scale_raw_bboxes, batch_multi_scale_bboxes = model(batch_imgs)

            with torch.no_grad():
                targets = tools.build_target_tensor(model=model,
                                                    batch_pred_bboxes=batch_multi_scale_bboxes,
                                                    batch_target_bboxes=batch_target_bboxes,
                                                    input_size=(384, 384))
                targets = targets.to(device)
            loss = yololoss(batch_multi_scale_raw_bboxes, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()

            if iteration % 100 == 0:
                positive_samples = batch_multi_scale_bboxes[torch.sum(targets[..., 5:], dim=-1) == 1]
                negative_samples = batch_multi_scale_bboxes[torch.sum(targets[..., 5:], dim=-1) == 0]

                print(' ')
                print('positive mean obj ', (positive_samples[:, 4]).mean())
                print('negative mean obj ', (negative_samples[:, 4]).mean())

                print('postive objs(>= 0.5): ', len(torch.nonzero(positive_samples[:, 4] > 0.5)))
                print('timer: %.4f sec.' % (t1 - t0))
                print('loss: ', loss)
                print('Epoch[%d / %d]' % (epoch + 1, args.total_epoch) + ' || iter[%d / %d] ' % (
                iteration, total_iter) + \
                      ' || Loss: %.4f ||' % (loss.item()) + ' || lr: %.8f ||' % (
                      optimizer.param_groups[0]['lr']) + ' || input size: %d ||' %
                      input_size[0], end=' ')

        lr_scheduler.step()

        if epoch >= 50:
          chkpt = {'epoch': epoch + 1,
                  'iteration': iteration,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

          print('Saving state, epoch:', epoch + 1)

          
          torch.save(chkpt,
                      args.save_folder + '/' + 'nota_face_detector' + '_' +
                      repr(epoch + 1) + '.pth')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)

    model = LightWeightFaceDetector()

    model = model.to(device)
    train(model, device)
