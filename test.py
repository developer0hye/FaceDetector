import os
import time
import random
import torch
import torch.optim as optim
import numpy as np
import cv2
from pathlib import Path
import argparse
from model import *
import tools
import dataset
from pathlib import Path

parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for testing')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--wp_epoch', type=int, default=6,
                    help='The upper bound of warm-up')
parser.add_argument('--weights', type=str, default=None,
                    help='load weights to resume training')
parser.add_argument('--dataset_root', default="test",
                    help='Location of dataset directory')
parser.add_argument('--save_folder', default="predicted_results",
                    help='Location of output directory')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--seed', default=21, type=int)

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(model, device):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    test_dataset = dataset.TestDatasetReader(dataset_path=args.dataset_root)
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=False,
                                              pin_memory=True)

    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's test OD network !")

    # loss counters
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    print('Testing on:', args.dataset_root)
    print('The dataset size:', len(test_dataset))
    print("----------------------------------------------------------")

    # create batch iterator
    if args.weights is not None:
        chkpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(chkpt)

    class_mapping_table = {0: "neutral",
                           1: "anger",
                           2: "surprise",
                           3: "smile",
                           4: "sad"}
    # start test
    model.eval()
    with torch.no_grad():
        for batch_imgs, inds in data_loader:
            batch_imgs = batch_imgs.to(device)
            # forward
            batch_multi_scale_raw_bboxes, batch_multi_scale_bboxes = model(batch_imgs)

            filtered_batch_multi_scale_bboxes = bboxes_filtering(batch_multi_scale_bboxes)

            for index, single_bboxes in zip(inds, filtered_batch_multi_scale_bboxes):
                # results_directory = os.path.join(args.save_folder, Path(test_dataset.img_files_path[index]).stem) + ".txt"
                # with open(results_directory, "w") as f:
                #     for idx_class, confidence, position in zip(single_bboxes['class'],
                #                                                single_bboxes['confidence'],
                #                                                single_bboxes['position']):
                #         f.write(str(int(round(idx_class))))
                #         f.write(" ")
                #         f.write(str(confidence))
                #         f.write(" ")
                #         f.write(str(position[0]))
                #         f.write(" ")
                #         f.write(str(position[1]))
                #         f.write(" ")
                #         f.write(str(position[2]))
                #         f.write(" ")
                #         f.write(str(position[3]))
                #         f.write("\n")

                img = cv2.imread(test_dataset.img_files_path[index])
                img_h, img_w = img.shape[:2]

                pad = 80 #출력되는 글자가 잘리는 문제를 완화하기위하여 패딩
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

                for idx in range(len(single_bboxes['position'])):
                    class_idx = single_bboxes['class'][idx]

                    bbox = single_bboxes['position'][idx]
                    bbox[[0, 2]] *= img_w
                    bbox[[1, 3]] *= img_h

                    bbox = bbox.astype(np.int32)

                    l = int(bbox[0] - bbox[2] / 2)+pad
                    r = int(bbox[0] + bbox[2] / 2)+pad

                    t = int(bbox[1] - bbox[3] / 2)+pad
                    b = int(bbox[1] + bbox[3] / 2)+pad

                    cv2.rectangle(img=img, pt1=(l, t), pt2=(r, b), color=(0, 255, 0))
                    cv2.putText(img, class_mapping_table[class_idx], (l, t), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imwrite(os.path.join(args.save_folder, Path(test_dataset.img_files_path[index]).name), img)
                cv2.imshow('img', img)
                if cv2.waitKey(30) == 27:
                    return


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)

    model = LightWeightFaceDetector()

    model = model.to(device)
    test(model, device)
