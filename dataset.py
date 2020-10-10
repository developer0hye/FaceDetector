import os
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn
import torchvision.transforms as transforms

# import augmentation
import tools


def read_annotation_file(path):
    with open(path, 'r') as label:
        objects_information = []
        for line in label:
            line = line.split()
            if len(line) == 5:  # 0: class, 1:x, 2:y, 3:w, 4:h
                object_information = []
                for data in line:
                    object_information.append(float(data))
                objects_information.append(object_information)
        objects_information = np.asarray(objects_information).astype(np.float32)
        return objects_information


class DatasetReader(Dataset):
    def __init__(self,
                 dataset_path,
                 model_input_size=(384, 384),
                 use_augmentation=True):

        self.dataset_path = dataset_path
        self.annotations = tools.read_annotation_files(os.path.join(self.dataset_path, "annotations"))
        self.normalize = transforms.Normalize(mean=[0.575, 0.533, 0.507], std=[0.235, 0.232, 0.233])

        self.model_input_size = model_input_size
        self.use_augmentation = use_augmentation

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_file_path = os.path.join(self.dataset_path, "img", annotation['annotation']['filename'])

        img = cv2.imread(img_file_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = tools.parse_bboxes(annotation)
        np.random.shuffle(label)

        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        # if self.use_augmentation:
        #     img, bboxes_xywh = augmentation.HorFlip(img, bboxes_xywh)
        #     bboxes_xyxy = augmentation.xywh2xyxy(bboxes_xywh)
        #     img, bboxes_xyxy, classes = augmentation.RandomCrop(img, bboxes_xyxy, classes)
        #     img, bboxes_xyxy, classes = augmentation.RandomTranslation(img, bboxes_xyxy, classes)
        #     img, bboxes_xyxy, classes = augmentation.RandomScale(img, bboxes_xyxy, classes)
        #     bboxes_xywh = augmentation.xyxy2xywh(bboxes_xyxy)

        classes = torch.from_numpy(classes)
        bboxes_xywh = torch.from_numpy(bboxes_xywh)
        bboxes = torch.cat([classes, bboxes_xywh], dim=-1)

        # to rgb
        img = cv2.resize(img, (self.model_input_size[0], self.model_input_size[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            img = img / 255.
            img = self.normalize(img)

        return img, bboxes, idx

    def __len__(self):
        return len(self.annotations)

def yolo_collate(batch_data):
    imgs = []
    batch_bboxes = []
    inds = []
    for img, bboxes, idx in batch_data:
        imgs.append(img)
        batch_bboxes.append(bboxes)
        inds.append(idx)
    return torch.stack(imgs, 0), batch_bboxes, inds

if __name__ == '__main__':
    dataset = DatasetReader(dataset_path="train",
                            model_input_size=(384, 384))

    for _ in dataset:
        pass