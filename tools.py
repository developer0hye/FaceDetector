import json
import os

import cv2
import numpy as np
import xmltodict

import torch
import copy

def parse_bboxes(annotation):
    bboxes = []

    class_mapping_table = {"neutral": 0,
                           "anger": 1,
                           "surprise": 2,
                           "smile": 3,
                           "sad": 4}

    img_width = int(annotation['annotation']['size']['width'])
    img_height = int(annotation['annotation']['size']['height'])

    for object in annotation['annotation']['object']:

        c = class_mapping_table[object['name']]
        bbox = object['bndbox']

        xmax = int(bbox['xmax'])/img_width
        xmin = int(bbox['xmin'])/img_width
        ymax = int(bbox['ymax'])/img_height
        ymin = int(bbox['ymin'])/img_height

        cx = (xmax + xmin) / 2.
        cy = (ymax + ymin) / 2.

        w = xmax-xmin
        h = ymax-ymin

        bboxes.append([c, cx, cy, w, h])

    bboxes = np.stack(bboxes).astype(np.float32)
    return bboxes

def read_annotation_file(path):
    with open(path, "r") as f:
        f = f.read()
        f = xmltodict.parse(f)
        f = json.loads(json.dumps(f))

        if isinstance(f['annotation']['object'], dict):
            f['annotation']['object'] = [f['annotation']['object']]

        return f

def read_annotation_files(dataset_dir):
    annotation_files_path = os.listdir(dataset_dir)
    annotation_files_path = [os.path.join(dataset_dir, annotation_file_path)
                             for annotation_file_path in annotation_files_path
                             if annotation_file_path.endswith(".xml")]

    annotations = [read_annotation_file(annotation_file_path)
                   for annotation_file_path in annotation_files_path]

    return annotations

def calc_statistics(train_dataset_path, annotations):
    #return RGB mean and standard deviation

    batch_mean = []
    batch_std = []

    for annotation in annotations:
        annotation = annotation['annotation']
        img_file_path = os.path.join(train_dataset_path, "img", annotation['filename'])
        img = cv2.imread(img_file_path)
        img = img.astype(np.float32)
        img = img / 255.
        img = img[:, :, (2, 1, 0)] #BGR2RGB
        img = img.transpose(2, 0, 1)
        img = img.reshape((3, -1))
        batch_mean.append(np.mean(img, axis=1).reshape(1, 3))
        batch_std.append(np.std(img, axis=1).reshape(1, 3))

    batch_mean = np.concatenate(batch_mean, axis=0).T  # 3, n
    batch_std = np.concatenate(batch_std, axis=0).T  # 3, n

    batch_mean = np.mean(batch_mean, axis=1)
    batch_std = np.mean(batch_std, axis=1)

    return batch_mean, batch_std


def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.clone()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.clone()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh


def iou_xyxy(boxA_xyxy, boxB_xyxy):
    # determine the (x, y)-coordinates of the intersection rectangle
    x11, y11, x12, y12 = torch.split(boxA_xyxy, 1, dim=1)
    x21, y21, x22, y22 = torch.split(boxB_xyxy, 1, dim=1)

    xA = torch.max(x11, x21.T)
    yA = torch.max(y11, y21.T)
    xB = torch.min(x12, x22.T)
    yB = torch.min(y12, y22.T)

    interArea = (xB - xA).clamp(0) * (yB - yA).clamp(0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    unionArea = (boxAArea + boxBArea.T - interArea)
    iou = interArea / (unionArea + 1e-6)

    # return the intersection over union value
    return iou


def iou_xywh(boxA_xywh, boxB_xywh):
    boxA_xyxy = xywh2xyxy(boxA_xywh)
    boxB_xyxy = xywh2xyxy(boxB_xywh)

    # return the intersection over union value
    return iou_xyxy(boxA_xyxy, boxB_xyxy)

def whiou(box1_wh, box2_wh):
    # determine the (x, y)-coordinates of the intersection rectangle
    eps = 1e-6

    w1, h1 = torch.split(box1_wh, 1, dim=1)
    w2, h2 = torch.split(box2_wh, 1, dim=1)

    innerW = torch.min(w1, w2.T).clamp(0)
    innerH = torch.min(h1, h2.T).clamp(0)

    interArea = innerW * innerH
    box1Area = w1 * h1
    box2Area = w2 * h2
    iou = interArea / (box1Area + box2Area.T - interArea + eps)

    # return the intersection over union value
    return iou

def build_target_tensor(model,
                        batch_pred_bboxes,
                        batch_target_bboxes,
                        input_size):
    batch_pred_bboxes = batch_pred_bboxes.cpu()
    batch_target_bboxes = copy.deepcopy(batch_target_bboxes)
    h, w = input_size
    o = (4 + 1 + model.num_classes)

    batch_size = len(batch_target_bboxes)
    batch_target_tensor = []

    for _ in range(batch_size):
        single_target_tensor = []
        for idx, stride in enumerate(model.strides):
            for _ in range(len(model.anchors_mask[idx])):
                single_target_tensor.append(torch.zeros((h // stride, w // stride, o), dtype=torch.float32))
        batch_target_tensor.append(single_target_tensor)

    for idx_batch in range(batch_size):
        single_target_bboxes = []
        for single_target_bbox in batch_target_bboxes[idx_batch]:
            c = int(torch.round(single_target_bbox[0]))

            bbox_xy = single_target_bbox[1:3].clone().view(1, 2)
            bbox_wh = single_target_bbox[3:].clone().view(1, 2)

            bbox_wh[0, 0] *= h
            bbox_wh[0, 1] *= w

            iou = whiou(bbox_wh, model.anchors_wh)
            iou, idx_yolo_layer = torch.max(iou, dim=-1)

            grid_h, grid_w = batch_target_tensor[idx_batch][idx_yolo_layer].shape[:2]

            grid_tx = bbox_xy[0, 0] * grid_w
            grid_ty = bbox_xy[0, 1] * grid_h

            idx_grid_tx = int(torch.floor(grid_tx))
            idx_grid_ty = int(torch.floor(grid_ty))

            if batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] == 1.:
                continue

            single_target_bboxes.append(single_target_bbox[1:])

            tx = grid_tx - torch.floor(grid_tx)
            ty = grid_ty - torch.floor(grid_ty)

            tw = torch.log(bbox_wh[0, 0] / model.anchors_wh[idx_yolo_layer, 0])
            th = torch.log(bbox_wh[0, 1] / model.anchors_wh[idx_yolo_layer, 1])

            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, [0, 1, 2, 3]] = torch.tensor([tx, ty, tw, th])
            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 4] = 1.0
            batch_target_tensor[idx_batch][idx_yolo_layer][idx_grid_ty, idx_grid_tx, 5 + c] = 1.0

        single_target_bboxes = torch.stack(single_target_bboxes)
        iou = iou_xywh(batch_pred_bboxes[idx_batch, :, :4], single_target_bboxes)
        iou, _ = torch.max(iou, dim=-1)

        single_target_tensor = []
        for idx_yolo_layer in range(len(batch_target_tensor[idx_batch])):
            single_target_tensor.append(batch_target_tensor[idx_batch][idx_yolo_layer].view(-1, o))
        single_target_tensor = torch.cat(single_target_tensor)
        single_target_tensor = single_target_tensor.unsqueeze(0)

        single_target_tensor[..., 4] = iou
        batch_target_tensor[idx_batch] = single_target_tensor

    batch_target_tensor = torch.cat(batch_target_tensor, dim=0)

    return batch_target_tensor