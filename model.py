import cv2
import numpy as np
import torch
from torch import nn

import rexnetv1

import tools
import torchvision.transforms as transforms

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 padding=0,
                 stride=1,
                 dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class Detection(nn.Module):
    def __init__(self,
                 anchor_wh):
        super(Detection, self).__init__()

        self.anchor_wh = anchor_wh

    def forward(self, x, input_img_w, input_img_h):
        x_raw = x.clone()
        x_raw = x_raw.flatten(start_dim=2)
        x_raw = x_raw.transpose(1, 2)

        with torch.no_grad():
            h, w = x.shape[2:]
            grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
            grid_xy = torch.stack([grid_x, grid_y])
            grid_xy = grid_xy.unsqueeze(0).to(x.device)

            anchors_wh = self.anchor_wh.view(1, 2, 1, 1).to(x.device)

            x[:, [0, 1]] = grid_xy + torch.sigmoid(x[:, [0, 1]])  # range: (0, feat_w), (0, feat_h)
            x[:, [2, 3]] = anchors_wh * torch.exp(x[:, [2, 3]])  # range: (0, input_img_w), (0, input_img_h)

            x[:, 0] = x[:, 0] / w
            x[:, 1] = x[:, 1] / h

            x[:, 2] = x[:, 2] / input_img_w
            x[:, 3] = x[:, 3] / input_img_h

            x[:, 4] = torch.sigmoid(x[:, 4])
            x[:, 5:] = torch.sigmoid(x[:, 5:])

            x = x.flatten(start_dim=2)
            x = x.transpose(1, 2)

        return x_raw, x


class LightWeightFaceDetector(nn.Module):
    def __init__(self,
                 num_classes=5,
                 anchors_wh=[[34.86, 62.54],
                             [70.43, 119.33],
                             [134.76, 209.19]],
                 anchors_mask=[[0], [1], [2]],
                 strides=[8, 16, 32]):
        super(LightWeightFaceDetector, self).__init__()

        self.num_classes = num_classes

        self.anchors_wh = torch.tensor(anchors_wh, dtype=torch.float32)
        self.anchors_mask = torch.tensor(anchors_mask, dtype=torch.long)

        self.strides = strides

        self.backbone = rexnetv1.ReXNetV1(width_mult=1.0)
        self.backbone.load_state_dict(torch.load('rexnetv1_1.0x.pth'))
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.detection_layers = nn.ModuleList([])

        for mask in self.anchors_mask:
            for anchor_wh in self.anchors_wh[mask]:
                self.detection_layers.append(Detection(anchor_wh=anchor_wh))

        self.pyramid_s32 = Conv_BN_LeakyReLU(1280, 256, 1)
        self.pyramid_s16 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.pyramid_s8 = Conv_BN_LeakyReLU(61, 256, 3, 1)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest')

        self.head_bbox_regression = nn.Sequential(Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                  Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                  Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                  Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                  nn.Conv2d(256, 4, 1))

        self.head_faceness = nn.Sequential(Conv_BN_LeakyReLU(256, 256, 3, 1),
                                           Conv_BN_LeakyReLU(256, 256, 3, 1),
                                           Conv_BN_LeakyReLU(256, 256, 3, 1),
                                           Conv_BN_LeakyReLU(256, 256, 3, 1),
                                           nn.Conv2d(256, 1, 1))

        self.head_classifcation = nn.Sequential(Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                Conv_BN_LeakyReLU(256, 256, 3, 1),
                                                nn.Conv2d(256, self.num_classes, 1))

    def forward(self, x):
        input_img_h, input_img_w = x.shape[2:]

        multi_scale_features = []
        for idx_layer, layer in enumerate(self.backbone.features[:-1]):
            print(x.shape)
            x = layer(x)

            if idx_layer in [7, 13, 21]:
                multi_scale_features.append(x.clone())

        C5 = multi_scale_features[-1]
        C4 = multi_scale_features[-2]
        C3 = multi_scale_features[-3]

        P5 = self.pyramid_s32(C5)
        P4 = self.upsample(P5) + self.pyramid_s16(C4)
        P3 = self.upsample(P4) + self.pyramid_s8(C3)

        batch_multi_scale_raw_bboxes = []  # for training
        batch_multi_scale_bboxes = []  # for inference

        for P, detection_layer in zip([P3, P4, P5], self.detection_layers):
            P_bbox_regression = self.head_bbox_regression(P)
            P_faceness = self.head_faceness(P)
            P_classification = self.head_classifcation(P)
            P = torch.cat([P_bbox_regression, P_faceness, P_classification], dim=1)

            batch_single_scale_raw_bboxes, batch_single_scale_bboxes = detection_layer(P,
                                                                                       input_img_w,
                                                                                       input_img_h)

            batch_multi_scale_raw_bboxes.append(batch_single_scale_raw_bboxes)
            batch_multi_scale_bboxes.append(batch_single_scale_bboxes)

        batch_multi_scale_raw_bboxes = torch.cat(batch_multi_scale_raw_bboxes, dim=1)
        batch_multi_scale_bboxes = torch.cat(batch_multi_scale_bboxes, dim=1)

        return batch_multi_scale_raw_bboxes, batch_multi_scale_bboxes


def nms(dets, scores, nms_thresh=0.45):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

    keep = []  # store the final bounding boxes
    while order.size > 0:
        i = order[0]  # the index of the bbox with highest confidence
        keep.append(i)  # save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def bboxes_filtering(batch_multi_scale_bboxes):
    filtered_batch_multi_scale_bboxes = []

    for single_multi_scale_bboxes in batch_multi_scale_bboxes:
        filtered_single_multi_scale_bboxes = {}

        objectness = single_multi_scale_bboxes[:, 4]
        class_prob, class_idx = torch.max(single_multi_scale_bboxes[:, 5:], dim=1)

        num_classes = single_multi_scale_bboxes[:, 5:].shape[-1]

        confidence = objectness * class_prob
        is_postive = confidence > 1e-4

        position = single_multi_scale_bboxes[is_postive, :4]
        confidence, class_idx = confidence[is_postive], class_idx[is_postive]

        position = position.cpu().numpy()
        confidence = confidence.cpu().numpy()
        class_idx = class_idx.cpu().numpy()

        def xywh2xyxy(box_xywh):
            box_xyxy = box_xywh.copy()
            box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
            box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
            box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
            box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
            return box_xyxy

        # NMS
        keep = np.zeros(len(position), dtype=np.int)
        for i in range(num_classes):
            inds = np.where(class_idx == i)[0]
            if len(inds) == 0:
                continue

            c_bboxes = position[inds]
            c_scores = confidence[inds]
            c_keep = nms(xywh2xyxy(c_bboxes), c_scores, 0.45)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        position = position[keep]
        confidence = confidence[keep]
        class_idx = class_idx[keep]

        filtered_single_multi_scale_bboxes["position"] = position
        filtered_single_multi_scale_bboxes["confidence"] = confidence
        filtered_single_multi_scale_bboxes["class"] = class_idx

        filtered_batch_multi_scale_bboxes.append(filtered_single_multi_scale_bboxes)

    return filtered_batch_multi_scale_bboxes


def yololoss(preds, targets):
    loss_xy_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_wh_func = nn.MSELoss(reduction='none')
    loss_obj_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_class_func = nn.BCEWithLogitsLoss(reduction='none')

    batch_loss_xy = []
    batch_loss_wh = []
    batch_loss_positive_obj = []
    batch_loss_negative_obj = []
    batch_loss_positive_class_obj = []
    batch_loss_negative_class_obj = []

    for single_preds, single_targets in zip(preds, targets):
        t_obj = torch.sum(single_targets[:, 5:], dim=1)

        is_positive = (t_obj == 1.)
        is_negative = (t_obj == 0.)

        positive_single_preds = single_preds[is_positive]
        negative_single_preds_objectness = single_preds[is_negative, 4]

        positive_single_targets = single_targets[is_positive]
        negative_single_targets_objectness = single_targets[is_negative, 4]

        batch_loss_xy.append(loss_xy_func(positive_single_preds[:, [0, 1]], positive_single_targets[:, [0, 1]]))
        batch_loss_wh.append(loss_wh_func(positive_single_preds[:, [2, 3]], positive_single_targets[:, [2, 3]]))

        loss_positive_obj = loss_obj_func(positive_single_preds[:, 4], torch.ones_like(positive_single_targets[:, 4]))
        batch_loss_positive_obj.append(loss_positive_obj)

        loss_negative_obj = loss_obj_func(negative_single_preds_objectness,
                                          torch.zeros_like(negative_single_targets_objectness))
        loss_negative_obj = loss_negative_obj[negative_single_targets_objectness < 0.7]

        batch_loss_negative_obj.append(loss_negative_obj)

        loss_class_obj = loss_class_func(positive_single_preds[:, 5:], positive_single_targets[:, 5:])
        batch_loss_positive_class_obj.append(loss_class_obj[positive_single_targets[:, 5:] == 1.])
        batch_loss_negative_class_obj.append(loss_class_obj[positive_single_targets[:, 5:] == 0.])

    num_positve = len(torch.nonzero(torch.sum(targets[..., 5:], dim=-1) == 1.))
    num_classes = targets[..., 5:].shape[-1]

    batch_loss_xy = torch.sum(torch.cat(batch_loss_xy, dim=0)) / num_positve
    batch_loss_wh = torch.sum(torch.cat(batch_loss_wh, dim=0)) / num_positve
    batch_loss_positive_obj = torch.sum(torch.cat(batch_loss_positive_obj, dim=0)) / num_positve
    batch_loss_negative_obj = torch.sum(torch.cat(batch_loss_negative_obj, dim=0)) / num_positve
    batch_loss_positive_class_obj = torch.sum(torch.cat(batch_loss_positive_class_obj, dim=0)) / num_positve
    batch_loss_negative_class_obj = torch.sum(
        torch.cat(batch_loss_negative_class_obj, dim=0)) / num_classes / num_positve

    loss = batch_loss_xy + batch_loss_wh + \
           batch_loss_positive_obj + batch_loss_negative_obj + \
           batch_loss_positive_class_obj + batch_loss_negative_class_obj

    return loss


if __name__ == '__main__':
    model = LightWeightFaceDetector()
    torch.save(model.state_dict(), "mem_check.pth")

    import dataset
    import torch.utils.data as data

    train_dataset = dataset.DatasetReader(dataset_path="example",
                                          use_augmentation=True)

    data_loader = data.DataLoader(train_dataset, 2,
                                  num_workers=1,
                                  shuffle=False,
                                  collate_fn=dataset.yolo_collate,
                                  pin_memory=True,
                                  drop_last=False)

    epochs = 100
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        print("epoch: ", epoch)
        model.train()
        #
        # if epoch == 10:
        #     for p in model.backbone.parameters():
        #         p.requires_grad = True
        #
        for img, batch_target_bboxes, inds in data_loader:
            batch_multi_scale_raw_bboxes, batch_multi_scale_bboxes = model(img)

            targets = tools.build_target_tensor(model=model,
                                                batch_pred_bboxes=batch_multi_scale_bboxes,
                                                batch_target_bboxes=batch_target_bboxes,
                                                input_size=(384, 384))

            loss = yololoss(batch_multi_scale_raw_bboxes, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
            # for m in model.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.track_running_stats = False

            img = cv2.imread("example/img/3.8.jpg")
            img = cv2.resize(img, (384, 384))
            img = img[:, :, (2, 1, 0)]
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            img = img / 255.
            img = transforms.Normalize(mean=[0.575, 0.533, 0.507], std=[0.235, 0.232, 0.233])(img)
            img = img.unsqueeze(0)

            _, batch_multi_scale_bboxes = model(img)

            filtered_batch_multi_scale_bboxes = bboxes_filtering(batch_multi_scale_bboxes)
            filtered_single_multi_scale_bboxes = filtered_batch_multi_scale_bboxes[0]

            img_draw = cv2.imread("example/img/3.8.jpg")
            img_h, img_w = img_draw.shape[:2]

            for idx in range(len(filtered_single_multi_scale_bboxes['position'])):
                bbox = filtered_single_multi_scale_bboxes['position'][idx]
                bbox[[0, 2]] *= img_w
                bbox[[1, 3]] *= img_h
                print(filtered_single_multi_scale_bboxes['confidence'][idx])
                bbox = bbox.astype(np.int32)

                l = int(bbox[0] - bbox[2] / 2)
                r = int(bbox[0] + bbox[2] / 2)

                t = int(bbox[1] - bbox[3] / 2)
                b = int(bbox[1] + bbox[3] / 2)

                cv2.rectangle(img=img_draw, pt1=(l, t), pt2=(r, b), color=(0, 255, 0))

                print(bbox)
            cv2.imshow('img', img_draw)
            cv2.waitKey(30)
            print(len(filtered_single_multi_scale_bboxes['position']))
