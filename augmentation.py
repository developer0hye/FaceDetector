import cv2
import numpy as np
import random

def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.copy()
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2.
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2.
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2.
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2.
    return box_xyxy

def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.copy()
    box_xywh[:, 0] = (box_xyxy[:, 0] + box_xyxy[:, 2]) / 2.
    box_xywh[:, 1] = (box_xyxy[:, 1] + box_xyxy[:, 3]) / 2.
    box_xywh[:, 2] = box_xyxy[:, 2] - box_xyxy[:, 0]
    box_xywh[:, 3] = box_xyxy[:, 3] - box_xyxy[:, 1]
    return box_xywh

def HorFlip(img, bboxes_xywh, p=0.5):
    if random.random() < p:
        img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
        bboxes_xywh[:, 0] = 1. - bboxes_xywh[:, 0]
        return img, bboxes_xywh
    return img, bboxes_xywh

def RandomTranslation(img, bboxes_xyxy, classes, p=0.5):
    if random.random() < p:
        height, width = img.shape[0:2]

        l_bboxes = round(width * np.min(bboxes_xyxy[:, 0]))
        r_bboxes = width-round(width * np.max(bboxes_xyxy[:, 2]))

        t_bboxes = round(np.min(height * bboxes_xyxy[:, 1]))
        b_bboxes = height-round(height * np.max(bboxes_xyxy[:, 3]))

        tx = random.randint(-l_bboxes, r_bboxes)
        ty = random.randint(-t_bboxes, b_bboxes)

        # translation matrix
        tm = np.float32([[1, 0, tx],
                         [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

        img = cv2.warpAffine(img, tm, (width, height), borderValue=(0, 0, 0))

        bboxes_xyxy[:, [0, 2]] += (tx / width)
        bboxes_xyxy[:, [1, 3]] += (ty / height)
        bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

        return img, bboxes_xyxy, classes
    return img, bboxes_xyxy, classes

def RandomScale(img, bboxes_xyxy, classes,
                scale=[0.5, 2.0],
                p=0.5):

    if random.random() < p:
        height, width = img.shape[0:2]

        while scale[0] <= scale[1]:
            random_scale = random.uniform(scale[0], scale[1])

            cx = width//2
            cy = height//2

            tx = cx - random_scale * cx
            ty = cy - random_scale * cy

            l_outer_bboxes = round(width * np.min(bboxes_xyxy[:, 0])) * random_scale + tx
            r_outer_bboxes = round(width * np.max(bboxes_xyxy[:, 2])) * random_scale + tx

            t_outer_bboxes = round(height * np.min(bboxes_xyxy[:, 1])) * random_scale + tx
            b_outer_bboxes = round(height * np.max(bboxes_xyxy[:, 3])) * random_scale + ty

            min_w_bboxes = random_scale * width * np.min(bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0])
            min_h_bboxes = random_scale * height * np.min(bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])

            if l_outer_bboxes < 0 or r_outer_bboxes >= width or t_outer_bboxes < 0 or b_outer_bboxes >= height:
                scale[1] = random_scale - 0.01
                continue

            if min_w_bboxes < 4 or min_h_bboxes < 4:
                scale[0] = random_scale + 0.01
                continue

            # # scale matrix
            sm = np.float32([[random_scale, 0, tx],
                             [0, random_scale, ty]])  # [1, 0, tx], [1, 0, ty]

            augmented_img = cv2.warpAffine(img, sm, (width, height), borderValue=(0, 0, 0))

            augmented_bboxes_xyxy = random_scale * bboxes_xyxy
            augmented_bboxes_xyxy[:, [0, 2]] += (tx / width)
            augmented_bboxes_xyxy[:, [1, 3]] += (ty / height)
            augmented_bboxes_xyxy = np.clip(augmented_bboxes_xyxy, 0., 1.)

            return augmented_img, augmented_bboxes_xyxy, classes

    return img, bboxes_xyxy, classes

def RandomCrop(img, bboxes_xyxy, classes, p=0.5):
    if random.random() < p:
        height, width = img.shape[0:2]

        min_width = width // 4
        min_height = height // 4

        while (min_width <= width and min_height <= height):
            clipped_w = random.randint(min_width, width)
            clipped_h = random.randint(min_height, height)

            l = random.randint(0, width - clipped_w + 1)
            t = random.randint(0, height - clipped_h + 1)
            r = l + clipped_w
            b = t + clipped_h

            w_bboxes = np.round(width * (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0])).astype(np.int32)
            h_bboxes = np.round(height * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])).astype(np.int32)

            l_clipped_bboxes = np.round(np.clip(width * bboxes_xyxy[:, 0], a_min=l, a_max=r)).astype(np.int32)
            t_clipped_bboxes = np.round(np.clip(height * bboxes_xyxy[:, 1], a_min=t, a_max=b)).astype(np.int32)
            r_clipped_bboxes = np.round(np.clip(width * bboxes_xyxy[:, 2], a_min=l, a_max=r)).astype(np.int32)
            b_clipped_bboxes = np.round(np.clip(height * bboxes_xyxy[:, 3], a_min=t, a_max=b)).astype(np.int32)

            w_clipped_bboxes = r_clipped_bboxes - l_clipped_bboxes
            h_clipped_bboxes = b_clipped_bboxes - t_clipped_bboxes

            inner_bboxes = ((w_clipped_bboxes * h_clipped_bboxes) > 0)

            #0인건 일단 제거
            if not(True in inner_bboxes):
                min_width = clipped_w + 1
                min_height = clipped_h + 1
                continue

            w_bboxes = w_bboxes[inner_bboxes]
            h_bboxes = h_bboxes[inner_bboxes]

            w_clipped_bboxes = w_clipped_bboxes[inner_bboxes]
            h_clipped_bboxes = h_clipped_bboxes[inner_bboxes]

            if np.min(((w_clipped_bboxes * h_clipped_bboxes) / (w_bboxes * h_bboxes))) <= 0.99:
                min_width = clipped_w + 1
                min_height = clipped_h + 1
                continue

            l_clipped_bboxes = ((l_clipped_bboxes[inner_bboxes, np.newaxis]-l)/clipped_w)
            t_clipped_bboxes = ((t_clipped_bboxes[inner_bboxes, np.newaxis]-t)/clipped_h)
            r_clipped_bboxes = ((r_clipped_bboxes[inner_bboxes, np.newaxis]-l)/clipped_w)
            b_clipped_bboxes = ((b_clipped_bboxes[inner_bboxes, np.newaxis]-t)/clipped_h)

            classes = classes[inner_bboxes]

            augmented_bboxes_xyxy = np.concatenate([l_clipped_bboxes, t_clipped_bboxes,
                                                    r_clipped_bboxes, b_clipped_bboxes], axis=-1).astype(np.float32)

            augmented_bboxes_xyxy = np.clip(augmented_bboxes_xyxy, a_min=0., a_max=1.0)
            augmented_img = img[t:b, l:r]
            return augmented_img, augmented_bboxes_xyxy, classes
    return img, bboxes_xyxy, classes
