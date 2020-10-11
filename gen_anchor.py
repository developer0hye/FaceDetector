import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import cv2
import pandas as pd
from numpy.random import RandomState
import tools
import random

def read_wh_from_label_file(path):
    with open(path, 'r') as label:
        objects_information = []

        for line in label:
            line = line.split()
            if len(line) == 5:  # 0: class, 1:x, 2:y, 3:w, 4:h
                object_information = []
                for data in line[3:5]:
                    object_information.append(float(data))
                objects_information.append(object_information)
        objects_information = np.asarray(objects_information).astype(np.float32)

        return objects_information


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)

    return centroids[:k]


def get_IOU(bboxes1, bboxes2):
    # https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-6)
    return iou


def avg_IOU(points, centroids):
    num_points = points.shape[0]
    num_centroids = centroids.shape[0]

    points_xyxy = np.zeros((num_points, 4), dtype=points.dtype)
    centroids_xyxy = np.zeros((num_centroids, 4), dtype=centroids.dtype)

    points_xyxy[:, [2, 3]] = points
    centroids_xyxy[:, [2, 3]] = centroids

    ious = get_IOU(points_xyxy, centroids_xyxy)
    ious = np.max(ious, axis=-1)

    return np.mean(ious)


def whiou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    w1, h1 = np.split(boxA, 2, axis=1)
    w2, h2 = np.split(boxB, 2, axis=1)

    innerW = np.minimum(w1, w2.T)
    innerH = np.minimum(h1, h2.T)

    interArea = np.maximum(innerW, 0) * np.maximum(innerH, 0)
    boxAArea = (w1) * (h1)
    boxBArea = (w2) * (h2)
    iou = interArea / (boxAArea + boxBArea.T - interArea + 1e-6)

    # return the intersection over union value
    return iou


def distance_IOU(points, centroids):
    d = 1. - whiou(points, centroids)
    return d


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = distance_IOU(points, centroids)
    return np.argmin(distances, axis=1)


def move_centroids_mean(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([np.mean(points[closest == k], axis=0) for k in range(centroids.shape[0])])


def printAnchorBoxesAreaOrder(labels_wh, centroids, ax=None):
    print("hi", centroids)
    num_clusters = centroids.shape[0]

    # areas = centroids[:, 0] * centroids[:, 1]
    # sorted_indices = np.argsort(areas)
    # centroids = centroids[sorted_indices]

    closest_centroids = closest_centroid(labels_wh, centroids)
    print(move_centroids_mean(labels_wh, closest_centroids, centroids))

    datas = []
    for idx in range(centroids.shape[0]):
        clustered_wh = labels_wh[closest_centroids == idx]

        centroid_w = np.round(centroids[idx, 0], 4)
        centroid_h = np.round(centroids[idx, 1], 4)

        mean_w = np.mean(clustered_wh[..., 0])
        mean_h = np.mean(clustered_wh[..., 1])

        std_w = np.std(clustered_wh[..., 0], ddof=1)
        std_h = np.std(clustered_wh[..., 1], ddof=1)

        datas.append([centroid_w, centroid_h, mean_w, mean_h, std_w, std_h])
        # if ax is not None:
        #     rect = patches.Rectangle((min_w, min_h),
        #                              max_w - min_w, max_h - min_h,
        #                              linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)

    datas = np.array(datas)
    datas = datas.reshape(-1, 6)
    datas = np.round(datas, 3)
    print(avg_IOU(labels_wh, centroids))

    df = pd.DataFrame(datas,
                      columns=['centroid_w', 'centroid_h', 'mean_w', 'mean_h', 'std_w', 'std_h'])

    print(df.min)
    return centroids


def sort_centroids(centroids):
    areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(areas)
    centroids = centroids[sorted_indices]
    return centroids


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


if __name__ == "__main__":

    np.random.seed(12)
    random.seed(12)

    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default='train',
                        help='dataset path\n')
    parser.add_argument('-output_dir', default='output/', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default=3, type=int,
                        help='number of clusters\n')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_dataset_path = args.path
    annotations = tools.read_annotation_files(os.path.join(train_dataset_path, "annotations"))

    annotations_w = []
    annotations_h = []

    model_input_width = 384
    model_input_height = 384

    for annotation in annotations:
        img_width = int(annotation['annotation']['size']['width'])
        img_height = int(annotation['annotation']['size']['height'])

        bboxes = annotation['annotation']['object']
        for bbox in bboxes:
            bbox = bbox['bndbox']
            xmax = model_input_width*int(bbox['xmax'])/img_width
            xmin = model_input_width*int(bbox['xmin'])/img_width
            ymax = model_input_height*int(bbox['ymax'])/img_height
            ymin = model_input_height*int(bbox['ymin'])/img_height

            annotations_w.append(xmax-xmin)
            annotations_h.append(ymax-ymin)

    annotations_w = np.stack(annotations_w)[:, np.newaxis]
    annotations_h = np.stack(annotations_h)[:, np.newaxis]

    annotations_wh = np.concatenate([annotations_w, annotations_h], axis=1)
    centroids_wh = initialize_centroids(annotations_wh, args.num_clusters)

    for i in range(10):
        closest_centroids = closest_centroid(annotations_wh, centroids_wh)
        centroids_wh = move_centroids_mean(annotations_wh, closest_centroids, centroids_wh)

    centroids_wh = centroids_wh[np.argsort(centroids_wh[:, 0]*centroids_wh[:, 1])]
    print(centroids_wh)
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(annotations_wh[:, 0], annotations_wh[:, 1])

    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.scatter(annotations_wh[:, 0], annotations_wh[:, 1], c=closest_centroid(annotations_wh, centroids_wh))
    ax2.scatter(centroids_wh[:, 0], centroids_wh[:, 1], c='r', s=100)


    # #
    # # f3 = plt.figure()
    # # ax3 = f3.add_subplot(111)
    # # ax3.scatter(labels_wh[:, 0], labels_wh[:, 1], c=closest_centroid(labels_wh, best_centroids_wh))
    # # ax3.scatter(best_centroids_wh[:, 0], best_centroids_wh[:, 1], c='r', s=100)
    # #
    #
    # best_centroids_wh = printAnchorBoxesAreaOrder(labels_wh, centroids=centroids_wh)

    plt.show()
