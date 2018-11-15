import random
import argparse
import numpy as np
from setting.config_manager import ConfigDecoder
import os
import json

#https://github.com/experiencor/basic-yolo-keras/blob/master/gen_anchors.py
argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='./setting/window_configure.json',
    help='path to configuration file')

argparser.add_argument(
    '-a',
    '--anchors',
    default=5,
    help='number of anchors to use')

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def print_anchors(centroids,last_grid):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.4f,%0.4f, ' % (anchors[i,0]*last_grid, anchors[i,1]*last_grid)

    #there should not be comma after last anchor, that's why
    r += '%0.4f,%0.4f' % (anchors[sorted_indices[-1:],0]*last_grid, anchors[sorted_indices[-1:],1]*last_grid)
    r += "]"
    print(r)

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def main(argv):
    config_path = args.conf
    num_anchors = args.anchors

    config = ConfigDecoder(config_path)
    data_path = config.get_path("data_path")
    last_grid = config.get_model("output_shape")
    data_list = []
    annotation_dims = []

    with open(data_path + 'train.txt') as f:
        datas = f.readlines()
        for par in datas:
            data_list.append(par.split('\n')[0])

    for file_name in data_list:
        f = open(data_path + file_name + '.txt', "r")
        for info in f.readlines():
            clazz, cx, cy, w, h = [float(i) for i in info.split(' ')]
            annotation_dims.append((w, h))
    # run k_mean to find the anchors

    annotation_dims = np.array(annotation_dims)

    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.5f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids,last_grid)

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)