import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy
from skimage import io
import math
import copy
import time
import os

def adjacent(array, x, y, size = 10):
    # 使用这个函数来存储一个像素点的邻近点到一个列表中
    adjacent = []
    for i in range(-size,size+1):
        for j in range(-size,size+1):
            if x+i >= 0 and y+j>=0 and x+i < array.shape[0] and y+j < array.shape[1]:
                adjacent.append((x+i, y+j))
    return adjacent

def dilated_adjacent(array, overlapping_x, overlapping_y, size = 15):
    global _adj
    tmp = np.zeros(array.shape)
    tmp[overlapping_x, overlapping_y] = array[overlapping_x, overlapping_y].copy()
    
    kernel = np.ones((size, size), np.uint8)
    tmp = cv2.dilate(tmp, kernel, iterations=1)
    tmp[overlapping_x, overlapping_y] -= array[overlapping_x, overlapping_y]
    
    # 测试用
#     _adj.append(tmp)
    
    xs, ys = np.where(tmp > 0)
    ret = []
    for x, y in zip(xs, ys):
        ret.append((x, y))
    return ret
    
# 两个函数用来计算两个点连成射线后的点集中是否有可以match的区域
def line_detect(img, point1, point2, sign):
    point1_x, point1_y = point1
    point2_x, point2_y = point2
    if point1_x != point2_x:
        k = float((point1_y - point2_y)/(point1_x - point2_x))
        b = ((point1_y - k*point1_x)+(point2_y - k*point2_x))/2
        for x in range(img.shape[1]):
            y = int(k*x+b)
            if y>=0 and y<img.shape[0]:
                if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                    continue
                else:
                    return img[x,y]
                
        for y in range(img.shape[0]):
            x = int((y-b)/k)
            if x>=0 and x<img.shape[1]:
                if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                    continue
                else:
                    return img[x,y]
    else:
        x = point1_x
        for y in range(img.shape[0]):
            if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                    continue
            else:
                return img[x,y] 
    return 0

def ray_detect(img, point_start, point_end, sign):
    point1_x, point1_y = point_start
    point2_x, point2_y = point_end
    direction_x = point2_x - point1_x
    direction_y = point2_y - point1_y
    if point1_x != point2_x:
        k = float((point1_y - point2_y)/(point1_x - point2_x))
        b = ((point1_y - k*point1_x)+(point2_y - k*point2_x))/2
        if direction_x > 0:
            for x in range(point2_x, img.shape[1]):
                y = int(k*x+b)
                if y>=0 and y<img.shape[0]:
                    if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                        continue
                    else:
                        return int(img[x,y])
        if direction_x < 0:
            for x in range(0, point2_x):
                y = int(k*x+b)
                if y>=0 and y<img.shape[0]:
                    if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                        continue
                    else:
                        return int(img[x,y])
        if direction_y > 0: 
            for y in range(point2_y, img.shape[0]):
                x = int((y-b)/k)
                if x>=0 and x<img.shape[1]:
                    if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                        continue
                    else:
                        return int(img[x,y])
        if direction_y < 0: 
            for y in range(0, point2_y):
                x = int((y-b)/k)
                if x>=0 and x<img.shape[1]:
                    if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                        continue
                    else:
                        return int(img[x,y])
    else:
        x = point1_x
        for y in range(img.shape[0]):
            if img[x,y] == 1 or img[x,y] == -1 or img[x,y] == sign:
                continue
            else:
                return int(img[x,y])
    return 0

class UnionFind(object):
    def __init__(self, classes):
        self.__parent = np.concatenate([np.array([0, 1]), classes])
    def find(self, x):
        if x == self.__parent[x]:
            return x
        else:
            self.__parent[x] = self.find(self.__parent[x])
            return self.__parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if (px != py):
            self.__parent[px] = py

def crossing_reconstruct(image, overlapped, non_overlapped):
    ### watershed for crossing areas
    kernel = np.ones((1,1),np.uint8)
    opening_crossing = cv2.morphologyEx(overlapped,cv2.MORPH_OPEN, kernel)
    # sure foreground area
    sure_fg_crossing = cv2.erode(opening_crossing,kernel,iterations=1)
    # sure background area
    sure_bg_crossing = cv2.dilate(opening_crossing,kernel,iterations=1)
    crossing_edge = cv2.subtract(sure_bg_crossing,sure_fg_crossing)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg_crossing)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[crossing_edge == 1] = 0
    markers_crossing = cv2.watershed(image, markers)
    
    # overlapping_class列表存储crossing有的类别，例如，有两个crossing重叠区域，overlapping_class列表中就有两个元素
    overlapping_class = [i for i in range(2, np.max(markers_crossing)+1)]
    overlapping_x_class = {}
    overlapping_y_class = {}
    
    # iterate through all markers values
    for i in overlapping_class:
        overlapping_x_class[i], overlapping_y_class[i] = np.where(markers == i)

    # overlapping_center_class是一个字典，存储每个crossing区域的中点坐标
    overlapping_center_class = {}
    for i in overlapping_class:
        overlapping_center_class[i] = (int(np.mean(overlapping_x_class[i])),int(np.mean(overlapping_y_class[i])))

    # overlapped_areas是crossing区域和相邻size = k的区域的点集，k可以在adjacent函数中调整
    overlapped_areas = {}
    for i in overlapping_class:
        overlapped_areas[i] = dilated_adjacent(overlapped, overlapping_x_class[i], overlapping_y_class[i],
                                   size = 9)
#         radius = int(np.sqrt(len(markers_crossing[markers_crossing == i]) / 2.0))
#         overlapped_areas[i] = adjacent(overlapped, overlapping_center_class[i][0], overlapping_center_class[i][1],
#                                        size = radius)
    
    # 对non-overlapping的部分watershed处理
    ### noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(non_overlapped, cv2.MORPH_OPEN, kernel)

    # sure foreground area
    sure_fg = cv2.erode(opening,kernel,iterations=1)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=2)
    unknown = cv2.subtract(sure_bg,sure_fg)
    sure_fg = np.uint8(sure_fg)

    # Marker labelling
    hole = sure_fg.copy()
    cv2.floodFill(hole,None,(0,0),255) # 找到洞孔
    hole = cv2.bitwise_not(hole)
    ret, markers = cv2.connectedComponents(hole)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0
    # watershed
    markers = cv2.watershed(image, markers)
    class_num = np.max(markers)
    non_overlapping_class = np.arange(2, class_num + 1)

    # overlapping_match_nonoverlapping是一个字典，用来存储和crossing区域相邻的non-overlapping区域的编号标记
    overlapping_match_nonoverlapping = {}
    for i in overlapping_class:
        overlapping_match_nonoverlapping[i] = set()
        for j in overlapped_areas[i]:
            if markers[j[0], j[1]] > 1:
                overlapping_match_nonoverlapping[i].add(markers[j[0], j[1]])  

    # overlapping_match_nonoverlapping_center用来存储和crossing区域相邻的non-overlapping子区域的区域中心坐标
    overlapping_match_nonoverlapping_center = {}
    for i in overlapping_match_nonoverlapping:
        adj_x, adj_y = zip(*overlapped_areas[i])
        adj_x, adj_y = np.array(adj_x), np.array(adj_y)

        overlapping_match_nonoverlapping_center[i] = {}
        for k in overlapping_match_nonoverlapping[i]:
            k_adj_x = adj_x[np.where(markers[adj_x, adj_y] == k)[0]]
            k_adj_y = adj_y[np.where(markers[adj_x, adj_y] == k)[0]]
            c = (int(np.mean(k_adj_x)), int(np.mean(k_adj_y)))
    #             temp_x, temp_y = np.where(markers == k)
    #             c = (int(np.mean(temp_x)),int(np.mean(temp_y)))
            overlapping_match_nonoverlapping_center[i][k] = c

    # UnionFind for combination
    uf = UnionFind(non_overlapping_class)

    output = copy.deepcopy(markers)
    for i in overlapping_match_nonoverlapping:
        temp_area = np.ones(markers.shape)
        for (x, y) in overlapped_areas[i]:
            temp_area[x, y] = markers[x, y]

        # Ray detect of non-overlapping part in a cycle
        for j in overlapping_match_nonoverlapping_center[i]:
            tag = ray_detect(temp_area, overlapping_match_nonoverlapping_center[i][j], overlapping_center_class[i], j)
            if tag in overlapping_match_nonoverlapping_center[i]:
                uf.union(tag, j)

    chromosome = {}  # dictionary key is non-overlap class for chromosome, value is overlapping region class

    for tag in non_overlapping_class:
        output[output == tag] = uf.find(tag)
        chromosome[uf.find(tag)] = []

    for i in overlapping_class:
        for j in non_overlapping_class:
            if j in overlapping_match_nonoverlapping[i] and i not in chromosome[uf.find(j)]:
                chromosome[uf.find(j)].append(i)  

    cp_image = []
    for i in chromosome:
        mask = np.zeros(output.shape)
        mask[(output == i)] = 1
        for j in chromosome[i]:
            mask[markers_crossing == j] = 1
        cp_image.append(mask)
        
    return cp_image