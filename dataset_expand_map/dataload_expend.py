import copy
import cv2
import torch
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
from util.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, norm2, split_edge_seqence_by_step
import math
from shapely.geometry import Polygon
import pyclipper
import os
from shapely.geometry import Polygon
import pyclipper


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None

        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        # 根据交通文本测试集标记进行修改
        # if self.text == "1":
        #     self.label = 1
        # else:
        #     self.label = -1


        remove_points = []
        if len(points) > 4:
            # 移除多余的顶点
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text（长度为2）
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)







class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.scale = cfg.scale
        self.alpha = cfg.fuc_k
        self.mask_cnt = len(cfg.fuc_k)

    def sigmoid_alpha(self, x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x*k/dm)) - 1)*betak
        return np.maximum(0, res)

    # Vatti clipping algorithm
    def shrink_polygon_pyclipper(self, polygon, shrink_ratio):
        polygon_shape = Polygon(polygon)  # 创建多边形
        distance = min(int(polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length),20)  # 计算论文中的D距离
        # print("distance is {}".format(distance))
        subject = [tuple(l) for l in polygon]  # 每个点坐标
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)  # -distance就是缩小shrink
        if shrinked == []:
            shrinked = np.array(shrinked)
        else:
            # shrinked = np.array(shrinked[0])
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
        return shrinked

    # 对每个polygon(TextInstance)进行shrunk
    def get_expand_map(self, polygon, expand_mask, train_expand_mask, ratio):
        polygon1 = polygon.points

        # 如果已经标记为忽略的文本实例，则进行全名掩码
        if (Polygon(polygon1).area <=0 or polygon.text == '#'):
            cv2.fillPoly(train_expand_mask, [polygon.points.astype(np.int32)], color=(0,))

        # 对每个polygon进行shrunk，不进行shrunk，直接标注原始多边形
        else:
            try:
                shrinked = self.shrink_polygon_pyclipper(polygon1, ratio)

                if shrinked.size == 0:  # shrink后如果这个区域没了，那么也mask一下，忽略改文字区域
                    cv2.fillPoly(train_expand_mask, polygon1.astype(np.int32)[np.newaxis, :, :], color=(0,))


                # print("[shrinked.astype(np.int32)].shape is {}".format(shrinked.astype(np.int32).shape))
                cv2.fillPoly(expand_mask, [shrinked.astype(np.int32)], color=(1,))
            except Exception as e:
                print(e)
                print('area:', Polygon(polygon1).area)

    # 需要根据图像尺寸再修改
    def make_text_region(self, img, polygons, ratio):


        # self.scale = 1
        h, w = img.shape[0]//self.scale, img.shape[1]//self.scale
        # 图像全部置为1
        mask_ones = np.ones(img.shape[:2], np.uint8)
        # 图像全部置为0
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w, self.mask_cnt), np.float)
        expand_mask = np.zeros((h, w), np.uint8)
        train_expand_mask = np.ones((h, w), np.uint8)

        if polygons is None:
            return tr_mask, train_mask, expand_mask, train_expand_mask

        # pwd = os.getcwd()  # 绝对路径
        # print("make text region pwd is {}".format(pwd))


        for polygon in polygons:
            # print("polygon type is {}".format(type(polygon))) # TextInstance
            instance_mask = mask_zeros.copy()

            # 得到膨胀实例图
            self.get_expand_map(polygon, expand_mask, train_expand_mask, ratio)

            # 可以用来填充任意形状的图型.可以用来绘制多边形,
            # fillPoly（） ： 多个多边形填充
            # 函数原型——
            # cv2.fillPoly(image, [多边形顶点array1, 多边形顶点array2, …], RGB color)
            cv2.fillPoly(instance_mask, [polygon.points.astype(np.int32)], color=(1,))

            # cv2.imwrite(os.path.join(pwd, "test_train/instance_mask1.png"), expand_mask)


            # 用于距离转换，计算图像中非零点1到最近背景点（即0）的距离，dmp表示距离图
            dmp = ndimg.distance_transform_edt(instance_mask[::self.scale, ::self.scale])  # distance transform
            for i, k in enumerate(self.alpha):
                # 表示以重叠部分区域的最大值最为最终的真值，对应通道数有阿尔法个长度值，使用其中文本像素概率表示文本区域
                tr_mask[:, :, i] = np.maximum(tr_mask[:, :, i], self.sigmoid_alpha(dmp, k))


            # 在全1中将不能识别的文本实例标注多边形使用0抹去，表示抹去没有意义的文本实例标注多边形
            if polygon.text == '#':
                cv2.fillPoly(mask_ones, [polygon.points.astype(np.int32)], color=(0,))
                continue


        expand_mask = expand_mask[::self.scale, ::self.scale]
        train_mask = mask_ones[::self.scale, ::self.scale]
        train_expand_mask = train_expand_mask[::self.scale, ::self.scale]

        # cv2.imwrite(os.path.join(pwd, "test_train/train_mask.png"), train_mask)

        # tr_mask：表示在全0中将标注的文本实例多边形使用1表示出；
        # train_mask：表示在全1中抹去将没有意义的文本实例标注多边形（使用0表示）
        # expand_mask:表述对每个文本实例进行膨胀处理

        expand_mask = expand_mask[...,None]
        train_expand_mask = train_expand_mask[...,None]

        expand_mask = expand_mask.transpose(2, 0, 1)
        train_expand_mask = train_expand_mask.transpose(2, 0, 1)
        return tr_mask, train_mask, expand_mask, train_expand_mask


    def get_training_data(self, image, polygons, image_id, image_path, ratio):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
            h, w, _ = image.shape

        # tr_mask, train_mask = self.make_text_region(image, polygons, ratio)
        tr_mask, train_mask, expand_mask, train_expand_mask = self.make_text_region(image, polygons,ratio)
        # clip value (0, 1)将数组a中的所有数限定到范围a_min和a_max中，且数组长度不变
        tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)
        expand_mask = np.clip(expand_mask, 0, 1)
        train_expand_mask = np.clip(train_expand_mask, 0, 1)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W,
            }

            # return image, train_mask, tr_mask, meta
            return image, train_mask, tr_mask, expand_mask, train_expand_mask, meta

        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        tr_mask = torch.from_numpy(tr_mask).float()
        expand_mask = torch.from_numpy(expand_mask).byte()
        train_expand_mask = torch.from_numpy(train_expand_mask).byte()

        return image, train_mask, tr_mask, expand_mask, train_expand_mask

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
