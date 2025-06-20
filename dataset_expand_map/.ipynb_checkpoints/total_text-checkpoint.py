import warnings
warnings.filterwarnings("ignore")
import os
import re
import numpy as np
import scipy.io as io
import sys
sys.path.append("..")
import util
from util import strs
# from util.strs import remove_all
from dataset_expand_map.data_util import pil_load_img
from dataset_expand_map.dataload_expend import TextDataset, TextInstance
import cv2
from util.config import config as cfg
# from util.io import read_lines


class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        # 相当于for..in..if逻辑功能
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

    @staticmethod
    def parse_mat(mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path + ".mat")
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = libio.read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line, '\xef\xbb\xbf')
            gt = line.split(',')
            xx = gt[0].replace("x: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            yy = gt[1].replace("y: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            try:
                xx = [int(x) for x in re.split(r" *", xx)]
                yy = [int(y) for y in re.split(r" *", yy)]
            except:
                xx = [int(x) for x in re.split(r" +", xx)]
                yy = [int(y) for y in re.split(r" +", yy)]
            if len(xx) < 4 or len(yy) < 4:  # too few points
                continue
            text = gt[-1].split('\'')[1]
            try:
                ori = gt[-2].split('\'')[1]
            except:
                ori = 'c'
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))
        # print(polygon)
        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)
        # polygons = self.parse_carve_txt(annotation_path)

        ratio = cfg.ratio

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path, ratio = ratio)

        # return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import time
    from util.augmentation import Augmentation, BaseTransformNresize
    from util import canvas as cav

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    # transform = Augmentation(
    #     size=640, mean=means, std=stds
    # )
    transform = BaseTransformNresize(
        mean=means, std=stds
    )

    trainset = TotalText(
        data_root='../data/total-text-mat',
        ignore_list=None,
        is_training=False,
        transform=transform
    )

    pwd = os.getcwd()  # 绝对路径
    print("pwd is {}".format(pwd))



    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, expand_mask, train_expand_maks, meta = trainset[idx]
        # img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)
        cv2.imwrite(os.path.join(pwd, "test/img_idx3/img{}.png".format(idx)), img)
        # cv2.imshow("img{}".format(idx), img)
        # cv2.waitkey(4)
        print("img_readed")


        if idx ==3:
            gt_contour = []
            for annot, n_annot in zip(meta['annotation'], meta['n_annotation']):
                if n_annot.item() > 0:
                    gt_contour.append(np.array(annot[:n_annot], dtype=np.int))
            image_show = cv2.polylines(img, gt_contour, True, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(pwd, "test/img_idx3/image_show{}.png".format(idx)), image_show)
            print("img_gt_contour_showed")


            # heatmap_train_mask = cav.heatmap(np.array(train_mask_temp * 255, dtype=np.uint8))
            # train_mask = train_mask[...,None]
            # print(train_mask.mean())
            # import cv2
            # cv2.imwrite(os.path.join(pwd, "test/img_idx3/heatmap_train_mask11111{}.png".format(0)),train_mask*255)


            # 膨胀图
            expand_mask = expand_mask[...,None]
            import cv2
            cv2.imwrite(os.path.join(pwd, "test/img_idx3/instance_expand_mask{}.png".format(idx)),expand_mask*255)

            # 膨胀掩码图
            train_expand_maks = train_expand_maks[...,None]
            import cv2
            cv2.imwrite(os.path.join(pwd, "test/img_idx3/instance_train_expand_maks{}.png".format(idx)),train_expand_maks*255)

            # 膨胀掩码图
            for i in range(tr_mask.shape[2]):
                heatmap = cav.heatmap(np.array(tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]), dtype=np.uint8))
                # heatmap = np.array(tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]), dtype=np.uint8)
                # # cv2.imshow("tr_mask_{}".format(i),heatmap)
                cv2.imwrite(os.path.join(pwd,"test/img_idx3/tr_mask_heatmap/heatmap{}.png".format(i)), heatmap*255)
                # cv2.imwrite(os.path.join(pwd, "test/img_idx3/heatmap_tr_maks{}.png".format(i)),
                #             tr_mask[:, :, i] * 255)
            break
