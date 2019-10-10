

from pytorchgo.utils import logger
import os
import numpy as np
import cv2
import torch,random
import torch.utils.data as data
from .rect import draw_det_box
from PIL import Image

coco_cat2trainid = {
'toilet':62,
'teddy bear':78,
'cup':42,
'bicycle':2,
'kite':34,
'carrot':52,
'stop sign':12,
'tennis racket':39,
'donut':55,
'snowboard':32,
'sandwich':49,
'motorcycle':4,
'oven':70,
'keyboard':67,
'scissors':77,
'airplane':5,
'couch':58,
'mouse':65,
'fire hydrant':11,
'boat':9,
'apple':48,
'sheep':19,
'horse':18,
'banana':47,
'baseball glove':36,
'tv':63,
'traffic light':10,
'chair':57,
'bowl':46,
'microwave':69,
'bench':14,
'book':74,
'elephant':21,
'orange':50,
'tie':28,
'clock':75,
'bird':15,
'knife':44,
'pizza':54,
'fork':43,
'hair drier':79,
'frisbee':30,
'umbrella':26,
'bottle':40,
'bus':6,
'bear':22,
'vase':76,
'toothbrush':80,
'spoon':45,
'train':7,
'sink':72,
'potted plant':59,
'handbag':27,
'cell phone':68,
'toaster':71,
'broccoli':51,
'refrigerator':73,
'laptop':64,
'remote':66,
'surfboard':38,
'cow':20,
'dining table':61,
'hot dog':53,
'car':3,
'sports ball':33,
'skateboard':37,
'dog':17,
'bed':60,
'cat':16,
'person':1,
'skis':31,
'giraffe':24,
'truck':8,
'parking meter':13,
'suitcase':29,
'cake':56,
'wine glass':41,
'baseball bat':35,
'backpack':25,
'zebra':23,
}
coco_trainid2cat = {value:key for key,value in coco_cat2trainid.items()}
COCO14_CLASS = [coco_trainid2cat[i] for i in range(1,81)]
PASCAL_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
IMG_MEAN_RGB = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)

coco14_json_path = "data/list/cl_coco14.json"
voc07_json_path = "data/list/cl_voc07.json"
voc12_json_path = "data/list/cl_voc12.json"



IS_DEBUG = 0
if IS_DEBUG == 1:
    print("IS_Debug is true!")


def generate_tuple(positive_imgsets, shot_num):
    chosen_class_id = random.choice(list(positive_imgsets.keys()))
    positive_imgset = positive_imgsets[chosen_class_id]

    _index = list(range(len(positive_imgset)))
    random.Random().shuffle(_index)
    _index = _index[:(shot_num+ 1)]
    support_index = _index[:shot_num]
    query_index = _index[-1]

    return support_index, query_index, chosen_class_id

class FSLDataset(data.Dataset):
    def __init__(self, params, image_size=(300, 300),ds_name="cl_voc12",query_image_augs=None):
        self.image_size = image_size
        self.params = params
        self.ds_name = ds_name

        if ds_name == "cl_voc12":
            self._CLASS = PASCAL_CLASS
            self.json_path = voc12_json_path
        elif ds_name == "cl_voc07":
            self._CLASS = PASCAL_CLASS
            self.json_path = voc07_json_path
        elif ds_name == "cl_coco14":
            self._CLASS = COCO14_CLASS
            self.json_path = coco14_json_path

        else:
            raise


        self.load_items()

        if "val"  in self.params['image_sets']:
            self._data = self.data_json['exp'][str(self.params['subset_id'])]["val"]
            self.data_size = len(self._data)
        elif "test"  in self.params['image_sets']:
            self._data = self.data_json['exp'][str(self.params['subset_id'])]["test"]
            self.data_size = len(self._data)
        elif "train" in  self.params['image_sets']:
            self.data_size = 5000  # TODO when training
        else:
            raise
        self.positive_imgsets = self.data_json[self.params['image_sets']]["positive_images"]

        self.query_image_augs = query_image_augs




    def load_items(self):
        with open(self.json_path, 'r') as f:
            import json
            logger.warn("loading  data from json file {}....".format(self.json_path))
            self.data_json = json.load(f)



    def __getitem__(self, index):
        def get_item(idx):
            if "train" not in self.params['image_sets']:#val or test
                class_id = str(self._data[idx]['class_id'])
                support_index = self._data[idx]['support_index'][:self.params['k_shot']]
                query_index = self._data[idx]['query_index']

                support_imgset = self.positive_imgsets[class_id]
                query_imgset = self.positive_imgsets[class_id]

            else:#train
                support_index, query_index, class_id = \
                    generate_tuple(self.positive_imgsets, self.params['k_shot'])
                support_imgset = self.positive_imgsets[class_id]
                query_imgset = self.positive_imgsets[class_id]


            def get_img_path(file_name):
                if "coco" in self.ds_name:
                    if "train" in file_name:
                        return os.path.join("dataset/coco14/train2014",file_name)
                    elif "val" in file_name:
                        return os.path.join("dataset/coco14/val2014", file_name)
                    else:
                        raise
                elif "voc" in self.ds_name:
                    return os.path.join("dataset/", file_name)#VOCdevkit
                else:
                    raise

            metadata = dict(
                class_id = class_id,
                class_name = self._CLASS[int(class_id) - 1],
                img_name = query_imgset[query_index]['name'],
                query_image_path = get_img_path(query_imgset[query_index]['img_path']),
            )
            return [get_img_path(support_imgset[v]['img_path']) for v in support_index], \
                   [support_imgset[v]['bbox'] for v in support_index],\
                    get_img_path(query_imgset[query_index]['img_path']), \
                    query_imgset[query_index]['bbox'], \
                    metadata

        def read_BGR_PIL(img_path):
            result_image = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)
            return result_image[:, :, [2, 1, 0]]  # RGB to BGR for later ass opencv imwrite of network feed!

        support_images, support_bboxs, query_image, query_bbox, metadata = get_item(index)

        query_bboxs_original = np.stack(query_bbox, axis=0)
        query_image = read_BGR_PIL(query_image)

        query_image = cv2.resize(query_image, self.image_size)  # resize
        cl_query_image = np.copy(query_image).astype(np.uint8)

        k_shot = len(support_images)
        cl_support_images = []
        output_support_images_concat = []

        for k in range(k_shot):
            support_image = support_images[k]
            support_image = read_BGR_PIL(support_image)

            support_image = cv2.resize(support_image, self.image_size)  # resize

            origin_support_image = np.copy(support_image)
            height, width, channels = support_image.shape
            support_mask = np.zeros((height, width), np.float32)
            bboxs = support_bboxs[k]

            for bbox in bboxs:
                min_x = bbox[0]
                min_y = bbox[1]
                max_x = bbox[2]
                max_y = bbox[3]
                min_x = int(min_x * width)  # normalize 1
                max_x = int(max_x * width)
                min_y = int(min_y * height)
                max_y = int(max_y * height)
                support_mask[min_y:max_y, min_x:max_x] = 1

                _bbox = np.array(bbox[:4]).reshape((4))

                origin_support_image = draw_det_box(origin_support_image, _bbox, class_name=metadata['class_name'],
                                                    color=(255, 0, 0))

            origin_support_image = cv2.resize(origin_support_image, self.image_size).astype(np.uint8)
            cl_support_images.append(origin_support_image)

            masked = np.copy(support_image)
            masked -= IMG_MEAN_RGB  # sub BGR mean
            masked = masked.transpose((2, 0, 1))  # W,H,C->C,W,H

            output_support_images_concat.append(masked)

        if False:
            to_be_drawed = []
            for support_img in cl_support_images:
                to_be_drawed.append(support_img)
            to_be_drawed.append(draw_det_box(query_image, query_bboxs_original[0], class_name=metadata['class_name'],
                                                    color=(255, 0, 0)))
            im = Image.fromarray(np.uint8(np.concatenate(to_be_drawed, axis=1)))
            im.save(
                "{}_query_image_{}.jpg".format(index, metadata["class_name"]))
            print("class_name: {}".format(metadata["class_name"]))


        for i, bb in enumerate(query_bbox):
            bb.append(0)  # add default class, notice here!!!

        if self.query_image_augs is not None:
            query_bbox_np = np.stack(query_bbox, axis=0)
            img, boxes, labels = self.query_image_augs(query_image, query_bbox_np[:, :4], query_bbox_np[:, 4])
            img -= IMG_MEAN_RGB
            query_image = img.transpose((2, 0, 1))  # W,H,C->C,W,H
            query_bbox = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:

            query_image -= IMG_MEAN_RGB
            query_image = query_image.transpose((2, 0, 1))  # W,H,C->C,W,H

        output_support_images_concat = np.stack(output_support_images_concat, axis=0)
        output_support_images_concat = np.squeeze(output_support_images_concat)  # only for one-shot!!!

        for iii in range(query_bboxs_original.shape[0]):
            cl_query_image = draw_det_box(cl_query_image, query_bboxs_original[iii], color=(255, 0, 0),
                                           class_name=metadata['class_name'])

        metadata['cl_query_image'] = cl_query_image
        metadata['cl_support_images'] = cl_support_images

        return torch.from_numpy(output_support_images_concat.copy()), torch.from_numpy(
            query_image.copy()), query_bbox, metadata

    def __len__(self):
        return self.data_size



def detection_collate(batch):
    support_images = []
    query_images = []
    query_bboxes = []
    metadata_list = []
    for sample in batch:
        support_images.append(sample[0])
        query_images.append(sample[1])
        query_bboxes.append(torch.FloatTensor(sample[2]))
        metadata_list.append(sample[3])
    return torch.stack(support_images, 0), torch.stack(query_images, 0), query_bboxes, metadata_list



if __name__ == '__main__':
    params = {
        "image_sets": 'test',
        "k_shot": 3,
        "subset_id": "1",
    }
    dataset = FSLDataset(params, query_image_augs=None)
    data_loader = data.DataLoader(dataset, batch_size=1, num_workers=1,
                                  shuffle=False, pin_memory=True,collate_fn=detection_collate)

    for idx,data in enumerate(data_loader):
        print(idx)