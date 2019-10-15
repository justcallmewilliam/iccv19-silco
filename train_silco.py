# -*- coding: utf-8 -*-
# Author: Tao Hu <taohu620@gmail.com>

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from layers.modules import MultiBoxLoss
from tqdm import tqdm
from pytorchgo.utils import logger
from data.cl_dataset import FSLDataset, detection_collate
from cl_utils.augmentations import SSDAugmentation
from silco_module import build_ssd, build_ssd_support
import numpy as np


num_classes = 2
iterations = 60000
stepvalues = (40000,)
log_per_iter = 300
save_per_iter = 6000
shot_num = 5
lr = 1e-4
weight_decay = 5e-4
image_size = 300
batch_size = 4


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=image_size, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--batch_size', default=batch_size, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=iterations, type=int, help='Number of training iterations')
parser.add_argument('--lr', '--learning-rate', default=lr, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=weight_decay, type=float, help='Weight decay for SGD')
parser.add_argument('--subset', default="0", choices=['0', '1',], type=str)
parser.add_argument('--dataset', default="cl_voc12", choices=['cl_voc12', 'cl_voc07', 'cl_coco14'], type=str)
parser.add_argument('--shot_num', default=5, type=int, help='shot number')
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_load', default="", type=str)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if "coco" in args.dataset:
    args.iterations = args.iterations*2
    stepvalues = [sv*2 for sv in stepvalues]
    print("coco dataset! iter num is changed to {}".format(args.iterations))

train_setting = {
    "image_sets": 'train',
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}


val_setting = {
    "image_sets": 'val',
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}

test_setting = {
    "image_sets": 'test',
    "k_shot": args.shot_num,
    "subset_id": args.subset,
}

def adjust_learning_rate(optimizer, lr, iteration):
    if iteration>=stepvalues[0]:
        lr = lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




def train():

    logger.info("current cuda device: {}".format(torch.cuda.current_device()))

    few_shot_net = build_ssd(args.dim, num_classes)
    support_net = build_ssd_support(args.dim, num_classes)


    vgg16_state_dict = torch.load(args.basenet)
    new_params = {}
    for index, i in enumerate(vgg16_state_dict):
        #if index >= 20:
        #    continue
        new_params[i] = vgg16_state_dict[i]
        logger.info("recovering weight for student model(loading vgg16 weight): {}".format(i))
    support_net.support_vgg.load_state_dict(new_params)

    logger.info('Loading base network...')
    few_shot_net.query_vgg.load_state_dict(torch.load(args.basenet))



    few_shot_net = few_shot_net.cuda()
    support_net = support_net.cuda()

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    logger.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    few_shot_net.extras.apply(weights_init)
    few_shot_net.loc.apply(weights_init)
    few_shot_net.conf.apply(weights_init)

    optimizer = optim.SGD(list(few_shot_net.parameters()) + list(support_net.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, size=args.dim, overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False, use_gpu=True)


    few_shot_net.train()
    support_net.train()
    best_val_result = 0
    logger.info('Loading Dataset...')

    dataset = FSLDataset(params=train_setting, image_size=(args.dim, args.dim),query_image_augs=SSDAugmentation(args.dim),ds_name=args.dataset)

    epoch_size = len(dataset) // args.batch_size


    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True, collate_fn=detection_collate)
    batch_iterator = iter(data_loader)

    lr=args.lr
    for iteration in tqdm(range(args.iterations + 1),total=args.iterations, desc="training {}".format(logger.get_logger_dir())):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        if iteration in stepvalues:
            lr=adjust_learning_rate(optimizer,lr, iteration)

        # load train data
        first_images, images, targets, metadata = next(batch_iterator)
        #embed()

        first_images = Variable(first_images.cuda())
        images = Variable(images.cuda())
        targets = [Variable(anno.cuda(), volatile=True) for anno in targets]

        fusion_support = support_net(first_images)
        out = few_shot_net(fusion_support, images, is_train =True)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        if iteration % log_per_iter == 0 and iteration > 0:
            logger.info('''LR: {}\t Iter: {}\t Loss_l: {:.5f}\t Loss_c: {:.5f}\t Loss_total: {:.5f}\t best_result: {:.5f}'''.format(lr,iteration,loss_l.data[0],loss_c.data[0], loss.data[0], best_val_result))


        if iteration % save_per_iter == 0 and iteration > 0:
            few_shot_net.eval()
            support_net.eval()
            cur_eval_result = do_eval(few_shot_net, support_net=support_net, test_setting=val_setting, base_dir=logger.get_logger_dir(),
                                      )
            few_shot_net.train()
            support_net.train()

            is_best = True if cur_eval_result > best_val_result else False
            if is_best:
                best_val_result = cur_eval_result
                torch.save({
                    'iteration': iteration,
                    'optim_state_dict': optimizer.state_dict(),
                    'support_model_state_dict': support_net.state_dict(),
                    'query_model_state_dict': few_shot_net.state_dict(),
                    'best_mean_iu': best_val_result,
                }, os.path.join(logger.get_logger_dir(), 'cherry.pth'))
            else:
                logger.info("current snapshot is not good enough, skip~~")

            logger.info('current iter: {} current_result: {:.5f}'.format(iteration, cur_eval_result))

    few_shot_net.eval()
    support_net.eval()
    test_result = do_eval(few_shot_net, support_net, test_setting=test_setting, base_dir=logger.get_logger_dir())
    logger.info(
        "test result={:.5f}, best validation result={:.5f}".format(test_result, best_val_result))
    logger.info("Congrats~")



def do_eval(few_shot_net, support_net, test_setting, base_dir=logger.get_logger_dir()):
    tmp_eval = os.path.join(base_dir, "eval_tmp")
    ground_truth_dir = os.path.join(tmp_eval, "ground_truth")
    predicted_dir = os.path.join(tmp_eval, "detection")
    vis_dir = os.path.join(tmp_eval, "vis")
    mAP_result_dir = os.path.join(tmp_eval, "mAP_result")

    def create_dirs(dir_name):
        global tmp_eval, ground_truth_dir, predicted_dir, vis_dir, mAP_result_dir

        tmp_eval = os.path.join(base_dir, dir_name)
        ground_truth_dir = os.path.join(tmp_eval, "ground_truth")
        predicted_dir = os.path.join(tmp_eval, "detection")
        vis_dir = os.path.join(tmp_eval, "vis")
        mAP_result_dir = os.path.join(tmp_eval, "mAP_result")

        if os.path.isdir(tmp_eval):
            import shutil
            shutil.rmtree(tmp_eval, ignore_errors=True)
        os.makedirs(tmp_eval)
        os.makedirs(ground_truth_dir)
        os.makedirs(predicted_dir)
        os.makedirs(vis_dir)
        os.mkdir(mAP_result_dir)
        return (tmp_eval, ground_truth_dir, predicted_dir, vis_dir, mAP_result_dir)

    tmp_eval, ground_truth_dir, predicted_dir, vis_dir, mAP_result_dir = create_dirs(dir_name="eval_tmp")


    dataset = FSLDataset(params=test_setting, image_size=(args.dim, args.dim), ds_name=args.dataset)
    num_images = len(dataset)

    data_loader = data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=True, collate_fn=detection_collate)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    w = image_size
    h = image_size

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="online {}".format(test_setting['image_sets'])):

        with open(os.path.join(ground_truth_dir, "{}.txt".format(i)), "w") as f_gt:
            with open(os.path.join(predicted_dir, "{}.txt".format(i)), "w") as f_predict:
                # if i > 500:break
                first_images, images, targets, metadata = batch
                class_name = metadata[0]['class_name']

                first_images = Variable(first_images.cuda())
                x = Variable(images.cuda())



                query_origin_img = metadata[0]['cl_query_image']#np.transpose(images.numpy()[0], (1, 2, 0))#W*H*C
                gt_bboxes = targets[0].numpy()
                for _ in range(gt_bboxes.shape[0]):
                    gt_bboxes[_, 0] *= w
                    gt_bboxes[_, 2] *= w
                    gt_bboxes[_, 1] *= h
                    gt_bboxes[_, 3] *= h
                    f_gt.write(
                        "targetobject {} {} {} {}\n".format(int(gt_bboxes[_, 0]), int(gt_bboxes[_, 1]),
                                                    int(gt_bboxes[_, 2]),
                                                    int(gt_bboxes[_, 3])))

                fusion_support = support_net(first_images)
                detections = few_shot_net(fusion_support, x, is_train=False).data


                vis_flag = 0
                # skip j = 0, because it's the background class
                for j in range(1, detections.size(1)):
                    dets = detections[0, j, :]
                    mask = dets[:, 0].gt(0).expand(5, dets.size(0)).t()# greater than 0 will be visualized!
                    dets =  torch.masked_select(dets, mask).view(-1, 5)
                    torch_dets = dets.clone()
                    if dets.dim() == 0:
                        continue
                    boxes = dets[:, 1:].cpu().numpy()
                    boxes[:, 0] = (boxes[:, 0]*w)
                    boxes[:, 1] = (boxes[:, 1]*h)
                    boxes[:, 2] = (boxes[:, 2]*w)
                    boxes[:, 3] = (boxes[:, 3]*h)

                    boxes[:, 0][boxes[:, 0] < 0] = 0
                    boxes[:, 1][boxes[:, 1] < 0] = 0
                    boxes[:, 2][boxes[:, 2] > image_size] = image_size
                    boxes[:, 3][boxes[:, 3] > image_size] = image_size

                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes,scores[:, np.newaxis])).astype(np.float32,copy=False)
                    all_boxes[j][i] = cls_dets

                    for _ in range(cls_dets.shape[0]):
                        f_predict.write(
                            "targetobject {} {} {} {} {}\n".format(cls_dets[_, 4], int(cls_dets[_, 0]), int(cls_dets[_, 1]),
                                                           int(cls_dets[_, 2]), int(cls_dets[_, 3])))



    from cl_utils.mAP_lib.pascalvoc_interactive import get_mAP
    cwd = os.getcwd()
    mAP = get_mAP(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               tmp_eval), "ground_truth", "detection", "mAP_result")
    os.chdir(cwd)
    return mAP


if __name__ == '__main__':
    if args.test:
        if args.test_load == "":
            base_dir = os.path.join(os.path.join("train_log", os.path.basename(__file__).replace(".py", "")))
        else:
            base_dir = args.test_load
        print("start test...")
        query_net = build_ssd(args.dim, num_classes, top_k=200)
        support_net = build_ssd_support(args.dim, num_classes)

        #saved_dict = torch.load(os.path.join(base_dir, "cherry.pth"))
        saved_dict = torch.load(args.test_load)
        support_net.load_state_dict(saved_dict['support_model_state_dict'], strict=True)
        query_net.load_state_dict(saved_dict['query_model_state_dict'], strict=True)

        query_net.eval()
        do_eval(few_shot_net=query_net, support_net=support_net, test_setting=test_setting, base_dir=base_dir)
        print("online validation result: {}".format(saved_dict['best_mean_iu']))

    else:
        logger.set_logger_dir("{}_{}_subset{}".format(os.path.basename(__file__).replace(".py", ""),args.dataset, args.subset))
        train()
