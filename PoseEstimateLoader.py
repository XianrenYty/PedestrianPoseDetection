import os
import cv2
import paddle
from SPPE.src.main_fast_inference import InferenNet_fastRes50
from SPPE.src.utils.img import crop_dets
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction


class SPPE_FastPose(object):
    def __init__(self,
                 backbone,
                 input_height=320,
                 input_width=256,
                 device='cuda'):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device

        self.model = InferenNet_fastRes50()
        self.model.eval()

    def predict(self, image, bboxs, bboxs_scores):
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        pose_hm = self.model(inps)

        # Cut eyes and ears.
        pose_hm = paddle.concat([pose_hm[:, :1, :, :], pose_hm[:, 5:, :, :]], axis=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
        return result