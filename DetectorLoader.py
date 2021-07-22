import paddle
from Detection.Utils import ResizePadding, non_max_suppression
import numpy as np
from paddle.vision.transforms import to_tensor
from Detection.Models import Darknet
class TinyYOLOv3_onecls(object):
    """Load trained Tiny-YOLOv3 one class (person) detection model.
    Args:
        input_size: (int) Size of input image must be divisible by 32. Default: 416,
        config_file: (str) Path to Yolo model structure config file.,
        weight_file: (str) Path to trained weights file.,
        nms: (float) Non-Maximum Suppression overlap threshold.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=416,
                 config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                 weight_file='Models/yolo-tiny-onecls/best-model.pdparams',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size
        self.model = Darknet(config_file)
        self.model.set_state_dict(paddle.load(weight_file))
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)

    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) Resize to input_size before feed and will return bboxs
                with scale to image original size.,
            expand_bb: (int) Expand boundary of the boxs.
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)
        image = to_tensor(image).unsqueeze(0)
        scf = paddle.min(self.input_size / paddle.to_tensor([image_size], dtype='float32'), 1)[0]

        detected = self.model(image)
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
        if detected is not None:
            detected[:, 0] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, 2] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, 1] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 3] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2].numpy() - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4].numpy() + expand_bb)

        return detected