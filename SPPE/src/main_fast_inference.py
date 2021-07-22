from SPPE.src.models.FastPose import FastPose
import paddle
from paddle import nn
class InferenNet_fastRes50(nn.Layer):
    def __init__(self, weights_file='./Models/sppe/fast_res50_256x192.pdparams'):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17)
        print('Loading pose model from {}'.format(weights_file))
        self.pyranet.set_state_dict(paddle.load(weights_file))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)

        return out
