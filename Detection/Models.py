from paddle import nn
import paddle.nn.functional as F
import numpy as np
import paddle
from .Utils import to_cpu, parse_model_config, build_targets

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]  # [3]
    module_list = nn.LayerList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_sublayer(
                f"conv_{module_i}",

                nn.Conv2D(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias_attr=not bn,
                ),
            )
            if bn:
                modules.add_sublayer(f"batch_norm_{module_i}", nn.BatchNorm2D(filters, momentum=0.9, epsilon=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_sublayer(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_sublayer(f"_debug_padding_{module_i}", nn.Pad2D([0, 1, 0, 1]))
            maxpool = nn.MaxPool2D(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_sublayer(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_sublayer(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_sublayer(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_sublayer(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_sublayer(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Darknet(nn.Layer):
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = paddle.concat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module.sublayers()[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(paddle.concat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    a = paddle.Tensor([1, 2, 3])
                    bn_b = paddle.to_tensor(weights[ptr: ptr + num_b]).reshape(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = paddle.to_tensor(weights[ptr: ptr + num_b]).reshape(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = paddle.to_tensor(weights[ptr: ptr + num_b]).reshape(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = paddle.to_tensor(weights[ptr: ptr + num_b]).reshape(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = paddle.to_tensor(weights[ptr: ptr + num_b]).reshape(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = paddle.to_tensor(weights[ptr: ptr + num_w]).reshape(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

class Upsample(nn.Layer):
    """ nn.Upsample is deprecated """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Layer):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Layer):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size

        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = paddle.arange(g).tile([g, 1]).reshape([1, 1, g, g]).cast('float32')
        self.grid_y = paddle.arange(g).tile([g, 1]).t().reshape([1, 1, g, g]).cast('float32')
        self.scaled_anchors = paddle.to_tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].reshape((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].reshape((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support

        self.img_dim = img_dim
        num_samples = x.shape[0]
        grid_size = x.shape[2]

        prediction = (
            x.reshape((num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size))
                .transpose((0, 1, 3, 4, 2))
        )

        # Get outputs
        x = F.sigmoid(prediction[:, :, :, :, 0])  # Center x
        y = F.sigmoid(prediction[:, :, :, :, 1])  # Center y
        w = prediction[:, :, :, :, 2]  # Width
        h = prediction[:, :, :, :, 3]  # Height
        pred_conf = F.sigmoid(prediction[:, :, :, :, 4])  # Conf
        pred_cls = F.sigmoid(prediction[:, :, :, :, 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        pred_boxes = paddle.zeros(prediction[:, :, :, :, :4].shape)
        pred_boxes[:, :, :, :, 0] = x + self.grid_x
        pred_boxes[:, :, :, :, 1] = y + self.grid_y
        pred_boxes[:, :, :, :, 2] = paddle.exp(w) * self.anchor_w
        pred_boxes[:, :, :, :, 3] = paddle.exp(h) * self.anchor_h

        output = paddle.concat(
            (
                pred_boxes.reshape((num_samples, -1, 4)) * self.stride,
                pred_conf.reshape((num_samples, -1, 1)),
                pred_cls.reshape((num_samples, -1, self.num_classes)),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask.bool()], tx[obj_mask.bool()])
            loss_y = self.mse_loss(y[obj_mask.bool()], ty[obj_mask.bool()])
            loss_w = self.mse_loss(w[obj_mask.bool()], tw[obj_mask.bool()])
            loss_h = self.mse_loss(h[obj_mask.bool()], th[obj_mask.bool()])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask.bool()], tconf[obj_mask.bool()])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask.bool()], tconf[noobj_mask.bool()])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask.bool()], tcls[obj_mask.bool()])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask.bool()].mean()
            conf_obj = pred_conf[obj_mask.bool()].mean()
            conf_noobj = pred_conf[noobj_mask.bool()].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = paddle.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = paddle.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = paddle.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss