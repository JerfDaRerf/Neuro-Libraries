import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import double, float64, nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

from cs639.loading import *

from torchvision import models, transforms, ops
from torchvision.models import feature_extraction
from torchvision.ops import sigmoid_focal_loss

# Detection backbone network: A tiny RegNet model coupled with a Feature Pyramid Network (FPN).
class DetectorBackboneWithFPN(nn.Module):

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        
        # Initialize with ImageNet pre-trained weights for faster convergence
        _cnn = models.regnet_x_400mf(pretrained=True)
        
        # Wrap the ConvNet with torchvision's feature extractor
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )
        
        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. 
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]
        
        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")
        
        
        # Initialize additional Conv layers for FPN.
        # Add three lateral 1x1 conv and three output 3x3 modules.
        self.fpn_params = nn.ModuleDict()

        self.fpn_params["1x1p5"] = nn.Conv2d(dummy_out_shapes[2][1][1], self.out_channels, kernel_size = (1, 1), stride = 1, padding = 0, bias = False) 
        self.fpn_params["1x1p4"] = nn.Conv2d(dummy_out_shapes[1][1][1], self.out_channels, kernel_size = (1, 1), stride = 1, padding = 0, bias = False) 
        self.fpn_params["1x1p3"] = nn.Conv2d(dummy_out_shapes[0][1][1], self.out_channels, kernel_size = (1, 1), stride = 1, padding = 0, bias = False) 
        self.fpn_params["3x3"] = nn.Conv2d(self.out_channels, self.out_channels, kernel_size = (3, 3), stride = 1, padding = 1, bias = False) 
        self.fpn_params["batchnorm"] = nn.BatchNorm2d(self.out_channels)
        self.fpn_params["relu"] = nn.ReLU()
      
    @property
    def fpn_strides(self):
       
        return {"p3": 8, "p4": 16, "p5": 32}
    
    def forward(self, images: torch.Tensor):
    
        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)
    
        fpn_feats = {"p3": None, "p4": None, "p5": None}
    
        fpn_feats["p5"] = self.fpn_params["1x1p5"](backbone_feats["c5"])
        fpn_feats["p4"] = self.fpn_params["1x1p4"](backbone_feats["c4"]) + F.interpolate(fpn_feats["p5"], scale_factor = 2)
        fpn_feats["p3"] = self.fpn_params["1x1p3"](backbone_feats["c3"]) + F.interpolate(fpn_feats["p4"], scale_factor = 2)

        fpn_feats["p5"] = self.fpn_params["3x3"](fpn_feats["p5"])
        fpn_feats["p4"] = self.fpn_params["3x3"](fpn_feats["p4"])
        fpn_feats["p3"] = self.fpn_params["3x3"](fpn_feats["p3"])
        fpn_feats["p3"] = self.fpn_params["batchnorm"](fpn_feats["p3"])
        fpn_feats["p3"] = self.fpn_params["relu"](fpn_feats["p3"])
        fpn_feats["p3"] = self.fpn_params["3x3"](fpn_feats["p3"])

        return fpn_feats


# Map every location in FPN feature map to a point on the image. 
# This point represents the center of the receptive field of this location.
def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",) -> Dict[str, torch.Tensor]:
  
    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        location_coords[level_name] = torch.zeros((feat_shape[3] * feat_shape[2], 2), device = device)
        a = 0
        for i in range(feat_shape[2]):
          for j in range(feat_shape[3]):
            coordinate = [level_stride * (i + 0.5), level_stride * (j + 0.5)]
            coordinate = torch.tensor(coordinate, device = device)
            location_coords[level_name][a]  = coordinate
            a += 1
            
    return location_coords

# Non-maximum suppression removes overlapping bounding boxes.
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    #############################################################################
    areas = torch.zeros_like(scores, dtype=torch.float64)
    
    interVals = torch.zeros((scores.size(dim=0), scores.size(dim=0)), dtype=float64)
    vert_ones = torch.ones((scores.size(dim=0), 1), dtype=float64)
    horiz_ones = torch.ones((1, scores.size(dim=0)), dtype=float64)

    # Vectorized area
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    areas_vec = areas.clone().reshape(1,-1).t()

    # Vertical
    inputX1 = boxes[:,0].reshape(1, -1).clone()
    X1Vals_v = torch.matmul(vert_ones, inputX1.type(torch.DoubleTensor))  
    inputX2 = boxes[:,2].reshape(1, -1).clone()
    X2Vals_v = torch.matmul(vert_ones, inputX2.type(torch.DoubleTensor))
    inputY1 = boxes[:,1].reshape(1, -1).clone()
    Y1Vals_vert = torch.matmul(vert_ones, inputY1.type(torch.DoubleTensor))
    inputY2 = boxes[:,3].reshape(1, -1).clone()
    Y2Vals_vert = torch.matmul(vert_ones, inputY2.type(torch.DoubleTensor))

    # Horizontal
    inputX1 = boxes[:,0].reshape(-1, 1).clone()
    X1Vals_horiz = torch.matmul(inputX1.type(torch.DoubleTensor), horiz_ones)  
    inputX2 = boxes[:,2].reshape(-1, 1).clone()
    X2Vals_horiz = torch.matmul(inputX2.type(torch.DoubleTensor), horiz_ones)
    inputY1 = boxes[:,1].reshape(-1, 1).clone()
    Y1Vals_h = torch.matmul(inputY1.type(torch.DoubleTensor), horiz_ones)
    inputY2 = boxes[:,3].reshape(-1, 1).clone()
    Y2Vals_h = torch.matmul(inputY2.type(torch.DoubleTensor), horiz_ones)

    # Vectorized IoU values
    x_intersect_vals = (torch.min(X2Vals_horiz, X2Vals_v) - torch.max(X1Vals_horiz, X1Vals_v))
    y_intersect_vals = (torch.min(Y2Vals_h, Y2Vals_vert) - torch.max(Y1Vals_h, Y1Vals_vert))
    x_intersect_vals[x_intersect_vals < 0] = 0
    y_intersect_vals[y_intersect_vals < 0] = 0
    
    interVals = x_intersect_vals * y_intersect_vals
    areas = areas_vec + areas_vec.t()
    areas = areas.to(boxes.device)
    interVals = interVals.to(boxes.device)
    iouVals = interVals / (areas - interVals)
    
    currMax, currMaxIndex = torch.max(scores, dim=0)
    currMin, currMinIndex = torch.min(scores, dim=0)

    # Get firstMax and firstMin for scores range
    firstMax = currMax
    firstMin = currMin
    scores_mask = (scores < firstMax)
    scores_copy = torch.clone(scores)
    
    # While there are valid scores, continue
    while scores_mask.int().sum() > 0 and currMax >= firstMin:
      iou_mask2 = (iouVals[currMaxIndex] <= iou_threshold)
      scores_mask = torch.logical_and(scores_mask, iou_mask2)
      scores_mask[currMaxIndex] = False

      # Append currMax to keep
      if(keep == None):
        keep = torch.zeros(1)
        keep[0] = currMaxIndex
      else:
        new = torch.tensor([currMaxIndex])        
        keep = torch.cat((keep, new)) #tuple
      scores_copy[~scores_mask]  = firstMin - 1
      currMax, currMaxIndex = torch.max(scores_copy, dim = 0)
    if keep != None:
      keep = keep.int()
      
    return keep

# Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,):
  
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    #keep = nms(boxes_for_nms, scores, iou_threshold).to('cuda')
    keep = nms(boxes_for_nms, scores, iou_threshold)
    if keep != None:
        keep = keep.to('cuda')
    else:
        keep = torch.empty((0,), dtype=torch.long, device="cuda")
    return keep

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]

# FCOS prediction network that accepts FPN feature maps from different levels and makes three predictions at every location: 
# bounding boxes, class ID and centerness.
# see Figure 2 (right side) in FCOS paper: https://arxiv.org/abs/1904.01355
class FCOSPredictionNetwork(nn.Module):
    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):

        super().__init__()

       
        # Create a stem of alternating 3x3 convolution layers and RELU activation modules.
      
        stem_cls = []
        stem_box = []

        for i in range(len(stem_channels)):
            # For the first layer, use the initial number of input channels.
            if i == 0:
                # Create a convolutional layer for classification tasks.
                # This layer uses a 3x3 kernel, stride of 1, padding of 1 to maintain spatial dimensions, and has a bias.
                CNNLayer_cls = nn.Conv2d(in_channels, stem_channels[i], (3,3), stride=1, padding=1, bias=True)
                # Create a similar convolutional layer for bounding box regression tasks.
                CNNLayer_box = nn.Conv2d(in_channels, stem_channels[i], (3,3), stride=1, padding=1, bias=True)
            else:
                # For subsequent layers, use the channel size of the previous layer as input channels.
                # Convolutional layer for classification, parameters similar to the first, adjusting in_channels based on the previous size.
                CNNLayer_cls = nn.Conv2d(stem_channels[i - 1], stem_channels[i], (3,3), stride=1, padding=1, bias=True)
                # Convolutional layer for bounding box regression, with parameters matching the classification layer.
                CNNLayer_box = nn.Conv2d(stem_channels[i - 1], stem_channels[i], (3,3), stride=1, padding=1, bias=True) 
        
        # Initialize the weights of the classification layer with a normal distribution for better convergence.
        nn.init.normal_(CNNLayer_cls.weight, mean=0, std=0.01)
        # Initialize the bias of the classification layer to 0.
        nn.init.constant_(CNNLayer_cls.bias, 0)

        stem_cls.append(CNNLayer_cls)
        stem_cls.append(nn.ReLU())
        stem_box.append(CNNLayer_box)
        stem_box.append(nn.ReLU())
        
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Three 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. 

        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        self.pred_cls = nn.Conv2d(stem_channels[i], num_classes, (3,3), stride = 1, padding = 1, bias = True)
        self.pred_box = nn.Conv2d(stem_channels[i], 4, (3,3), stride = 1, padding = 1, bias = True)
        self.pred_ctr = nn.Conv2d(stem_channels[i], 1, (3,3), stride = 1, padding = 1, bias = True)
        nn.init.normal_(self.pred_cls.weight, mean = 0, std = 0.01)
        nn.init.constant_(self.pred_cls.bias, 0)
        nn.init.normal_(self.pred_box.weight, mean = 0, std = 0.01)
        nn.init.constant_(self.pred_box.bias, 0)
        nn.init.normal_(self.pred_ctr.weight, mean = 0, std = 0.01)
        nn.init.constant_(self.pred_ctr.bias, 0)

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    # Accept FPN feature maps and predict the desired outputs at every location. 
    # Format them such that channels are placed at the last dimension, and (H, W) are flattened
    #  Returns:
    # List of dictionaries, each having keys {"p3", "p4", "p5"}:
    # 1. Classification logits: `(batch_size, H * W, num_classes)`.
    # 2. Box regression deltas: `(batch_size, H * W, 4)`
    # 3. Centerness logits:     `(batch_size, H * W, 1)`
    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        # Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. 
      
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for key in feats_per_fpn_level.keys():
          cls_out = self.pred_cls(self.stem_cls(feats_per_fpn_level[key]))
          B, C, H, W = cls_out.shape
          cls_out = cls_out.permute(0, 3, 2, 1).reshape(B, H*W, -1)
          class_logits[key] = cls_out

          stem_box_out = self.stem_box(feats_per_fpn_level[key])

          box_out = self.pred_box(stem_box_out)
          box_out = box_out.permute(0, 3, 2, 1).reshape(B, H*W, -1)
          boxreg_deltas[key] = box_out 

          ctr_out = self.pred_ctr(stem_box_out)
          ctr_out = ctr_out.permute(0, 3, 2, 1).reshape(B, H*W, -1)
          centerness_logits[key] = ctr_out

        return [class_logits, boxreg_deltas, centerness_logits]


# Match centers of the locations of FPN feature with a set of GT bounding boxes of the input image.
# Since our model makes predictions at every FPN feature map location, we must supervise it with an appropriate GT box
@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,) -> TensorDict:

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]
        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes

# Compute distances from feature locations to GT box edges.
def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int) -> torch.Tensor:
      
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    deltas = None

    deltas = torch.zeros((gt_boxes.shape[0], 4)) 
    clean_gt_boxes = gt_boxes[:, :4] 
    deltas[:, 0] = (locations[:, 0] - clean_gt_boxes[:, 0]) / stride
    deltas[:, 1] = (locations[:, 1] - clean_gt_boxes[:, 1]) / stride    
    deltas[:, 2] = (clean_gt_boxes[:, 2] - locations[:, 0]) / stride
    deltas[:, 3] = (clean_gt_boxes[:, 3] - locations[:, 1]) / stride
    deltas[(torch.all(clean_gt_boxes == -1, dim = 1))] = -1
   
    return deltas

# Given edge deltas (left, top, right, bottom) and feature locations of FPN, get the resulting bounding box co-ordinates by applying deltas on locations. 
# This method is used for inference in FCOS: deltas are outputs from model, and applying them to anchors will give us final box predictions.
def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int) -> torch.Tensor:
    output_boxes = None
    
    output_boxes = torch.zeros_like(deltas) 
    clipped_deltas = torch.clone(deltas)
    clipped_deltas[deltas.lt(0)] = 0

    output_boxes[:, 0] = locations[:, 0] - (clipped_deltas[:, 0] * stride)
    output_boxes[:, 1] = locations[:, 1] - (clipped_deltas[:, 1] * stride)   
    output_boxes[:, 2] = locations[:, 0] + (clipped_deltas[:, 2] * stride)
    output_boxes[:, 3] = locations[:, 1] + (clipped_deltas[:, 3] * stride)

    return output_boxes

# Given LTRB deltas of GT boxes, compute GT targets for supervising the centerness regression predictor.
def fcos_make_centerness_targets(deltas: torch.Tensor):

    centerness = None
    centerness = torch.sqrt(torch.div(torch.minimum(deltas[:, 0], deltas[:, 2]) * torch.minimum(deltas[:, 1], deltas[:, 3]) , (torch.maximum(deltas[:, 0], deltas[:, 2]) * torch.maximum(deltas[:, 1], deltas[:, 3]))))
    centerness[(torch.all(deltas == -1, dim = 1))] = -1
  
    return centerness

# This class puts together everything implemented so far
# It contains a backbone with FPN, and prediction layers (head). It computes loss during training and predicts boxes during inference.
class FCOS(nn.Module):
    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]):
        super().__init__()
        self.num_classes = num_classes
          
        self.backbone = None
        self.pred_net = None

        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)
       
        # Averaging factor for training loss; EMA of foreground locations.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,):


        # Process the image through backbone, FPN, and prediction head 
        # to obtain model predictions at every FPN location.                 
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class 
        # logits, deltas, and centerness.                                    
        
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
       
        fpn_feats = self.backbone.forward(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net.forward(fpn_feats)
          
        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        fpn_shapes = {}
        for i in fpn_feats.keys():
          fpn_shapes[i] = fpn_feats[i].shape
        locations_per_fpn_level = get_fpn_location_coords(fpn_shapes, self.backbone.fpn_strides, device = "cuda")

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        # Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`.
          
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        
        for i in range(gt_boxes.shape[0]):
          matched_gt_boxes.append(fcos_match_locations_to_gt(locations_per_fpn_level, self.backbone.fpn_strides, gt_boxes[i]))

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        
        for i in range(gt_boxes.shape[0]):
          newDict = {}
          for key in locations_per_fpn_level.keys():
            newDict[key] = (fcos_get_deltas_from_locations(locations_per_fpn_level[key], matched_gt_boxes[i][key], self.backbone.fpn_strides[key]))
          matched_gt_deltas.append(newDict)

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image
          
        # Calculate losses per location for classification, box reg and centerness.

        loss_cls, loss_box, loss_ctr = None, None, None
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits

        #16, 1029, 5
        flattened_matched_gt_boxes = torch.flatten(matched_gt_boxes[:, :, 4])
        flattened_matched_gt_boxes[flattened_matched_gt_boxes == -1] = self.num_classes
        flattened_matched_gt_boxes = flattened_matched_gt_boxes.long()
        classes = F.one_hot(flattened_matched_gt_boxes, num_classes = self.num_classes + 1)
        classes = classes.reshape((matched_gt_boxes.shape[0], matched_gt_boxes.shape[1], self.num_classes + 1))
        classes = classes[:, :, 0:self.num_classes]
        loss_cls = sigmoid_focal_loss(inputs = pred_cls_logits, targets = classes.float())

        collapsed_pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4).to("cuda")
        collapsed_matched_gt_deltas = matched_gt_deltas.view(-1, 4).to("cuda")
        loss_box = 0.25 * F.l1_loss(
          collapsed_pred_boxreg_deltas, collapsed_matched_gt_deltas, reduction="none"
        )
        loss_box[collapsed_matched_gt_deltas < 0] *= 0.0

        collapsed_pred_ctr_logits = pred_ctr_logits.view(-1)

        gt_centerness = torch.zeros_like(pred_ctr_logits)
        
        for B in range(matched_gt_deltas.shape[0]):
          gt_centerness[B, :, 0] = fcos_make_centerness_targets(matched_gt_deltas[B, :, :])
        collaped_gt_centerness = gt_centerness.view(-1)
        loss_ctr = F.binary_cross_entropy_with_logits(
          collapsed_pred_ctr_logits, collaped_gt_centerness, reduction="none"
        )
        loss_ctr[collaped_gt_centerness < 0] *= 0.0

        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    # Run inference on a single input image (batch size = 1). Other input arguments are same as those computed in `forward` method.
    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,):

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # This helps in getting rid of excessive amount of boxes far away from object centers.
            #
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond the height and
            #      and width of input image.
            ##################################################################

            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1: Get the most confidently predicted class and its score for every box
            # Replace "PASS" statement with your code
            # Get the maximum on C and put it as the value for that row
            
            level_pred_scores, level_pred_classes = torch.max(level_pred_scores, 1)
            # Step 2: Only retain predictions with confidence score > threshold
            # Clone the prediction scores
            
            pred_scores_mask = (level_pred_scores > test_score_thresh)
            level_pred_scores = level_pred_scores[pred_scores_mask]
            level_pred_classes = level_pred_classes[pred_scores_mask]
            pred_scores_mask = pred_scores_mask.int()

            # Get indices of all instances of 1s
            lps_zeros = (torch.nonzero(pred_scores_mask, as_tuple=False))
            
            # Step 3: Use level_deltas and locations for something

            # Initialize width, height, and stride for current layer
            width = images.size(dim=2)
            height = images.size(dim=3)
            stride = self.backbone.fpn_strides[level_name]

            # Chooses rows from level_deltas based on indices stored in lps_zeros
            level_deltas = level_deltas[lps_zeros[:,0]]
            level_locations = level_locations[lps_zeros[:,0]]

            # Applies the deltas to the pixels designated by lps_zeros_coords
            if lps_zeros.size() != 0:
              level_boxes = fcos_apply_deltas_to_locations(level_deltas, level_locations, stride)

            # Deals with edge case of no objects
            if(level_boxes == None):
              level_boxes = []

            # Step 4: Use `images` to get (height, width) for clipping.

            # If the there are boxes, clips boxes based on width
            if(level_boxes.size(dim=0) > 0):
              level_boxes[:, 0:2:2] [level_boxes[:, 0:2:2] > width] = width
              level_boxes[:, 1:3:2] [level_boxes[:, 1:3:2] > height] = height
              level_boxes[level_boxes < 0] = 0
            
            level_pred_boxes = level_boxes
           
            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        # Combine predictions from all levels and perform NMS
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
