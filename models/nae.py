from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from losses.oim import get_oim_func
from models.backbone.resnet import build_resnet
from models.custom_modules import CustomBatchNorm1d, convert_bn_to_cabn, register_targets_for_cabn

class NAE(nn.Module):
    def __init__(self, cfg):
        super(NAE, self).__init__()

        backbone, box_head = build_resnet(name="resnet50")
        if cfg.MODEL.ROI_HEAD.CABN.BACKBONE:
            box_head = convert_bn_to_cabn(box_head)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,

            # hyper params
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        
        box_predictor = CoordRegressor(2048, num_classes=2, RCNN_bbox_bn=cfg.MODEL.ROI_HEAD.BN_NECK)
        
        roi_heads = NormAwareRoiHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            oim_type=cfg.MODEL.LOSS.TYPE,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            cabn_last=cfg.MODEL.ROI_HEAD.CABN.LAST,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        # self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        # self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features)
            embeddings, _ = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(features, proposals, images.image_sizes, targets, query_img_as_gallery)
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        # losses["loss_box_reg"] *= self.lw_box_reg
        # losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses


class NormAwareRoiHeads(RoIHeads):

    def __init__(
        self, 
        num_pids, 
        num_cq_size, 
        oim_momentum, 
        oim_scalar, 
        oim_type,
        faster_rcnn_predictor, 
        reid_head,
        cabn_last,
        *args, 
        **kwargs
    ):
        super(NormAwareRoiHeads, self).__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding(
            featmap_names=['feat_res4', 'feat_res5'],
            in_channels=[1024, 2048],
            dim=256,
            cabn_last=cabn_last,
        )
        self.reid_loss = get_oim_func(256, num_pids, num_cq_size, oim_momentum, oim_scalar, str(oim_type))
        self.oim_type = str(oim_type)
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, \
                    'target labels must of int64 type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        if self.training: register_targets_for_cabn(self.reid_head, torch.cat(labels).long())
        rcnn_features = self.reid_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features['feat_res5'])
        if self.training: register_targets_for_cabn(self.embedding_head, torch.cat(labels).long())
        embeddings_, class_logits = self.embedding_head(rcnn_features)

        result, losses = [], {}
        if self.training:
            if self.oim_type == '2':
                max_iou_list = []
                for batch_index in range(len(proposals)):
                    box_p = proposals[batch_index]
                    box_t = targets[batch_index]['boxes']
                    ious = box_ops.box_iou(box_p, box_t)
                    ious_max = torch.max(ious, dim=1)[0]
                    max_iou_list.append(ious_max)
                obj_scores = torch.cat(max_iou_list, dim=0)
                
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_proposal_cls, loss_proposal_reg = \
                norm_aware_rcnn_loss(class_logits, box_regression,
                                     det_labels, regression_targets)

            if self.oim_type == '2':
                loss_box_reid = self.reid_loss(embeddings_, labels, obj_scores)
            else:
                loss_box_reid = self.reid_loss(embeddings_, labels)

            losses = dict(loss_proposal_cls=loss_proposal_cls,
                          loss_proposal_reg=loss_proposal_reg,
                          loss_box_reid=loss_box_reid)
        else:
            gt_det = None
            if query_img_as_gallery:
                gt_det = {"boxes": targets[0]["boxes"], "embeddings":embeddings_}
            
            boxes, scores, embeddings, labels = \
                self.postprocess_boxes(class_logits, box_regression, embeddings_,
                                            proposals, image_shapes, gt_det=gt_det)
            
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes
    
    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=False,
    ):
        """
        class_logits: 2D tensor(n_roi_per_img*bs C)
        box_regression: 2D tensor(n_roi_per_img*bs C*4)
        proposals: list[tensor(n_roi_per_img 4)]
        image_shapes: list[tuple[H, W]]
        box_features: 2D tensor(n_roi_per_img*bs dim_feat)]
        """

        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # tensor(n_roi_per_img*bs C 4)

        pred_scores = torch.sigmoid(class_logits)

        ### added
        if embeddings is not None:
            # box_features = box_features * pred_scores.view(-1, 1)  # CWS
            embeddings = embeddings.split(boxes_per_image, 0)
        ###

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img C 4)], length=bs
        pred_scores = pred_scores.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img C)], length=bs

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        n_iter = 0
        # go through batch_size
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            #
            if embeddings is not None:
                embeddings = embeddings[n_iter]

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)  # tensor(n_roi_per_img C 4)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]  # tensor(n_roi_per_img C-1 4)
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)  # 2D tensor(n_roi_per_img*(C-1) 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            if embeddings is not None:
                embeddings = embeddings[inds]

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if embeddings is not None:
                embeddings = embeddings[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            if embeddings is not None:
                all_embeddings.append(embeddings)
            n_iter += 1

        return all_boxes, all_scores, all_embeddings, all_labels 


class NormAwareEmbedding(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256, cabn_last=False):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            if cabn_last:
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim),
                    CustomBatchNorm1d(indv_dim)
                )
            else: 
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim),
                    nn.BatchNorm1d(indv_dim)
                )
            
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        if cabn_last: 
            self.rescaler = CustomBatchNorm1d(1, affine=True)
        else:
            self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        outputs = []
        for k in self.featmap_names:
            v = featmaps[k]
            v = self._flatten_fc_input(v)
            outputs.append(
                self.projectors[k](v)
            )
        embeddings = torch.cat(outputs, dim=1)
        norms = embeddings.norm(2, 1, keepdim=True)
        embeddings = embeddings / \
            norms.expand_as(embeddings).clamp(min=1e-12)
        norms = self.rescaler(norms).squeeze()
        return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class CoordRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """

    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(CoordRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def norm_aware_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(
        class_logits, labels.float())

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss