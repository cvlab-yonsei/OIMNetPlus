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

from losses.oim import OIMLoss, LOIMLoss
from models.backbone.resnet import build_resnet
from models.custom_modules import PrototypeNorm1d, register_targets_for_pn, convert_bn_to_pn

class BaseNet(nn.Module):
    def __init__(self, cfg):
        super(BaseNet, self).__init__()

        backbone, box_head = build_resnet(name="resnet50")

        if cfg.MODEL.ROI_HEAD.AUGMENT:
            box_head = convert_bn_to_pn(box_head)

        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32),), aspect_ratios=((0.5, 1.0, 2.0),)
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
        box_predictor = BBoxPredictor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        
        roi_heads = BaseRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            oim_type=cfg.MODEL.LOSS.TYPE,
            oim_eps=cfg.MODEL.LOSS.OIM_EPS, 
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            norm_type=cfg.MODEL.ROI_HEAD.NORM_TYPE,
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
            embeddings = self.roi_heads.embedding_head(box_features)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery
            )
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
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses


class BaseRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        oim_type,
        oim_eps, 
        faster_rcnn_predictor,
        reid_head,
        norm_type,
        *args,
        **kwargs
    ):
        super(BaseRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = ReIDEmbedding(
            featmap_names=['feat_res5'],
            in_channels=[2048],
            dim=256,
            norm_type=norm_type,
        )
        if oim_type == 'OIM':
            self.reid_loss = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        if oim_type == 'LOIM': 
            self.reid_loss = LOIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar, oim_eps)
        self.oim_type = oim_type
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
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)
        if self.training: register_targets_for_pn(self.reid_head, torch.cat(labels).long())
        rcnn_features = self.reid_head(roi_pooled_features)
        class_logits, box_regression = self.box_predictor(rcnn_features['feat_res5'])
        if self.training: register_targets_for_pn(self.embedding_head, torch.cat(labels).long())
        embeddings_ = self.embedding_head(rcnn_features)

        result, losses = [], {}
        if self.training:
            if self.oim_type == 'LOIM':
                max_iou_list = []
                # step1. compute IoU between all proposals and all ground-truth boxes
                # step2. select the maximum ground-truth box for each proposal
                # step3. (within the LOIM loss) filter out background proposals
                for batch_index in range(len(proposals)):
                    box_p = proposals[batch_index]
                    box_t = targets[batch_index]['boxes']
                    ious = box_ops.box_iou(box_p, box_t) 
                    ious_max = torch.max(ious, dim=1)[0] 
                    max_iou_list.append(ious_max)
                ious = torch.cat(max_iou_list, dim=0)

            det_labels = [y.clamp(0, 1) for y in labels]
            loss_proposal_cls, loss_proposal_reg = \
                rcnn_loss(class_logits, box_regression,
                                     det_labels, regression_targets)

            if self.oim_type == 'LOIM':
                ious = torch.clamp(ious, min=0.7) # min = cfg.MODEL.RPN.POS_THRESH_TRAIN (just to be safe)
                loss_box_reid = self.reid_loss.forward(embeddings_, labels, ious)
            else: 
                loss_box_reid = self.reid_loss.forward(embeddings_, labels)

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

        pred_scores = F.softmax(class_logits, -1)

        if embeddings is not None:
            embeddings = embeddings.split(boxes_per_image, 0)

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
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]  # tensor(n_roi_per_img C-1 4)
            scores = scores[:, 1:]  # tensor(n_roi_per_img C-1)
            labels = labels[:, 1:]  # tensor(n_roi_per_img C-1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)  # 2D tensor(n_roi_per_img*(C-1) 4)
            scores = scores.flatten()
            labels = labels.flatten()

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

class ReIDEmbedding(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256, norm_type='none'):
        super(ReIDEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            if norm_type == 'none':
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim, bias=False), 
                )

                init.normal_(proj[0].weight, std=0.01)

            if norm_type == 'protonorm':
                proj = nn.Sequential(
                    nn.Linear(in_channel, indv_dim, bias=False), 
                    PrototypeNorm1d(indv_dim)
                )

                init.normal_(proj[0].weight, std=0.01)

            self.projectors[ftname] = proj

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
        '''
        outputs = []
        for k in self.featmap_names:
            v = featmaps[k]
            v = self._flatten_fc_input(v)
            outputs.append(
                self.projectors[k](v)
            )
        return F.normalize(torch.cat(outputs, dim=1))

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

class BBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, bn_neck=True):
        super(BBoxPredictor, self).__init__()

        # classification
        self.bbox_cls = nn.Linear(in_channels, num_classes)
        init.normal_(self.bbox_cls.weight, std=0.01)
        init.constant_(self.bbox_cls.bias, 0)

        # regression
        self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
        init.normal_(self.bbox_pred.weight, std=0.01)
        init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)

        bbox_scores = self.bbox_cls(x)
        bbox_deltas = self.bbox_pred(x)
        return bbox_scores, bbox_deltas

def rcnn_loss(class_logits, box_regression, labels, regression_targets):

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

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
