import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.dense_heads import DeformableDETRHead

@MODELS.register_module()
class DQDETRHead(DeformableDETRHead):
    
    def __init__(self, 
                 ccm_params=[10, 100, 500], 
                 *args, **kwargs):
        super(DQDETRHead, self).__init__(*args, **kwargs)
        self.ccm_params = ccm_params
        self.loss_ccm = nn.CrossEntropyLoss()

    def loss(self, hidden_states, references, enc_outputs_class,
             enc_outputs_coord, batch_data_samples, ccm_logits=None):
        """
        计算训练 Loss
        """
        init_reference, inter_references = references
        outputs_classes = []
        outputs_coords = []
        
        # 遍历每一层 Decoder 进行预测
        for lvl in range(hidden_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            # 使用 logit 代替 inverse_sigmoid，并加 clamp 保护
            reference = torch.clamp(reference, min=1e-5, max=1-1e-5)
            reference = torch.logit(reference)
            
            outputs_class = self.cls_branches[lvl](hidden_states[lvl])
            tmp = self.reg_branches[lvl](hidden_states[lvl])
            
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        # 2. 计算常规 DETR Loss
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            batch_img_metas.append(data_sample.metainfo)

        loss_dict = self.loss_by_feat(
            all_cls_scores,
            all_bbox_preds,
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas
        )
        
        # 3. 计算 CCM Loss
        if ccm_logits is not None:
            gt_counts = [len(sample.gt_instances) for sample in batch_data_samples]
            target_labels = []
            
            for count in gt_counts:
                if count <= self.ccm_params[0]: label = 0
                elif count <= self.ccm_params[1]: label = 1
                elif count <= self.ccm_params[2]: label = 2
                else: label = 3
                target_labels.append(label)
            
            target_labels = torch.tensor(target_labels, device=ccm_logits.device)
            loss_ccm_val = self.loss_ccm(ccm_logits, target_labels)
            loss_dict['loss_ccm'] = loss_ccm_val

        return loss_dict

    def predict(self,
                hidden_states,
                references,
                batch_data_samples,
                rescale=True,
                **kwargs):
        """
        推理逻辑，修复了 max_per_img 越界问题
        """
        # 1. 提取最后一层的输出
        last_layer_hidden_state = hidden_states[-1]
        
        init_reference, inter_references = references
        last_layer_reference = inter_references[-1]
        
        # 2. 执行前向计算 (分类 + 回归)
        reference = torch.clamp(last_layer_reference, min=1e-5, max=1-1e-5)
        reference = torch.logit(reference)
        
        outputs_class = self.cls_branches[-1](last_layer_hidden_state)
        tmp = self.reg_branches[-1](last_layer_hidden_state)
        
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        
        outputs_coord = tmp.sigmoid()

        # 3. 动态调整 max_per_img
        # 原因：DQ-DETR 的 Query 数量是动态的 (如 300)，如果 max_per_img 设置为 1500，
        # 而 300 * num_classes < 1500，topk 就会报错。
        
        # 计算当前总共有多少个预测分数
        current_num_queries = outputs_class.shape[1]
        num_classes = outputs_class.shape[2]
        total_predictions = current_num_queries * num_classes
        
        # 获取原始配置的 max_per_img
        # 注意：test_cfg 可能在 self.test_cfg 中
        original_max_per_img = self.test_cfg.get('max_per_img', 100)
        
        # 临时调整 max_per_img，确保不超过总预测数
        safe_max_per_img = min(original_max_per_img, total_predictions)
        
        # 无论 test_cfg 是 dict 还是 ConfigDict，我们都尝试临时修改它
        # 如果是 ConfigDict，可能需要解锁，但在 runner 运行期通常没问题
        self.test_cfg['max_per_img'] = safe_max_per_img

        try:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]

            # 注意：predict_by_feat 期望列表输入 [layer1, layer2...]
            # 我们这里只传最后一层，所以要在外面包一层列表
            results_list = self.predict_by_feat(
                [outputs_class], # <--- 修正：包装成 list
                [outputs_coord], # <--- 修正：包装成 list
                batch_img_metas=batch_img_metas,
                rescale=rescale
            )
        finally:
            # 恢复原始配置，以免影响后续可能的大 Query 图像
            self.test_cfg['max_per_img'] = original_max_per_img

        return results_list