# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch


def prune_cgm_plus(model,pre_model,fisher,pre_fisher,sparsity_ratio,alpha=1.0):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), pre_model.named_parameters()):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        if name1 in fisher:
            assert name1 in pre_fisher
            # if 'mm_projector' in name1:
            #     print('skip mm_projector')
            #     continue
            # if 'lm_head' in name1:
            #     print('skip lm_head')
            #     continue
            #print(name1)
            
            w = param1.data
            w_pre = param2.data
            device = w.device

            delta_w=w-w_pre.to(device)
            delta_w2=(delta_w)**2
            f = fisher[name1].to(device)
            f_pre = pre_fisher[name1].to(device)
            #original 
            W_metric=(alpha*f_pre-f)*delta_w2 
            #================消融===================
            #W_metric=delta_w2
            #W_metric=-f * delta_w2 
            #W_metric=alpha*f_pre*delta_w2
            W_mask = (torch.zeros_like(W_metric) == 1).to(device) #all false
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True) # 1 indicate use ft , 0 indeicate use pretrain
            W_mask = torch.logical_not(W_mask)
            w[W_mask] = w_pre.to(device)[W_mask]


def prune_cgm(model,pre_model,fisher,pre_fisher,alpha=1.0):
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), pre_model.named_parameters()):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        if name1 in fisher:
            assert name1 in pre_fisher
            if 'mm_projector' in name1:
                print('skip mm_projector')
                continue
            #print(name1)
            w = param1.data
            w_pre = param2.data
            device = w.device

            f = fisher[name1].to(device)
            f_pre = pre_fisher[name1].to(device)
            denominator = f + alpha * f_pre
            mixer = torch.where(denominator != 0, f / denominator, torch.zeros_like(f))
            #print(torch.mean(mixer).item())
            param1.data.copy_(w_pre.to(device) + mixer * (w - w_pre.to(device)))

