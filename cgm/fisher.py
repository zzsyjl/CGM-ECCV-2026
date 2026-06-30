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

import time
from collections import OrderedDict

import torch
from tqdm import tqdm


def find_layers(module, layers=[torch.nn.Conv2d, torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
# def fisher_matrix_diag(trainer,model, nsamples=128):
#     dataloader = trainer.get_train_dataloader()
#     model.train()
#     from collections import OrderedDict
#     # Init 
#     fisher = OrderedDict()
#     layers = model.model.layers
#     for i in range(20,len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)  # Get all linear/conv layers in this transformer layer
#         for name in subset:
#             param = subset[name].weight
#             param.requires_grad = True
#             full_name = f"model.layers.{i}.{name}.weight"
#             fisher[full_name] = torch.zeros_like(param.data)

# #============ merger======================
#     mm_projector=model.model.mm_projector
#     merger_subset = find_layers(mm_projector)
#     for name in merger_subset:
#         param = merger_subset[name].weight
#         param.requires_grad = True
#         full_name = f"model.mm_projector.{name}.weight"
#         fisher[full_name] = torch.zeros_like(param.data)
# #=============header=====================
#     header=model.lm_head.weight
#     header.requires_grad = True
#     fisher['lm_head.weight']=torch.zeros_like(header.data)

#     # Accumulate squared gradients
#     nsamples_seen = 0
    
#     for step, inputs in enumerate(tqdm(dataloader, desc="Computing Fisher", total=nsamples)):
#         if nsamples_seen >= nsamples:
#             break
#         inputs = trainer._prepare_inputs(inputs)

#         # Forward 
#         outputs = model(**inputs)
#         loss = outputs.loss  # token level loss
#         # Backward
#         model.zero_grad()
#         loss.backward()

#         # Accumulate the squared gradient for relevant parameters
#         # for i in range(len(layers)):
#         #     layer = layers[i]
#         #     subset = find_layers(layer)
#         #     for name in subset:
#         #         param = subset[name].weight
#         #         full_name = f"model.layers.{i}.{name}.weight"
#         #         assert param.grad is not None #should be true
#         #         # Accumulate squared gradient 
#         #         fisher[full_name] += (param.grad.data ** 2)
#         for n,p in model.named_parameters():
#             if n in fisher:
#                 if nsamples_seen == 0:
#                     print(n)
#                 assert p.grad is not None
#                 fisher[n]+=(p.grad.data ** 2)

#         del loss, outputs, inputs
#         # if hasattr(torch, 'cuda') and torch.cuda.is_available():
#         #     torch.cuda.empty_cache()  # 清理未使用的缓存

#         nsamples_seen += 1

#     model.zero_grad(set_to_none=True)  # ✅ 彻底释放 grad 内存
#     for param in model.parameters():
#         if param.grad is not None:
#             param.grad = None  # 手动置空（更彻底）
#     # Normalized
#     for name in fisher:
#         fisher[name] = fisher[name] / nsamples_seen

#     # for name in fisher:
#     #     tensor = fisher[name]
#     #     mean = tensor.mean()
#     #     std = tensor.std()
#     #     std = std if std > 0 else 1.0
#     #     fisher[name] = (tensor - mean) / std

#     # for name in fisher:
#     #     mean = fisher[name].mean()
#     #     if mean != 0:
#     #         fisher[name] = fisher[name] / mean

#     for name in fisher:
#         fisher[name] = fisher[name].cpu()

#     return fisher

def fisher_matrix_diag(trainer, model, nsamples=128):
    dataloader = trainer.get_train_dataloader()
    model.train()
    from collections import OrderedDict
    import torch

    fisher = OrderedDict()
    tracked_params = OrderedDict()  # 新增：{param_name: param}

    layers = model.model.layers
    for i in range(20, len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            param = subset[name].weight
            param.requires_grad_(True)  # 更简洁的写法
            full_name = f"model.layers.{i}.{name}.weight"
            fisher[full_name] = torch.zeros_like(param.data)
            tracked_params[full_name] = param

    # mm_projector
    mm_projector = model.model.mm_projector
    merger_subset = find_layers(mm_projector)
    for name in merger_subset:
        param = merger_subset[name].weight
        param.requires_grad_(True)
        full_name = f"model.mm_projector.{name}.weight"
        fisher[full_name] = torch.zeros_like(param.data)
        tracked_params[full_name] = param

    # lm_head
    header = model.lm_head.weight
    header.requires_grad_(True)
    fisher['lm_head.weight'] = torch.zeros_like(header.data)
    tracked_params['lm_head.weight'] = header

    # Accumulate squared gradients
    nsamples_seen = 0

    from tqdm import tqdm
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    
    for step, inputs in enumerate(tqdm(dataloader, desc="Computing Fisher", total=nsamples)):
        if nsamples_seen >= nsamples:
            break
        inputs = trainer._prepare_inputs(inputs)

        outputs = model(**inputs)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()


        # for name, param in tracked_params.items():
        #     assert param.grad is not None, f"Gradient is None for {name}"
        #     fisher[name] += (param.grad.data ** 2)

        #del loss, outputs, inputs
        nsamples_seen += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    print(f"Fisher matrix computation took {end - start:.2f} seconds")
    # Clean up gradients thoroughly
    model.zero_grad(set_to_none=True)
    for param in tracked_params.values():
        param.grad = None

    # Normalize
    for name in fisher:
        fisher[name] /= nsamples_seen

    # Move to CPU
    for name in fisher:
        fisher[name] = fisher[name].cpu()

    return fisher
