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

import os
import time

import torch
import transformers

from cgm import runtime
from cgm.arguments import ModelArguments, DataArguments, TrainingArguments
from cgm.data import make_supervised_data_module
from cgm.fisher import fisher_matrix_diag
from cgm.model_io import safe_save_model_for_hf_trainer
from cgm.pruning import prune_cgm, prune_cgm_plus
from llava.mm_utils import get_model_name_from_path
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.train.llava_trainer import LLaVATrainer


def main():
    runtime.set_random_seed()
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runtime.local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model_base=None
    # 加载微调后的模型
    model_path = os.path.expanduser(model_args.model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,model_base,model_name)
    # 加载微调前的模型
    pre_model_path = os.path.expanduser(model_args.pre_model_name_or_path)
    pre_model_name = get_model_name_from_path(pre_model_path)
    pre_tokenizer, pre_model, pre_image_processor, pre_context_len = load_pretrained_model(pre_model_path, model_base,pre_model_name)

    print(model)

    
    
    for i in range(32):
        pre_model.model.layers[i].requires_grad_(False)

    # model.model.requires_grad_(False)

    for i in range(32):
        model.model.layers[i].requires_grad_(False)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    data_args.image_processor = model.model.vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    fisher = fisher_matrix_diag(trainer, model, nsamples=16)
    
    
    #fisher=fisher_matrix_diag(trainer,model, nsamples=16)
    if model_args.fisher_save_path:
        save_dir = os.path.dirname(model_args.fisher_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(fisher, model_args.fisher_save_path)
        print(f"Fisher matrix saved to {model_args.fisher_save_path}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    print(f"Fisher matrix total took {end - start:.2f} seconds")
    return


if __name__ == "__main__":
    main()
