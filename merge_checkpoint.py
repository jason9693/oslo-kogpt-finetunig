import torch.distributed as dist
import wandb
from datasets import load_dataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.utils import allocate_params
from oslo.torch.distributed import ParallelContext, ParallelMode


# 병렬 사이즈 설정
tp_size = 4
tp_depth = 1

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    tensor_parallel_depth=1,
)

model_reparallel = TensorParallel(
    AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("./test")), parallel_context
)
allocate_params(model_reparallel, parallel_context)
model_reparallel.from_parallelized("test/")

model_reparallel.save_parallelized("test_merge/", merge_checkpoints=True)