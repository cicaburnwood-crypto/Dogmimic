import datetime
import os
import torch
import torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
x = torch.tensor([float(dist.get_rank()+1)], device=f'cuda:{rank}')
dist.all_reduce(x)
expected = dist.get_world_size()*(dist.get_world_size()+1)/2
assert x.item() == expected
print(f'Rank {rank}: NCCL all_reduce={x.item()} PASS', flush=True)
dist.barrier()
dist.destroy_process_group()
