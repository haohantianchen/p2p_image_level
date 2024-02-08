import torch

device = torch.device("cuda:2")
g_cpu = torch.Generator(device).manual_seed(6666)
shape = [256, 4096, 40]
dtype = torch.float16
q = torch.randn(shape, dtype=dtype, device=device, generator=g_cpu)
k_t = q.transpose(-1, -2)
tmp = torch.bmm(q, k_t)
del q
del k_t
torch.cuda.empty_cache()
a = torch.cuda.memory_reserved(0)
print(a)
print(tmp.shape)
# tmp = tmp.float()
tmp.softmax(-1)
torch.cuda.empty_cache()
a = torch.cuda.memory_reserved(0)
print(a)
print(tmp.shape)
tmp = tmp.cuda(0)
a = torch.cuda.memory_reserved(0)
print(a)
print("ok")