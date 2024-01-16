import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

a = [torch.randn(642, 96) for i in range(8)]
b = [torch.randn(56, 96) for i in range(8)]

lm_am_sim_cp = [F.softmax(torch.matmul(a[i], b[i].T)/3, dim=1) for i in range(8)]
lm_am_sim = None

'''
for i in range(8):
    print(lm_am_sim_cp[i].size())
    if i == 0:
        lm_am_sim = lm_am_sim_cp[i]
    else:
        lm_am_sim = torch.cat([lm_am_sim, lm_am_sim_cp[i]], dim=0)

print(lm_am_sim.size())
exit()
'''

for i in range(8):
    plt.matshow(lm_am_sim_cp[i].numpy())
    plt.colorbar()
    if not os.path.exists(f'/home/work/workspace/fairseq/scripts/whale/cross_attn'):
        try: os.makedirs(f'/home/work/workspace/fairseq/scripts/whale/cross_attn')
        except: pass
    plt.savefig(f'/home/work/workspace/fairseq/scripts/whale/cross_attn/cross_attn_{i}.png')
