import numpy as np
import torch

def load_target_jt(device, file, offset):
    one_target_jt = np.load(f"/home/fleaven/robot/humanplus/HST/legged_gym/data/{file}").astype(np.float32)
    one_target_jt = torch.from_numpy(one_target_jt).to(device)
    target_jt = one_target_jt.unsqueeze(0)
    target_jt += offset

    size = torch.tensor([one_target_jt.shape[0]]).to(device)
    return target_jt, size



def load_target_jt_new(device, file, offset, freq):
    idx = [12,13,14,15,16,17,18,22,23,24,25]
    idx = np.concatenate((list(range(0,36)), list(range(36+7*3,36+7*3+6)),list(range(36+20*3,36+20*3+6))))
    fr = file[-12:-9]
    fr = int(fr[1:]) if fr[0]=='_' else int(fr)
    assert(freq<= fr)
    sampling_rate = fr // freq #向快对齐
    one_target_jt = np.load(f"/home/fleaven/dataset/test_amass/{file}").astype(np.float32)
    one_target_jt = one_target_jt[::sampling_rate, idx]
    target_jt = torch.from_numpy(one_target_jt).to(device).unsqueeze(0)
    #target_jt += offset
    return target_jt

def load_target_jt_upb(device, file, offset, freq):
    idx = [7+12,7+13,7+14,7+15,7+16,7+17,7+18,7+22,7+23,7+24,7+25]
    idx = list(range(0,7))+idx

    with open(file) as f:
        files = f.readlines()
    
    target_jt_tensor = torch.zeros((2048,20*freq,18), dtype=torch.float32, device=device)
    target_jt_lenth = torch.zeros(2048, dtype=torch.float32, device=device)
    i = 0
    for f in files[:2048]:
        fr = f[-13:-10]
        fr = int(fr[1:]) if fr[0]=='_' else int(fr)
        assert(freq<= fr)
        sampling_rate = fr // freq #向快对齐

        one_target_jt = np.load(f[:-1]).astype(np.float32)
        one_target_jt = one_target_jt[:freq*20*sampling_rate:sampling_rate, idx]
        target_jt_tensor[i,:one_target_jt.shape[0],:] = torch.tensor(one_target_jt, dtype=torch.float32, device=device)
        target_jt_lenth[i] = one_target_jt.shape[0]
        i += 1
    return target_jt_tensor, target_jt_lenth

def load_target_jt_pos(device, file, offset, freq):
    with open(file) as f:
        files = f.readlines()
    
    sz = len(files)
    target_jt_tensor = torch.zeros((sz,20*freq,36), dtype=torch.float32, device=device)
    target_jt_lenth = torch.zeros(sz, dtype=torch.float32, device=device)
    i = 0
    for f in files:
        # fr = f[-13:-10]
        # fr = int(fr[1:]) if fr[0]=='_' else int(fr)
        import re
        match = re.search(r'_(\d+)_', f)
        if match:
            fr = int(match.group(1))
        else:
            fr = 60
        assert(freq<= fr)
        sampling_rate = fr // freq #向快对齐

        # one_target_jt = np.load(f[:-1]).astype(np.float32)
        one_target_jt = np.load(f).astype(np.float32)
        one_target_jt = one_target_jt[:freq*20*sampling_rate:sampling_rate,:36]
        target_jt_tensor[i,:one_target_jt.shape[0],:] = torch.tensor(one_target_jt, dtype=torch.float32, device=device)
        target_jt_lenth[i] = one_target_jt.shape[0]
        i += 1
    return target_jt_tensor, target_jt_lenth
