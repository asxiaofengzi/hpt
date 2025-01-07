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
    fr = file[-12:-9]
    fr = int(fr[1:]) if fr[0]=='_' else int(fr)
    assert(freq<= fr)
    sampling_rate = fr // freq #向快对齐
    one_target_jt = np.load(f"/home/fleaven/dataset/test_amass/{file}").astype(np.float32)
    one_target_jt = one_target_jt[::sampling_rate]
    target_jt = torch.from_numpy(one_target_jt).to(device).unsqueeze(0)
    #target_jt += offset

    size = torch.tensor([target_jt.shape[1]]).to(device)
    return target_jt, size

