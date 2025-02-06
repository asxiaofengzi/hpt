import torch

class vpid_controler:
    def __init__(self, num_env, num_dof, Kp, Ki, Kd, dec, torque_limit):
        self.data = torch.zeros((num_env, num_dof, 5),dtype=torch.float, device=torque_limit.device)
        self.data[:,:,[0,1,2]] = torch.tensor([Kp, Ki, Kd], dtype=torch.float, device=self.data.device)
        self.data[:,:,3] = 1
        self.dec = dec
        self.torque_limit = torque_limit

    #   3个decimation的target_pos,最近1个decimation的dof_pos，
    #   最近的def_pos模仿对齐第二个target_pos 
    def compute_torque(self, torques, target_pos, dof_pos):
        # actions = actions.view(3,-1,29)
        # self.data[:,:,0] += (actions[0]>1)*0.01
        # self.data[:,:,0] += (actions[0]<-1)*-0.01
        # self.data[self.data[:,:,0]>10]=10.
        # self.data[self.data[:,:,0]<0.01]=0.01
        # self.data[:,:,1] += (actions[1]>1)*0.01
        # self.data[:,:,1] += (actions[1]<-1)*-0.01
        # self.data[self.data[:,:,1]>100]=100.
        # self.data[self.data[:,:,1]<0.01]=0.01
        # self.data[:,:,2] += (actions[2]>1)*0.01
        # self.data[:,:,2] += (actions[2]<-1)*-0.01
        # self.data[self.data[:,:,2]>100]=100.
        # self.data[self.data[:,:,2]<0.01]=0.01
        i = target_pos[1,:,:] - dof_pos[-1,:,:]
        v = (dof_pos[-1,:,:] - dof_pos[-2,:,:])*self.dec
        p = (target_pos[2,:,:] - target_pos[1,:,:]) - v
        # ta = (target_pos[2,:,:] - target_pos[1,:,:]) - (target_pos[-1,:,:] - target_pos[0,:,:])
        # a = (dof_pos[-1,:,:] + dof_pos[-3,:,:]- dof_pos[-2,:,:]-dof_pos[-2,:,:])*self.dec
        # d = ta - a
        output = (self.data[:,:,0] * p + self.data[:,:,1] * i)*self.torque_limit
        return (output+self.data[:,:,4])*self.data[:,:,3]

        # a1 = (dof_pos[-1-self.dec,:,:] + dof_pos[-3-self.dec,:,:]- dof_pos[-2-self.dec,:,:]-dof_pos[-2-self.dec,:,:])*self.dec
        # te = torques[1,:,:]-torques[0,:,:]
        # r = 1+d/(torch.abs(a)+1e-2)
        # r[r>10]=10
        # r[r<0.2]=0.2
        # self.data[:,:,3]*=r
        # self.data[torch.abs(te[:])> self.torque_limit*0.02][:,:,3]*=(ta - a)/(a-a1)*te

    def update_pos(self, torque, target_pos, dof_pos):
        a = (dof_pos[-1] + dof_pos[-3]- dof_pos[-2]-dof_pos[-2])*self.dec
        ta = (target_pos[2] - target_pos[1]) - (target_pos[1] - target_pos[0])
        ae = ta - a
        self.data[:,:,3] = 1
        self.data[:,:,3] *= 1+(((torque > 0) & (ae > 0.01)) | ((torque < 0) & (ae < -0.01)))*0.1
        self.data[:,:,3] *= 1+(((torque > 0) & (ae < -0.01)) | ((torque < 0) & (ae > 0.01)))*-0.09
        self.data[:,:,4] = (ae > 0)*0.003*self.torque_limit
        self.data[:,:,4] += (ae < 0)*-0.003*self.torque_limit
        return (torque+self.data[:,:,4])*self.data[:,:,3]

'''
        tensor([1.0000e+02, 1.0000e+02, 9.9984e+01, 1.0421e+01, 1.0000e+02, 1.0000e+02,
        1.0000e+02, 1.0000e+02, 1.0000e+02, 1.0000e+02, 1.5912e-02, 9.9998e+01,
        1.0000e-02, 1.0000e+02, 1.0000e+02, 1.0256e-02, 1.0000e+02, 1.0000e+02,
        1.0000e+02, 1.2224e-02, 1.0000e+02, 8.9542e+01, 1.0000e+02, 9.4324e+01,
        1.0000e-02, 9.9999e+01, 1.0000e+02, 1.0000e-02, 1.0000e+02],
       device='cuda:0')
        
        tensor([
        [0.4948, 0.6021, 0.6476],
        [7.4663, 9.9756, 9.9544],
        [3.5931, 4.5909, 4.7317],
        [0.5162, 0.7514, 0.7315],
        [5.9710, 8.0394, 7.8894],
        [0.4705, 0.5482, 0.5682],
        
        [1.4772, 1.8520, 1.8251],
        [0.5105, 0.6153, 0.6007],
        [5.4877, 7.2979, 7.1002],
        [2.4851, 3.1465, 3.0827],
        [3.8130, 4.9812, 4.8834],
        [2.4317, 3.0737, 3.2418],

        [1.3596, 1.7605, 1.7485],
        [1.1045, 1.1962, 1.2654],
        [1.5857, 2.1025, 2.1531],

        [3.9477, 5.0855, 5.0505],
        [2.0903, 2.9216, 2.8184],
        [5.1971, 6.7257, 6.7143],
        [6.9821, 8.8147, 8.9306],
        [3.5977, 4.5700, 4.6676],
        [2.9598, 4.1729, 4.0782],
        [0.1932, 0.2807, 0.2580],
        [7.4344, 9.5904, 9.6177],
        [4.0204, 5.1942, 5.1997],
        [3.4090, 4.3522, 4.3833],
        [1.3462, 1.6511, 1.5702],
        [1.7733, 2.5297, 2.4197],
        [2.2735, 3.0932, 3.0846],
        [0.9101, 1.1512, 1.1055]], device='cuda:0')

        tensor([[0.7818, 1.0333, 1.0636],
        [3.3260, 4.4841, 4.4433],
        [3.0333, 3.9545, 3.8765],
        [0.8592, 1.0311, 1.0404],
        [1.8211, 2.2867, 2.2465],
        [1.1452, 1.4293, 1.4351],
        [1.9306, 2.3046, 2.3433],
        [1.4487, 1.7681, 1.9320],
        [1.5333, 1.8940, 1.9527],
        [1.1141, 1.3186, 1.2840],
        [1.6799, 2.2068, 2.1161],
        [3.1253, 3.9842, 4.1294],
        [2.5373, 3.2804, 3.3650],
        [0.3948, 0.4877, 0.5046],
        [2.4502, 3.2685, 3.3202],
        [1.1168, 1.4723, 1.4031],
        [1.2248, 1.6732, 1.7363],
        [3.0415, 3.5355, 3.5273],
        [2.0380, 2.6247, 2.6556],
        [3.7075, 4.6400, 4.6464],
        [3.4378, 4.4410, 4.3856],
        [5.5174, 7.3143, 7.1495],
        [2.1714, 2.7712, 2.7656],
        [3.5784, 4.3753, 4.4744],
        [0.7502, 0.9066, 0.8996],
        [0.5810, 0.7853, 0.8161],
        [3.3386, 4.1660, 4.1388],
        [1.6370, 2.2033, 2.1294],
        [5.4571, 7.5908, 7.3813]], device='cuda:0') '''