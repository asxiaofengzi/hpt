import numpy as np
import time
import mujoco
import mujoco.viewer


def mj_play(data, fr):
    m = mujoco.MjModel.from_xml_path('/mnt/data1/xiaofengzi/hpt/HST/legged_gym/resources/robots/g1_description/g1_29dof_rev_1_0.urdf')
    m.opt.timestep = 1.0/fr
    d = mujoco.MjData(m)

    len = data.shape[0]
    step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and step<len:
            step_start = time.time()

            d.qpos = data[step, :7+29]
            d.qpos[3] = data[step, 6] # to w first
            d.qpos[4:7] = data[step, 3:6]

            mujoco.mj_forward(m, d)
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            step += 1
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def read_rtj(fpath):
    #get frame rate from file name
    # fr = fpath[-12:-9]
    # fr = int(fr[1:]) if fr[0]=='_' else int(fr)
    fr = 120

    jpos = np.load(fpath)
    jpos[:,2] += 0.793
    return jpos,fr

if __name__ == '__main__':
    jpos, fr = read_rtj('/mnt/data1/xiaofengzi/hpt/HST/legged_gym/data/111_120_222.npy')
    mj_play(jpos, fr)
