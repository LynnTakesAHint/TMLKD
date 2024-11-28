import os
import tools.config as config
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from geo_rnns.teacher_trainer_NeuTraj import NeutrajTrainer
from geo_rnns.teacher_trainer_T3S import T3STrainer

if __name__ == '__main__':
    print(config.config_to_str())
    if config.base_model == 'T3S':
        trajrnn = T3STrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    elif config.base_model == 'NeuTraj':
        trajrnn = NeutrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                                 sampling_num=config.sampling_num)
    else:
        raise NotImplementedError
    trajrnn.data_prepare()
    trajrnn.train()
