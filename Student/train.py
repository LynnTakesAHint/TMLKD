import os
import tools.config as config
import warnings
import glob
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
print('GPU is:', config.GPU)
from geo_rnns.tmlkd_trainer_neutraj import TMLKD_Trainer_NeuTraj
from geo_rnns.tmlkd_trainer_t3s import TMLKD_Trainer_T3S

if __name__ == '__main__':
    print(config.config_to_str())
    if config.base_model.lower() == 'neutraj':
        trajrnn = TMLKD_Trainer_NeuTraj(tagset_size=config.d, batch_size=config.batch_size,
                                 sampling_num=config.sampling_num)
    else:
        trajrnn = TMLKD_Trainer_T3S(tagset_size=config.d, batch_size=config.batch_size,
                                 sampling_num=config.sampling_num)
    trajrnn.data_prepare()
    trajrnn.TMLKD_train()
    acc1 = trajrnn.trained_model_eval(load_model=trajrnn.best_model)