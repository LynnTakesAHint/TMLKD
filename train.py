import os
import tools.config as config
import warnings
import glob
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
print('GPU is:', config.GPU)
from geo_rnns.tmlkd_trainer import TMLKD_Trainer

if __name__ == '__main__':
    print(config.config_to_str())
    trajrnn = TMLKD_Trainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare()
    if config.mode == 'train':
        trajrnn.TMLKD_train()
        acc1 = trajrnn.trained_model_eval(load_model=trajrnn.best_model)
    else:
        last_file = ""
        acc1 = trajrnn.trained_model_eval(load_model=last_file)
        pass
        # pattern = './model/*%s*ours_.txt'%config.distance_type
        # matching_files = glob.glob(pattern)
        # if matching_files:
        #     latest_file = max(matching_files, key=os.path.getmtime)
        #     acc1 = trajrnn.trained_model_eval(load_model=latest_file)
        #     print(config.config_to_str())
        # else:
        #     print("Not Found!")
