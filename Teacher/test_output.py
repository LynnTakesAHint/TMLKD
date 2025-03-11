import os
import glob
import warnings

warnings.filterwarnings('ignore')

import tools.config as config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from geo_rnns.share_embedding_obtain import TeacherModelEmbeddingGetter


def get_filenames(bd, _target_distance):
    folder_path = f'{config.data_folder}/model/'  # you can replace the directory when needed
    base_model = config.base_model
    save_tag = config.save_tag
    model_dicts = {}
    data_name = 't' + config.data_name[1:]
    for net_type in ['decoder', 'private']:
        target_pattern = f"{data_name}_{bd}_{_target_distance}_{base_model}_{net_type}_best_model{save_tag}.h5"
        pattern = os.path.join(folder_path, target_pattern)
        filename = glob.glob(pattern)[0]
        model_dicts[net_type] = filename
    target_pattern = f"{data_name}_{bd}_{base_model}_share_best_model{save_tag}.h5"
    pattern = os.path.join(folder_path, target_pattern)
    filename = glob.glob(pattern)[0]
    model_dicts['share'] = filename
    return model_dicts


if __name__ == '__main__':
    # Remember to change mode in config.py into 'output'
    print(config.config_to_str())
    trajrnn = TeacherModelEmbeddingGetter(tagset_size=config.d, batch_size=config.batch_size,
                                          sampling_num=config.sampling_num)
    trajrnn.data_prepare()
    share_embedding = None
    for idx, target_distance in enumerate(config.source_distance):
        print('Evaluate distance {}'.format(target_distance))
        bd = config.distance_type
        filename_dict = get_filenames(bd, target_distance)  # you can also directly write file names here
        print(filename_dict)
        trajrnn.test_models(filename_dict, idx, 100000)
