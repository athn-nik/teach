import sys
sys.path.append('.')

import argparse
from loguru import logger
import glob
import joblib
from tqdm import tqdm
import os

from teach.utils.file_io import read_json, write_json
from nlp_actions.nlp_consts import fix_spell
'''
RUN EXAMPLE
python divotion/dataset/add_babel_labels.py \
--input-path /is/cluster/nathanasiou/data/amass/processed_amass_smplh_wshape_30fps \ 
--out-path /is/cluster/nathanasiou/data/babel/babel-smplh30fps-gender \
--babel-path /is/cluster/nathanasiou/data/babel/babel_v2.1/
'''
def extract_frame_labels(babel_labels, fps, seqlen):

    seg_ids = []
    seg_acts = []

    if babel_labels['frame_ann'] is None:

        # 'transl' 'pose''betas'
        action_label = babel_labels['seq_ann']['labels'][0]['proc_label']
        seg_ids.append([0, seqlen])
        seg_acts.append(fix_spell(action_label))
    else:
        for seg_an in babel_labels['frame_ann']['labels']:
            action_label = fix_spell(seg_an['proc_label'])

            st_f = int(seg_an['start_t']*fps)
            end_f = int(seg_an['end_t']*fps)
            if end_f > seqlen:
                end_f = seqlen
            seg_ids.append([st_f, end_f])
            seg_acts.append(action_label)

    return seg_ids, seg_acts

def process_data(input_dir, out_dir, amass2babel, babel_data_train,
                 babel_data_val, babel_data_test):

    amass_subsets = glob.glob(f'{input_dir}/*/*')
    babel_keys = list(babel_data_train.keys()) + list(babel_data_test.keys()) + list(babel_data_val.keys())
    dataset_db_lists = {'train': [],
                        'val': [],
                        'test': []}
        
    for am_s_path in amass_subsets:
        amass_subset = joblib.load(am_s_path)
        logger.info(f'Loading the dataset from {am_s_path}')
        
        for sample in tqdm(amass_subset):

            if sample['fname'] not in amass2babel:
                continue
            split_of_seq = amass2babel[sample['fname']]['split']
            babel_seq_id = amass2babel[sample['fname']]['babel_id']
            if split_of_seq == 'train':
                babel_data_seq = babel_data_train[babel_seq_id]
            elif split_of_seq == 'val':
                babel_data_seq = babel_data_val[babel_seq_id]
            elif split_of_seq == 'test':
                babel_data_seq = babel_data_test[babel_seq_id]


            seg_indices, seg_actions = extract_frame_labels(babel_data_seq,
                                                            sample['fps'],
                                                            sample['poses'].shape[0])

            for index, seg in enumerate(seg_indices):

                sample_babel = {}
                sample_babel['fps'] = sample['fps']
                sample_babel['fname'] = sample['fname']
                for ams_k in ['poses', 'trans', 'joint_positions', 'markers']:
                    sample_babel[ams_k] = sample[ams_k][seg[0]:seg[1]]
                sample_babel['action'] = seg_actions[index]
                dataset_db_lists[split_of_seq].append(sample_babel)
    os.makedirs(out_dir, exist_ok=True)
    for k, v in dataset_db_lists.items():
        joblib.dump(v, f'{out_dir}/{k}.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-path', required=True, type=str,
                        help='input path of AMASS data in unzipped format without anything else.')
    parser.add_argument('--babel-path', required=True, type=str,
                        help='input path of AMASS data in unzipped format without anything else.')
    parser.add_argument('--out-path', required=True, type=str,
                        help='input path of AMASS data in unzipped format without anything else.')


    args = parser.parse_args()
    input_dir = args.input_path
    babel_dir = args.babel_path
    out_dir = args.out_path
    logger.info(f'Input arguments: \n {args}')

    babel_data_train = read_json(f'{babel_dir}/train.json')
    babel_data_val = read_json(f'{babel_dir}/val.json')
    babel_data_test = read_json(f'{babel_dir}/test.json')

    amass2babel = read_json(f'{babel_dir}/id2fname/amass-path2babel.json')

    db = process_data(input_dir, out_dir, amass2babel, babel_data_train,
                      babel_data_val, babel_data_test)
