import os
from extract_ssl_features import extract_features_audio

_BASE_PATH = '/proj/berzelius-2023-179/users/datasets/'
_DEFAULT_BVCC_PATH = ['BVCC/DATA/wav']
_DEFAULT_BVCC_OUT_PATH = ['BVCC/DATA/']
_DEFAULT_TENCENT_PATH = ['Tencent_corpus/withReverberationTrainDev', 'Tencent_corpus/withoutReverberationTrainDev']
_DEFAULT_TENCENT_OUT_PATH = ['Tencent_corpus/withReverberationTrainDev_features', 'Tencent_corpus/withoutReverberationTrainDev_features']
_DEFAULT_NISQA_PATH = ['NISQA_Corpus/NISQA_TRAIN_SIM/deg', 'NISQA_Corpus/NISQA_TRAIN_LIVE/deg', 'NISQA_Corpus/NISQA_VAL_SIM/deg', 'NISQA_Corpus/NISQA_VAL_LIVE/deg', 'NISQA_Corpus/NISQA_TEST_LIVETALK/deg']
_DEFAULT_NISQA_OUT_PATH = ['NISQA_Corpus/features/NISQA_TRAIN_SIM/', 'NISQA_Corpus/features/NISQA_TRAIN_LIVE/', 'NISQA_Corpus/features/NISQA_VAL_SIM/', 'NISQA_Corpus/features/NISQA_VAL_LIVE/', 'NISQA_Corpus/features/NISQA_TEST_LIVETALK/']

dataset = 'tencent' #'bvcc', 'tencent', 'nisqa'
layers_to_use = list(range(0, 12))
model_name = 'w2v2_base_tencent_finetuned_10epoch'
target_duration = 8.0
save_output = True

if dataset == 'bvcc':
    _IN_PATH = _DEFAULT_BVCC_PATH
    _OUT_PATH = _DEFAULT_BVCC_OUT_PATH
if dataset == 'tencent':
    _IN_PATH = _DEFAULT_TENCENT_PATH
    _OUT_PATH = _DEFAULT_TENCENT_OUT_PATH
if dataset == 'nisqa':
    _IN_PATH = _DEFAULT_NISQA_PATH
    _OUT_PATH = _DEFAULT_NISQA_OUT_PATH

for in_path, out_path in zip(_IN_PATH, _OUT_PATH):
    in_path = os.path.join(_BASE_PATH, in_path)
    out_path = os.path.join(_BASE_PATH, out_path)

    wav_paths = os.listdir(in_path)
    for i in range(len(wav_paths)):
        wav_paths[i] = os.path.join(in_path, wav_paths[i])

    features_audio = extract_features_audio(wav_paths, layers_to_use, model_name, target_duration, save_output, out_path)
