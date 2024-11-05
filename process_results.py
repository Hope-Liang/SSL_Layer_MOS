import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Result path')
parser.add_argument('--result_path', type=str, help='Path to the results folder.')

args = parser.parse_args()

# Example: python process_results.py --result_path=results_bvcc/w2v2_base_bvcc_finetuned_10epoch/layer0

def extract_lines_from_log_files(result_dir):
    # List to hold the extracted lines
    extracted_lines = []

    for i in range(5):
        file_path = os.path.join(result_dir, f'run{i}', 'train.log')
    
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('[Test][31][UTT]'):
                    extracted_lines.append(line.strip())
    
    return extracted_lines

def result_to_num(line, MSE, LCC, SRCC):
    parts = line.split()
    assert len(parts)==13
    MSE.append(float(parts[3]))
    LCC.append(float(parts[7]))
    SRCC.append(float(parts[11]))
    return MSE, LCC, SRCC

def process_single_log(log_path):
    logs = extract_lines_from_log_files(log_path)
    MSE, LCC, SRCC = [], [], []
    for log in logs:
        MSE, LCC, SRCC = result_to_num(log, MSE, LCC, SRCC)
    assert len(MSE)>=4
    return MSE, LCC, SRCC
    
def print_values(MSE, LCC, SRCC):
    print("MSE: ", f"{sum(MSE)/len(MSE):.4f}")
    print("LCC: ", f"{sum(LCC)/len(LCC):.4f}")
    print("SRCC: ", f"{sum(SRCC)/len(SRCC):.4f}")

def process_multi_log(log_folder_path):
    subfolders = [f.path for f in os.scandir(log_folder_path) if f.is_dir()]
    results = {'layer':[], 'MSE':[], 'LCC':[], 'SRCC':[]}
    for subfolder in subfolders:
        layer_num = int(subfolder.split('/')[-1][5:])
        MSE, LCC, SRCC = process_single_log(subfolder)
        results['layer'].append(layer_num)
        results['MSE'].append(MSE)
        results['LCC'].append(LCC)
        results['SRCC'].append(SRCC)
    df = pd.DataFrame(results)
    df = df.sort_values(by='layer')
    df['MSE_mean'] = df['MSE'].apply(lambda x: sum(x) / len(x))
    df['LCC_mean'] = df['LCC'].apply(lambda x: sum(x) / len(x))
    df['SRCC_mean'] = df['SRCC'].apply(lambda x: sum(x) / len(x))
    return df

def plot_curves(log_folder_paths, labels, figure_name):
    plt.figure(figsize=(10,2))
    plt.xlabel('Layer Number')
    plt.ylabel('LCC with MOS')
    plt.xticks(np.arange(0, 25))
    for log_folder_path, label in zip(log_folder_paths, labels):
        df = process_multi_log(log_folder_path)
        if log_folder_path.endswith('w2v2_xlsr_1b') or log_folder_path.endswith('w2v2_xlsr_2b'):
            plt.plot(1+df['layer']//2, df['LCC_mean'], label=label)
        else:
            plt.plot(1+df['layer'], df['LCC_mean'], label=label)
    plt.legend()
    plt.savefig(f'{figure_name}.png')

def plot_grids(model_name, figsize=(12, 3)):
    results_paths = ['results_bvcc', 'results_tencent', 'results_nisqa']
    arrays = []
    for results_path in results_paths:
        log_folder_path = os.path.join(results_path, model_name)
        df = process_multi_log(log_folder_path)
        arrays.append(np.array(df['LCC_mean']))

    normalized_arrays = [(arr - arr.min()) / (arr.max() - arr.min()) for arr in arrays]
    fig, ax = plt.subplots(figsize=figsize)

    for i, arr in enumerate(normalized_arrays):
        for j, val in enumerate(arr):
            color = plt.cm.RdYlGn(val)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color))

    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.set_xticks([])
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(np.arange(1, 13), fontsize=22)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['bvcc', 'tencent', 'nisqa'], fontsize=26)
    ax.set_aspect('equal')

    plt.savefig(f'figs/grid_{model_name}.png')


def plot_grids_colorbar(model_name, figsize=(12, 3)):
    results_paths = ['results_bvcc', 'results_tencent', 'results_nisqa']
    arrays = []
    for results_path in results_paths:
        log_folder_path = os.path.join(results_path, model_name)
        df = process_multi_log(log_folder_path)
        arrays.append(np.array(df['LCC_mean']))

    normalized_arrays = [(arr - arr.min()) / (arr.max() - arr.min()) for arr in arrays]
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1, 1]})
    cmap = plt.cm.RdYlGn

    for i, (ax, arr, norm_arr) in enumerate(zip(axs, arrays, normalized_arrays)):
        for j, val in enumerate(norm_arr):
            color = cmap(val)
            ax.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=color))

        ax.set_xlim(0, figsize[0])
        ax.set_ylim(0, 1)
        if i == 2:
            ax.set_xticks(np.arange(figsize[0]) + 0.5)
            ax.set_xticklabels(np.arange(1, figsize[0]+1), fontsize=15)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=arr.min(), vmax=arr.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.1, pad=0.02/(figsize[0]/12), aspect=2)
        
        ticks = [arr.min(), (arr.min() + arr.max()) / 2, arr.max()]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])  # Format tick labels as needed
        cbar.ax.tick_params(labelsize=15)

    axs[0].set_ylabel('bvcc', fontsize=15)
    axs[1].set_ylabel('tencent', fontsize=15)
    axs[2].set_ylabel('nisqa', fontsize=15)

    plt.tight_layout()
    plt.savefig(f'figs/grid_{model_name}_colorbar.png')


def plot_grids_finetune(dataset_name, figsize=(12, 4), color_all=True):
    
    model_suffixes = ['', f'_{dataset_name}_finetuned_1epoch', f'_{dataset_name}_finetuned_2epoch', f'_{dataset_name}_finetuned_3epoch', f'_{dataset_name}_finetuned_4epoch', f'_{dataset_name}_finetuned_5epoch', f'_{dataset_name}_finetuned_6epoch', f'_{dataset_name}_finetuned_7epoch', f'_{dataset_name}_finetuned_8epoch', f'_{dataset_name}_finetuned_9epoch', f'_{dataset_name}_finetuned_10epoch']
    arrays = []
    for model_suffix in model_suffixes:
        log_folder_path = os.path.join(f'results_{dataset_name}', f'w2v2_base{model_suffix}')
        df = process_multi_log(log_folder_path)
        arrays.append(np.array(df['LCC_mean']))

    all_values = np.concatenate(arrays)
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    if color_all:
        normalized_arrays = [(arr - min_val) / (max_val - min_val) for arr in arrays]
    else:
        normalized_arrays = [(arr - arr.min()) / (arr.max() - arr.min()) for arr in arrays]
    # Plotting the grids
    fig, ax = plt.subplots(figsize=figsize)

    for i, arr in enumerate(normalized_arrays):
        for j, val in enumerate(arr):
            color = plt.cm.RdYlGn(val)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color))
            # ax.text(j + 0.5, i + 0.5, f'{arrays[i][j]:.4f}', ha='center', va='center', fontsize=10, color='black')

    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.set_xticks([])
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(np.arange(1, 13), fontsize=30)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
    ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=30)

    ax.set_xlabel('Layer', fontsize=30)
    ax.set_ylabel('Epoch', fontsize=30)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    # cbar.set_label('LCC Value', fontsize=22)
    cbar.ax.tick_params(labelsize=28)

    plt.savefig(f'figs/grid_finetune_{dataset_name}.png')





# MSE, LCC, SRCC = process_single_log(args.result_path)
# print_values(MSE, LCC, SRCC)

# process_multi_log(args.result_path)

# For Arch Comp
# dataset = 'nisqa'
# PATHS = [f'results_{dataset}/w2v2_base', f'results_{dataset}/hubert_base', f'results_{dataset}/wavlm_base']
# LABELS = ['w2v2', 'hubert', 'wavlm']
# plot_curves(PATHS, LABELS, f'arch_comp_{dataset}')

# For Size Comp
# dataset = 'nisqa'
# PATHS = [f'results_{dataset}/w2v2_xlsr_300m', f'results_{dataset}/w2v2_xlsr_1b', f'results_{dataset}/w2v2_xlsr_2b']
# LABELS = ['w2v2_300m', 'w2v2_1b', 'w2v2_2b']
# plot_curves(PATHS, LABELS, f'size_comp_{dataset}')

# plot_grids('wavlm_base')
# plot_grids_colorbar('hubert_base')

# plot_grids('w2v2_xlsr_2b', (24,3))
# plot_grids_colorbar('w2v2_xlsr_300m', (24,3))

plot_grids_finetune('tencent', (12,11))