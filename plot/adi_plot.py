import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

directory = "plot\\results\\adi"
mpi_implementations = [ 'mpi_1']
cuda_implementations = ['cuda_multithreaded12', 'cuda_multithreaded4', 'cuda_multithreaded5', 'cuda_multithreaded6', 'cuda_multithreaded8', 'cuda_multithreaded9']
serial_implementations = ['serial_block', 'serial_opt']
open_implementations = []
implementations = ['cuda_multithreaded12', 'cuda_multithreaded4', 'cuda_multithreaded5', 'cuda_multithreaded6', 'cuda_multithreaded8', 'cuda_multithreaded9', 'mpi_1', 'serial_block', 'serial_opt']
threads = [ '1', '2', '3','4','6','8','12','16']
N2 = ['1000', '2000', '3000', '4000','5000','6000','7000','8000','9000','10000','11000','12000','13000','14000', '15000','1024','2048', '4096' ]
N2_c = ['1000', '2000', '3000', '4000','5000','1024','2048', '4096' ]
data = []
colors = sns.color_palette()

for filename in os.listdir(directory):
    if "truth" not in filename:
        dir = os.path.join(directory, filename)
        for single_file in os.listdir(dir):
            if "latest" not in single_file:
                file_path = os.path.join(dir, single_file)
                t_temp = single_file.split('_')[-1]
                t = t_temp.replace('.json', '')
                n = single_file.split('_')[-2]
                n2_t = n[2:]
                n2 = n2_t.replace('TSTEPS15', '')
                with open(file_path) as f:
                    d = json.load(f)
                    temp_data = {}
                    temp_data['threads'] = t
                    temp_data['N'] = n2
                    temp_data['N2'] = int(n2)
                    temp_data['speedup'] = 0
                    temp_data['implementation'] = filename
                    temp_data['mean'] = d['timing'][0]
                    temp_data['min'] = d['timing'][1]
                    temp_data['max'] = d['timing'][2]
                    temp_data['deviation'] = d['timing'][4]
                    data.append(temp_data)

df = pd.DataFrame(data)
baseline_df = df.loc[df['implementation'] == 'serial_base']
help = df.loc[df['implementation'] == 'mpi_1']
print(help)
for imp in implementations:
    if(imp in cuda_implementations):
        for nn in N2_c:
            m = df.loc[(df['implementation'] == imp) & (df['N'] == nn), 'mean'].iloc[0]
            base_m = baseline_df.loc[(baseline_df['N'] == nn), 'mean'].iloc[0]
            df.loc[(df['implementation'] == imp) & (df['N'] == nn), 'speedup'] = base_m /m 
    elif(imp in serial_implementations):
        for nn in N2:
            m = df.loc[(df['implementation'] == imp) & (df['N'] == nn), 'mean'].iloc[0]
            base_m = baseline_df.loc[(baseline_df['N'] == nn), 'mean'].iloc[0]
            df.loc[(df['implementation'] == imp) & (df['N'] == nn), 'speedup'] = base_m /m 
    else:
        for nn in N2:
            for thread in threads:
                m = df.loc[(df['implementation'] == imp) & (df['N'] == nn) & (df['threads'] == thread), 'mean'].iloc[0]
                base_m = baseline_df.loc[(baseline_df['N'] == nn), 'mean'].iloc[0]
                df.loc[(df['implementation'] == imp) & (df['N'] == nn) & (df['threads'] == thread), 'speedup'] = base_m /m 

mpi_df = df.loc[df['implementation'].isin(mpi_implementations)]
cuda_df = df.loc[df['implementation'].isin(cuda_implementations)]
open_df = df.loc[df['implementation'].isin(open_implementations)]
serial_df = df.loc[df['implementation'].isin(serial_implementations)]

mpi_df_8 = mpi_df.loc[mpi_df['threads']=='8'] #better than 8 discuss
open_df_8 = open_df.loc[open_df['threads']=='8']

mpi_df_8 = mpi_df_8.sort_values('N2')
cuda_df = cuda_df.sort_values('N2')
open_df_8 = open_df_8.sort_values('N2')
serial_df = serial_df.sort_values('N2')

plot_list = [mpi_implementations, cuda_implementations, serial_implementations, open_implementations]
df_list = [mpi_df_8, cuda_df, serial_df, open_df_8]
filename_list = ['mpi_t_8', 'cuda', 'serial', 'open_t_8']
title_list = ['ADI MPI - 8 processes', 'ADI cuda', 'ADI serial', 'ADI OpenMp - 8 threads']

for i,imp_list in enumerate(plot_list):
    fig, ax = plt.subplots(figsize=(9,5))
    df_current = df_list[i]
    filename = filename_list[i]
    for j, name in enumerate(imp_list):
        helper_df = df_current.loc[df['implementation']==name]
        y = helper_df.speedup.to_numpy()
        x = helper_df.N2.to_numpy()
        y_std = helper_df.deviation.to_numpy()
        error = y_std
        lower = y - error
        upper = y + error
        indx = j % 9
        color = colors[indx]
        marker = 'o'
        line_style ='-'
        label_name_ar = name.split('_')[1:]
        label_name = " ".join(str(element) for element in label_name_ar)
        ax.errorbar(x, y, yerr=error,
                          markersize=4, linewidth=1, elinewidth=0.4, capsize=1.5,
                         fmt=line_style+marker,
                         alpha=0.8, label=label_name, color=color, ecolor=color,)
        ax.axhline(y=1, color = '0.8', linestyle='--')
        # ax.text(x[-1], 1, 'speedup=1', ha='left', va='center')
        # ax.plot(x,y, color=color, label = name)
        # ax.plot(x, lower,  alpha=0.1)
        # ax.plot(x, upper,  alpha=0.1)
        # ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set_xlabel('N size')
    ax.set_ylabel('x speedup')
    if (i!=0):
        ax.legend(loc="upper right")
    plt.title(title_list[i])

    file_path = "plot\\imgs\\adi" + filename + ".png"
    plt.savefig(file_path)
