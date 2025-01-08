import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

colors = sns.color_palette()

def plotting_fun(
    data,
    mpi_implementations,
    cuda_implementations,
    serial_implementations,
    open_implementations,
    mpi_implementations_name,
    cuda_implementations_name,
    serial_implementations_name,
    open_implementations_name,
    threads,
    N2,
    N2_c,
    filename_list,
    title_list,
    plot_path,
    set_threads,
    set_n2,
    runtime,
    with_np = False
):
    implementations = (
        cuda_implementations
        + mpi_implementations
        + serial_implementations
        + open_implementations
    )

    df = pd.DataFrame(data)
    baseline_df = df.loc[df["implementation"] == "serial_base"]
    base_N = baseline_df["N"].tolist()
    if not runtime:
        for imp in implementations:
            if imp in cuda_implementations:
                helper = df.loc[df["implementation"] == imp]
                N_list = helper["N"].tolist()
                for nn in N_list:
                    if nn in base_N:
                        m = df.loc[
                            (df["implementation"] == imp) & (df["N"] == nn), "mean"
                        ].iloc[0]
                        base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                        df.loc[
                            (df["implementation"] == imp) & (df["N"] == nn), "speedup"
                        ] = (base_m / m)

            elif imp in serial_implementations:
                helper = df.loc[df["implementation"] == imp]
                N_list = helper["N"].tolist()
                for nn in N_list:
                    if nn in base_N:
                        m = df.loc[
                            (df["implementation"] == imp) & (df["N"] == nn), "mean"
                        ].iloc[0]
                        base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                        df.loc[
                            (df["implementation"] == imp) & (df["N"] == nn), "speedup"
                        ] = (base_m / m)
            else:
                helper = df.loc[df["implementation"] == imp]
                N_list = helper["N"].tolist()
                for nn in N_list:
                    if nn in base_N:
                        base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                        df.loc[
                            (df["implementation"] == imp) & (df["N"] == nn), "speedup"
                        ] = (base_m / df["mean"])
        
    df = df.drop_duplicates(subset =['implementation','N','tr'], keep='first')
    if with_np:
        data_np = []
        np_impl = []
        bench_directory = "plot/np_bench_gemver"
        for filename in os.listdir(bench_directory):
            if "truth" not in filename:
                dir = os.path.join(bench_directory, filename)
                for single_file in os.listdir(dir):
                        file_path = os.path.join(dir, single_file)
                        n = single_file.split('_')[-1]
                        n = n.split('.')[-2]
                        n2 = n[2:]
                        temp_data = {}
                        temp_data['N'] = n2
                        temp_data['N2'] = int(n2)
                        temp_data['tr'] = int(n2)
                        if temp_data['N2'] <=14020 :
                            with open(file_path) as f:
                                d = json.load(f)
                                
                                temp_data['threads'] = int(n2)
                                temp_data['speedup'] = d['timing'][0]
                                temp_data['implementation'] = filename
                                temp_data['algorithm'] = 'adi'
                                np_impl.append(filename)
                                temp_data['mean'] = d['timing'][0]
                                temp_data['deviation'] = d['timing'][4]
                                temp_data['deviation_window'] = d['timing'][4]
                                data_np.append(temp_data)

        df_np = pd.DataFrame(data_np)

        frames = [df, df_np]

        df = pd.concat(frames)
        
        mpi_implementations.append("dace_cpu_auto_opt")
        cuda_implementations.append("dace_cpu_auto_opt")
        open_implementations.append("dace_cpu_auto_opt")
        serial_implementations.append("dace_cpu_auto_opt")

    mpi_df = df.loc[df["implementation"].isin(mpi_implementations)]
    cuda_df = df.loc[df["implementation"].isin(cuda_implementations)]
    open_df = df.loc[df["implementation"].isin(open_implementations)]
    serial_df = df.loc[df["implementation"].isin(serial_implementations)]

    mpi_df_8 = mpi_df.loc[(mpi_df["threads"] == set_threads) |( mpi_df["implementation"] == "serial_base") |( mpi_df["implementation"] == "dace_cpu_auto_opt")]
    mpi_df_4000 = mpi_df.loc[mpi_df["N2"] == set_n2]
    open_df_8 = open_df.loc[(open_df["threads"] == set_threads) | (open_df["implementation"] == "serial_base") | (open_df["implementation"] == "serial_merge_loops") | (open_df["implementation"] == "dace_cpu_auto_opt")]
    open_df_4000 = open_df.loc[open_df["N2"] == set_n2]
    #serial_df = serial_df.loc[(serial_df["threads"] == set_threads) |( serial_df["implementation"] == "cuda_improved6")  |( serial_df["implementation"] == "dace_cpu_auto_opt")]

    # mpi_df_8 = mpi_df_8.loc[mpi_df_8['N2'] % 1000 == 0]
    # open_df_8 = open_df_8.loc[open_df_8['N2'] % 1000 == 0]
    # #open_df_4000 = open_df_4000.loc[open_df_4000['tr'] <17]
    # cuda_df = cuda_df.loc[cuda_df['N2'] % 1000 == 0]
    # serial_df = serial_df.loc[serial_df['N2'] % 1000 == 0]
    baseline_df = df.loc[(df["implementation"] == "serial_base") & (df["N2"] == 4000)]
    baset=baseline_df['speedup'].iloc[0]
    mpi_df_8 = mpi_df_8.sort_values("N2")
    mpi_df_4000 = mpi_df_4000.sort_values("tr")
    open_df_4000 = open_df_4000.sort_values("tr")
    cuda_df = cuda_df.sort_values("N2")
    open_df_8 = open_df_8.sort_values("N2")
    serial_df = serial_df.sort_values("N2")
    if with_np:
        np_impl_list = ["numpy_default", "numba_nopython_mode", "numba_nopython_mode_parallel", "pythran_default", "dace_cpu_fusion", "dace_cpu_parallel", "dace_cpu_auto_opt"]
        
        np_impl_titles = ["NumPy", "Numba (NoPython Mode)", "Numba (Parallel, NoPython Mode)", "Pythran", "DaCe (CPU Fusion)", "DaCe (CPU Parallel)", "DaCe (CPU Auto-Optimized)"]
    
        plot_list = [
            np_impl_list,
            mpi_implementations,
            cuda_implementations,
            serial_implementations,
            open_implementations,
            mpi_implementations,
            open_implementations
        ]
        plot_list_names = [
            np_impl_titles,
            mpi_implementations_name,
            cuda_implementations_name,
            serial_implementations_name,
            open_implementations_name,
            mpi_implementations_name,
            open_implementations_name
        ]
        df_np = df_np.sort_values("N2")

        df_list = [df_np, mpi_df_8, cuda_df, serial_df, open_df_8, mpi_df_4000, open_df_4000]
        i_val = 4
    else:
        plot_list = [
            mpi_implementations,
            cuda_implementations,
            serial_implementations,
            open_implementations,
            mpi_implementations,
            open_implementations
        ]
        plot_list_names = [
            mpi_implementations_name,
            cuda_implementations_name,
            serial_implementations_name,
            open_implementations_name,
            mpi_implementations_name,
            open_implementations_name
        ]
        i_val = 3
        df_list = [ mpi_df_8, cuda_df, serial_df, open_df_8, mpi_df_4000, open_df_4000]

    for i, imp_list in enumerate(plot_list):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax = plt.axes(facecolor='#E6E6E6')
        plt.grid(color='w', linestyle='solid')
        df_current = df_list[i]
        filename = filename_list[i]
        labels_list = plot_list_names[i]
        for j, name in enumerate(imp_list):
            helper_df = df_current.loc[df_current["implementation"] == name]
            print(labels_list)
            print(j)
            label_name = labels_list[j]
            y = helper_df.speedup.to_numpy()
            x = helper_df.tr.to_numpy() if i > i_val else helper_df.N2.to_numpy()
            y_std = helper_df.deviation.to_numpy()
            error = y_std
            indx = j % 9
            color = colors[indx]
            marker = "^"
            line_style = "-"

            ax.errorbar(
                x,
                y,
                yerr=error,
                markersize=4,
                linewidth=1,
                elinewidth=0.4,
                capsize=1.5,
                fmt=line_style + marker,
                alpha=0.8,
                label=label_name,
                color=color,
                ecolor=color,
            )

        if i > i_val:
            if i == i_val+1:
                ax.set_xlabel("# processes")
            else:
                ax.set_xlabel("# threads")
            x = [1, 2, 4, 6, 8, 12, 16]
            plt.xticks(x)
            ax.axhline(y=baset, color="1", linestyle="--", label = "Polybench baseline")
        else:
            ax.set_xlabel("N size")
            ax.set_yscale('log')

        
        if runtime:
            ax.set_ylabel("Runtime (sec)")
        else:
            ax.set_ylabel("x speedup")

        ax.legend(loc="upper right")
        if not runtime:
            ax.axhline(y=1, color="0.8", linestyle="--")

        plt_title = title_list[i]
        plt.title(plt_title)
        file_path = filename + ".png"
        plt.savefig(file_path)
