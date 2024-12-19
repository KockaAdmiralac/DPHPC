import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


colors = sns.color_palette()

def plotting_fun(
    data,
    mpi_implementations,
    cuda_implementations,
    serial_implementations,
    open_implementations,
    threads,
    N2,
    N2_c,
    filename_list,
    title_list,
    plot_path,
    set_threads,
    set_n2
):
    implementations = (
        cuda_implementations
        + mpi_implementations
        + serial_implementations
        + open_implementations
    )

    df = pd.DataFrame(data)
    baseline_df = df.loc[df["implementation"] == "serial_base"]
    for imp in implementations:
        if imp in cuda_implementations:
            for nn in N2_c:
                helper = df.loc[df["implementation"] == imp]
                if nn in helper.N:
                    m = df.loc[
                        (df["implementation"] == imp) & (df["N"] == nn), "mean"
                    ].iloc[0]

                    base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                    df.loc[(df["implementation"] == imp) & (df["N"] == nn), "speedup"] = (
                        base_m / m
                    )
        elif imp in serial_implementations:
            for nn in N2:
                helper = df.loc[df["implementation"] == imp]
                if nn in helper.N:
                    m = df.loc[
                        (df["implementation"] == imp) & (df["N"] == nn), "mean"
                    ].iloc[0]
                    base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                    df.loc[(df["implementation"] == imp) & (df["N"] == nn), "speedup"] = (
                        base_m / m
                    )
        else:
            for nn in N2:
                for thread in threads:
                    helper = df.loc[df["implementation"] == imp]
                    if nn in helper.N:
                        if thread in helper.threads:
                            m = df.loc[
                                (df["implementation"] == imp)
                                & (df["N"] == nn)
                                & (df["threads"] == thread),
                                "mean",
                            ].iloc[0]
                            base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                            df.loc[
                                (df["implementation"] == imp)
                                & (df["N"] == nn)
                                & (df["threads"] == thread),
                                "speedup",
                            ] = (
                                base_m / m
                            )

    mpi_df = df.loc[df["implementation"].isin(mpi_implementations)]
    cuda_df = df.loc[df["implementation"].isin(cuda_implementations)]
    open_df = df.loc[df["implementation"].isin(open_implementations)]
    serial_df = df.loc[df["implementation"].isin(serial_implementations)]

    mpi_df_8 = mpi_df.loc[mpi_df["threads"] == set_threads]
    mpi_df_4000 = mpi_df.loc[mpi_df["N2"] == set_n2]
    open_df_8 = open_df.loc[open_df["threads"] == set_threads]
    open_df_4000 = open_df.loc[open_df["N2"] == set_n2]

    mpi_df_8 = mpi_df_8.sort_values("N2")
    mpi_df_4000 = mpi_df_4000.sort_values("tr")
    open_df_4000 = open_df_4000.sort_values("tr")
    cuda_df = cuda_df.sort_values("N2")
    open_df_8 = open_df_8.sort_values("N2")
    serial_df = serial_df.sort_values("N2")

    plot_list = [
        mpi_implementations,
        cuda_implementations,
        serial_implementations,
        open_implementations,
        mpi_implementations,
        open_implementations,
    ]
    df_list = [mpi_df_8, cuda_df, serial_df, open_df_8, mpi_df_4000, open_df_4000]
    print(mpi_df_8)
    for i, imp_list in enumerate(plot_list):
        fig, ax = plt.subplots(figsize=(9, 5))
        df_current = df_list[i]
        filename = filename_list[i]
        for j, name in enumerate(imp_list):
            helper_df = df_current.loc[df["implementation"] == name]
            y = helper_df.speedup.to_numpy()
            x = helper_df.tr.to_numpy() if i > 3 else helper_df.N2.to_numpy()
            y_std = helper_df.deviation.to_numpy()
            error = y_std
            indx = j % 9
            color = colors[indx]
            marker = "^"
            line_style = "-"
            label_name_ar = name.split("_")[1:]
            label_name = " ".join(str(element) for element in label_name_ar)
            ax.errorbar(
                x,
                y,
                markersize=4,
                linewidth=1,
                elinewidth=0.4,
                capsize=1.5,
                fmt=line_style+marker,
                alpha=0.8,
                label=label_name,
                color=color,
                ecolor=color,
            )
            
        if i > 3:
            ax.set_xlabel("# threads")
        else:
            ax.set_xlabel("N size")
        ax.set_ylabel("x speedup")
        if i != 0:
            ax.legend(loc="upper right")

        ax.axhline(y=1, color="0.8", linestyle="--")
        # plt.yticks(np.arange(-12, 12, 2))
        # plt.ylim((0,10))
        plt_title = title_list[i]
        #ax.plot(x, x, color = 'red', linestyle="--", label = "Ideal linear bound")
        plt.title(plt_title)

        file_path = filename + ".png"
        plt.savefig(file_path)
