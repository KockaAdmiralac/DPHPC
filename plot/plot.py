import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

colors = sns.color_palette()


def plotting(
    directory,
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
    set_n2,
    gemver=True,
):
    data = []
    implementations = (
        cuda_implementations
        + mpi_implementations
        + serial_implementations
        + open_implementations
    )
    for filename in os.listdir(directory):
        if "truth" not in filename:
            dir = os.path.join(directory, filename)
            for single_file in os.listdir(dir):
                if "latest" not in single_file:
                    file_path = os.path.join(dir, single_file)
                    t_temp = single_file.split("_")[-1]
                    t = t_temp.replace(".json", "")
                    n = single_file.split("_")[-2]
                    if gemver:
                        n2 = n[2:]
                    else:
                        n2_t = n[2:]
                        n2 = n2_t.replace("TSTEPS15", "")
                    with open(file_path) as f:
                        d = json.load(f)
                        temp_data = {}
                        temp_data["threads"] = t
                        temp_data["N"] = n2
                        temp_data["N2"] = int(n2)
                        temp_data["tr"] = int(t)
                        temp_data["speedup"] = 0
                        temp_data["implementation"] = filename
                        temp_data["mean"] = d["timing"][0]
                        temp_data["min"] = d["timing"][1]
                        temp_data["max"] = d["timing"][2]
                        temp_data["deviation"] = d["timing"][4]
                        data.append(temp_data)

    df = pd.DataFrame(data)
    baseline_df = df.loc[df["implementation"] == "serial_base"]
    for imp in implementations:
        if imp in cuda_implementations:
            for nn in N2_c:
                m = df.loc[
                    (df["implementation"] == imp) & (df["N"] == nn), "mean"
                ].iloc[0]
                base_m = baseline_df.loc[(baseline_df["N"] == nn), "mean"].iloc[0]
                df.loc[(df["implementation"] == imp) & (df["N"] == nn), "speedup"] = (
                    base_m / m
                )
        elif imp in serial_implementations:
            for nn in N2:
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
            lower = y - error
            upper = y + error
            indx = j % 9
            color = colors[indx]
            marker = "o"
            line_style = "-"
            label_name_ar = name.split("_")[1:]
            label_name = " ".join(str(element) for element in label_name_ar)
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
            ax.axhline(y=1, color="0.8", linestyle="--")
        if i > 3:
            ax.set_xlabel("# threads")
        else:
            ax.set_xlabel("N size")
        ax.set_ylabel("x speedup")
        if i != 0:
            ax.legend(loc="upper right")
        plt.title(title_list[i])
        # plt.yticks(np.arange(-12, 12, 2))
        # plt.ylim((-4,4))

        file_path = plot_path + filename + ".png"
        plt.savefig(file_path)


p = os.path.join("plot", "adi.json")
with open(p) as f:
    adi_dictionary = json.load(f)
plotting(
    adi_dictionary["directory"],
    adi_dictionary["mpi_implementations"],
    adi_dictionary["cuda_implementations"],
    adi_dictionary["serial_implementations"],
    adi_dictionary["open_implementations"],
    adi_dictionary["threads"],
    adi_dictionary["N2"],
    adi_dictionary["N2_c"],
    adi_dictionary["filename_list"],
    adi_dictionary["title_list"],
    adi_dictionary["plot_path"],
    adi_dictionary["set_threads"],
    adi_dictionary["set_n2"],
    False,
)
p = os.path.join("plot", "gemver.json")
with open(p) as f:
    gem_dictionary = json.load(f)
plotting(
    gem_dictionary["directory"],
    gem_dictionary["mpi_implementations"],
    gem_dictionary["cuda_implementations"],
    gem_dictionary["serial_implementations"],
    gem_dictionary["open_implementations"],
    gem_dictionary["threads"],
    gem_dictionary["N2"],
    gem_dictionary["N2_c"],
    gem_dictionary["filename_list"],
    gem_dictionary["title_list"],
    gem_dictionary["plot_path"],
    gem_dictionary["set_threads"],
    gem_dictionary["set_n2"],
)
