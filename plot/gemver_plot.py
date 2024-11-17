import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

directory = "../results/gemver"
implementations = [ 'mpi_non_block']
threads = [ '2','4','6','8']
data = []

for filename in os.listdir(directory):
    if "truth" not in filename:
        dir = os.path.join(directory, filename)
        for single_file in os.listdir(dir):
            if "latest" in single_file:
                file_path = os.path.join(dir, single_file)
                t = single_file.split('.')[0][-1]
                n = single_file.split('_')[1]
                n2 = n[1:]
                with open(file_path) as f:
                    d = json.load(f)
                    temp_data = {}
                    temp_data['threads'] = t
                    temp_data['N'] = n2
                    temp_data['speedup'] = 0
                    temp_data['implementation'] = filename
                    temp_data['mean'] = d['timing'][0]
                    temp_data['deviation'] = d['deviations'][0]
                    data.append(temp_data)

df = pd.DataFrame(data)

baselinedata = df.loc[df['implementation'] == 'serial_base']

for imp in implementations:
    for t in threads:
        m = df.loc[(df['implementation'] == imp) & (df['threads'] == t), 'mean'].iloc[0]
        base_m = baselinedata.loc[(baselinedata['threads'] == t), 'mean'].iloc[0]
        df.loc[(df['implementation'] == imp) & (df['threads'] == t), 'speedup'] = base_m /m 

sns.set_theme()
plot = sns.lineplot(
    data=df, x="threads", y="speedup", hue="implementation", errorbar="sd", estimator=np.median
).set(title='Gemver Speedups')

file_path = "a.png"
plt.savefig(file_path)
plt.show()
