import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
data_path = Path("/home/hombresabio/AI/SpatkinModels/Simulate/Data")
filename = "prop_estim"
total_time = 2000
h0 = 4
h1 = 20
h_every = 2
n_sim = 500
colnames=['time', 'F', 'A','S', 'E1', 'E2', 'I1', 'I2', "R1", "R2"]

nE = 5
nI = 2
nR = 4

rE = 1.4
rI = 1.4
rR = 1/15
r_act = 1

v = 1/(1 / (2 * r_act) + nE / rE) 

def get_data(file):
  print(file)
  data = pd.read_csv(file / "Observables.dat", comment='#', header=0, names=colnames, delim_whitespace=True)
  data = data[data['time'] >= 125]
  data['R'] = data['R1'] + data['R2']
  data['I'] = data['I1'] + data['I2']
  data['E'] = data['E1'] + data['E2']
  data.drop(['R1','R2','E1','E2','I1','I2'], axis=1, inplace=True)
  return data

def get_one_stable(height):
    return height * nR / rR * v 

def ext_counter(data, total_time):
    if len(data[data['R']==0]) > 0:
        return data[data['R']==0].iloc[0]['time']
    elif data['time'].max() < total_time:
        return data[data['R']!=0].iloc[-1]['time']
    else:
        return np.nan

def chaos_counter(data, one_stable):
    if len(data[data['R']>=one_stable * 1.6]) > 0:
        return data[data['R']>= one_stable * 1.6].iloc[0]['time']
    else:
        return np.nan
    
def l_value(n_event, mean_t,n,T):
    return n_event / (n_event * mean_t + (n - n_event) * T)


def get_propensities(height, n_sim = n_sim, total_time = total_time):
    
    data_list = []
    for i in range(n_sim):
        data_list.append(get_data(data_path / filename / f'out{height}-{i}'))
    
    treshold = get_one_stable(height)
    T = total_time

    series_chaos = pd.Series(chaos_counter(data_list[i], treshold) for i in range(n_sim))
    series_ext = pd.Series(ext_counter(data_list[i], T) for i in range(n_sim))
    df = pd.DataFrame({'Chaos': series_chaos, 'Ext': series_ext})
    df['min'] = df.min(axis=1)  #.apply(lambda x: min(x['Chaos'], x['Ext']), axis=1)
    mean_t = df['min'].mean()
    l = l_value(len(df['min'].dropna()), mean_t, n_sim, T)
    events = df.idxmin(axis=1).value_counts(dropna=True).reindex(['Chaos', 'Ext']).fillna(0)
    l_ext = l * events['Ext'] / events.sum()
    l_chaos = l * events['Chaos'] / events.sum()

    return [l, l_ext, l_chaos, height]

prop_list = []
for height in range(h0, h1, h_every):
    prop_list.append(get_propensities(height, n_sim))
df = pd.DataFrame(prop_list, columns=['Lambda', 'Lambda Ext', 'Lambda Chaos', 'height'])

df.set_index('height', inplace=True)
df.plot()
ax = plt.gca()

plt.xlabel("Szerokość kanału (ilość komórek)")
plt.ylabel("Propensity zdarzenia")
plt.legend(['Porażka propagacji (Wygaśnięcie lub Chaos)', "Wygaśnięcie sygnału", "Wytworzenie się chaosu" ])
plt.savefig("propensities_result.svg")
#plt.savefig("propensities_result.png")

