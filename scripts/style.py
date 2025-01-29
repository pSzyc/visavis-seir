# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

    
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ['Carlito']
plt.rcParams['font.size'] = 8
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Carlito'


channel_length_to_color = {
    30: "C0",
    100: "C1",
    300: "C2",
    1000: "C3",
}

