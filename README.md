This code features the research article [_Information transmission in a cell monolayer: 
A numerical study_](https://doi.org/10.1101/2024.06.21.600012) by Paweł Nałęcz-Jawecki _et al._, 2024, and allows to reproduce images and
movies from the article.

# License

    Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Cloning the repository

```bash
git clone --recurse-submodules https://github.com/pSzyc/visavis-seir
```

# Setup

1. Install [python](https://www.python.org/downloads)>=3.10.
2. Install [rust](https://www.rust-lang.org/tools/install)>=1.78.
3. Create and activate python environment (optional, but highly recommended):

[Windows cmd]:
```bat
python -m venv %userprofile%\.venv\qeirq
%userprofile%\.venv\qeirq\Scripts\activate.bat
```


[Windows PowerShell] -- script execution must be enabled:
```
python -m venv $Home\qeirq
& "$Home\qeirq\Scripts\activate.ps1"
```

[Linux]:
```bash
python -m venv ~/.venv/qeirq
source ~/.venv/qeirq/bin/activate
```

4. Enter the cloned repository and install dependencies:
```bash
pip install -r requirements.txt
```

# Running

To run a sample simulation, enter the main repository directory and run
```bash
python run_simulation.py
```
You can edit the file to change simulation parameters. The results will be found in 'results' directory.


# Data and figures

All the figures and videos used in the papers are in the 'figures' and 'videos' folders.
Scripts that were used to create the figures can be found in folders 'figures/fig<X>/code'.
In folder 'data' you will find data that can be used to regenerate the figures.
The figures can regenerated using scripts 'figures/fig<X>/code/fig<X><panel>_plot.py'.
To redraw a plot, run:
```bash
python figures/fig<X>/code/fig<X><panel>_plot.py
```
By running 'figures/fig<X>/code/fig<X><panel>_get_data.py' scripts you can regenerate the data in folder 'data'.
Note that regenerating data for all figures will take ~1000 h of aggregated computational time and ~100 GB of storage.
You can reduce the computational cost by reducing the number of simulations (`n_simulations`) in *_get_data.py files.
Note that the data in folder 'data' will be overwritten with this operation.
