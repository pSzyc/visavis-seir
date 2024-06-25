This code features the research article "Information transmission in a cell monolayer: 
A numerical study" by Paweł Nałęcz-Jawecki et al., 2024, and allows to reproduce images and
movies from the article.


# Cloning

git clone --recurse-submodules https://github.com/pSzyc/visavis-seir


# Installation

1. Install python>=3.10 https://www.python.org/downloads.
2. Install rust>=1.78 https://www.rust-lang.org/tools/install.
3. Create and activate python environment (optional, but highly recommended):

[Windows cmd]:
python -m venv %userprofile%\.venv\qeirq
%userprofile%\.venv\qeirq\Scripts\activate.bat

[Windows PowerShell] -- script execution must be enabled:
python -m venv $Home\qeirq
& "$Home\qeirq\Scripts\activate.ps1"

[Linux]:
python -m venv ~/.venv/qeirq
source ~/.venv/bin/activate

4. Enter the cloned repository and install depencencies:
pip install -r requirements.txt


# Running

To run a sample simulation, enter the main repository directory and run
python run_simulation.py
You can edit the file to change simulation parameters.

