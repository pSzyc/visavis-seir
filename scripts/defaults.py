# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

    
import os

PARAMETERS_DEFAULT = {
  "e_subcompartments_count": 4,
  "i_subcompartments_count": 2,
  "r_subcompartments_count": 4,
  "c_rate": 1,
  "e_forward_rate": 1,
  "i_forward_rate": 1,
  "r_forward_rate": 0.0667,
}

#TEMP_DIR = f"/run/user/{os.getuid()}"
TEMP_DIR = "/tmp"
