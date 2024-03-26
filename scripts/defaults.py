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


# TEMP_DIR = f"/mnt/export/{os.environ['USER'] if 'USER' in os.environ else 'blabla'}/tmp" #f"/run/user/{os.environ['USER']}"
TEMP_DIR = f"/run/user/{os.environ['USER'] if 'USER' in os.environ else 'blabla'}"
# TEMP_DIR = "/tmp"

