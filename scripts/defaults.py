import os

PARAMETERS_DEFAULT = {
  "c_rate": 1,
  "e_incr": 1,
  "i_incr": 1,
  "r_incr": 0.0667,
}

MOL_STATES_DEFAULT = {
  "n_e": 4,
  "n_i": 2,
  "n_r": 4,
  }

# TEMP_DIR = f"/mnt/export/{os.environ['USER'] if 'USER' in os.environ else 'blabla'}/tmp" #f"/run/user/{os.environ['USER']}"
# TEMP_DIR = f"/run/user/{os.environ['USER']if 'USER' in os.environ else 'blabla'}"
TEMP_DIR = "/tmp"

