import os

PARAMETERS_DEFAULT = {
  "c_rate": 1,
  "e_incr": 1,
  "i_incr": 1,
  "r_incr": 0.0667
}

TEMP_DIR = f"/mnt/export/{os.environ['USER']}/tmp" #f"/run/user/{os.environ['USER']}"

