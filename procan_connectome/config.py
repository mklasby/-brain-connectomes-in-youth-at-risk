# STATIC VARS
import os
import logging 
this_dir = os.path.dirname(__file__)
DATA_PATH = os.path.join(this_dir, 'data')
PLOT_PATH = os.path.join(this_dir, 'plots')
RANDOM_STATE=42
LOGGER_LEVEL = logging.DEBUG