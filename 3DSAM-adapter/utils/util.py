from scipy import ndimage
import numpy as np
from medpy import metric
import logging
import os
import time
import torch


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime
