#!/usr/bin/env python 
from typing import List, Dict, Any
import subprocess
import fchic # 可以直接读取log信息
from ase.data import chemical_symbols, atomic_numbers
import numpy as np
import logging
import json
import copy
from pathlib import Path
import argparse
import pandas as pd
import sys
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import geometry_2_input, smi_2_geom, slurm, orca_2_gau 
from ase.data import chemical_symbols
from glob import glob
import lmdb
import pickle
import traceback

