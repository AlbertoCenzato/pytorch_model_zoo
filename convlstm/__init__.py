try:
    import torch
    import convlstmcpp
except ModuleNotFoundError as err:
    import os
    import subprocess
    dir_path = os.path.dirname(os.path.realpath(__file__))
    completed_process = subprocess.run("python setup.py install", cwd=os.path.join(dir_path, 'src'))

from .package.convlstm import ConvLSTM, ConvLSTMCell, HiddenState, HiddenStateStacked
from .package.convlstm_cpp import ConvLSTMCPPCell
from .package.convlstm_ch_pooling import ConvLSTMChannelPooling
