import os
import sys

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
sys.path.append(os.path.join(os.path.abspath(''), submodule_name))