"""Unit test package for omicron."""
import os

import cfg4py


def init_test_env():
    os.environ[cfg4py.envar] = 'TEST'
    src_dir = os.path.dirname(__file__)
    config_path = os.path.join(src_dir, '../omicron/config')

    return cfg4py.init(config_path)
