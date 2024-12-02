from typing import Union

import seutil as su

def save_setup_config(save_file: su.arg.RPath, **kwargs):
    """Log the setup config."""
    setup_config = {}
    setup_config.update(kwargs)
    su.io.dump(save_file, setup_config)