#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module loads an MVNX file and extracts its data in tabular format for the
specified fields.
"""


from pathlib import Path
import os
# for OmegaConf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf, MISSING
#
from emokine.mvnx import Mvnx, MvnxToTabular


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar MVNX_PATH: Path to the MVNX file to be converted.
    :cvar OUT_DIR: Path to directory where results are saved.
    :cvar SCHEMA_PATH: Optional path to an MVNX schema to be tested against.
    :cvar FIELDS: List of parameters to be extracted like position, speed...
      see complete list in ``MvnxToTabular.ALLOWED_FIELDS``. If none given,
      all fields will be extracted
    """
    MVNX_PATH: str = MISSING
    OUT_DIR: str = MISSING
    SCHEMA_PATH: Optional[str] = None
    FIELDS: Optional[List[str]] = None


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    # load MVNX + extract tabular data for selected fields (all fields if None)
    m = Mvnx(CONF.MVNX_PATH, CONF.SCHEMA_PATH)
    processor = MvnxToTabular(m)
    dataframes = processor(CONF.FIELDS)

    # create out_dir if doesn't exist and save results
    Path(CONF.OUT_DIR).mkdir(parents=True, exist_ok=True)
    for df_name, df in dataframes.items():
        outpath = os.path.join(CONF.OUT_DIR, df_name + ".csv")
        df.to_csv(outpath, index=False)
        print("saved dataframe to", outpath)
