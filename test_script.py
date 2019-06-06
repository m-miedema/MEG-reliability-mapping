#!/usr/bin/env python
"""
Set up ipython environment for ML testing, run on simulated data

@author: Mary Miedema
"""


import yaml
import numpy as np
import map_reliability_support.likelihood as lhood
import map_reliability_support.support_functions as support

with open('cfg.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#support.sim_replications(cfg)

testML = lhood.ML_reliability(cfg,'simsub','single-source',20)  
testML.calc_reliability_measures()
testML.map_reliability(0.9)
testML.map_reliability(0.6)


# wishlist: automatically copy cfg file to nifti output directory for settings reference
