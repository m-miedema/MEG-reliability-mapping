#!/usr/bin/env python
"""
Set up ipython environment for ML testing

@author: Mary Miedema
"""


import yaml
import numpy as np
import map_reliability_support.likelihood_biden as lhood

with open('cfgbeta3.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#testML = lhood.ML_reliability(cfg,'sub03','one_jitter',55) 
testML = lhood.ML_reliability(cfg,'sub03','SEF_good',20)  

testML.calc_reliability_measures()

testML.map_reliability(0.9)
testML.map_reliability(0.6)


# wishlist: automatically copy cfg file to nifti output directory for settings reference
