# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:43:43 2019

@author: Mary Miedema
"""

from scipy import spatial
import os
import mne
import pandas as pd
import numpy as np
import logging.config
import warnings

logger = logging.getLogger(__name__)
mne.set_log_level("WARNING")

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

def get_nn_src(srcFile, nnFile, neighbourhood, radius):
    '''Function to get the indices of the nearest neighbours for each point in a volume source space.
    
         src_file: location of volume source space file
         nn_file: name/location to write NN pandas dataframe to, should be csv file
         neighbourhood: choose extent of neighbourhood n
                      needs to be 1, 2, or False (to do: check this)
                      1: returns directly adjacent voxels (n = 6)
                      2: also includes diagonals (n = 26)
                      False: use radius instead
         radius: only used if neighbourhood not 1 or 2 (to do: optional arg)
    
    '''
    if os.path.exists(srcFile):
        src = mne.read_source_spaces(srcFile, verbose=False)
    else:
        logger.debug("Source file {} not found.".format(srcFile))
    if not os.path.exists(nnFile):
        logging.info("Finding nearest neighbours.")
        # make a place for NN file if necessary
        if not os.path.exists(os.path.dirname(nnFile)):
            os.makedirs(os.path.dirname(nnFile))
            
        # will create a dataframe for results
        nn_dfs = [] 
        
        # FIND NEAREST NEIGHBOURS
        all_pos = src[0].get('rr')
        src_vert = src[0].get('vertno')
        src_tree = spatial.cKDTree(all_pos)
        # define radius of neighbourhood
        spacing = max(all_pos[1]-all_pos[0],key=abs)
        if neighbourhood == 1:
            r = 1.01*spacing
        elif neighbourhood == 2:
            r = 1.01*spacing*(3**0.5)
        else:
            r = radius
        # find NN for each point used in source space
        for pt in src_vert:
            # find neighbours
            all_pt_NN = src_tree.query_ball_point(all_pos[pt],r)
            # discard original point
            all_pt_NN.remove(pt)
            # discard neighbours not in use in source space
            src_pt_NN = list(set(src_vert).intersection(all_pt_NN))
            # put results into dataframe
            pt_df = pd.DataFrame({'src_index':pt,'NN':src_pt_NN})
            # to do: it would be nice to rank neighbours in DF & get rid of neighbourhood/spacing options
            nn_dfs.append(pt_df)
        nn_df = pd.concat(nn_dfs)
        nn_df.to_csv(nnFile)