# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:43:43 2019

@author: Mary Miedema
"""

from scipy import spatial
from random import randint
import os
import mne
import pandas as pd
import numpy as np
import nibabel as nib
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
        
def sim_replications(cfg):
        base_subject = cfg["simTestData"]["base_subject"]
        print(base_subject)
        base_mode = cfg["simTestData"]["base_mode"]
        sim_subject = cfg["simTestData"]["sim_subject"]
        sim_mode = cfg["simTestData"]["sim_mode"]
        relDir = cfg["file_io"]["dirs"]["relDir"]
        replicDir = os.path.join(relDir, base_subject[0], base_mode[0])
        simDir = os.path.join(relDir, sim_subject[0], sim_mode[0])
        num_replic = cfg["reliability_mapping"]["data_division"]["num_replications"]
        
        
        
        # first read in the orignial replications
        
        for split_num in range(1,num_replic+1):
            # load ERB data for each 'replication'
            replicFile = os.path.join(replicDir, "".join(['replication_', 
                                str(split_num), '_', cfg["file_io"]["file_ext"]["nifti_file"]]))
            replic_obj = nib.load(replicFile)
            replic_data = replic_obj.get_fdata()#[:,:,:,self.timept]
            sim_data = []
            for ts in range(0,replic_data.shape[3]):
                # generate noise
                base_dist = replic_data[:,:,:,ts]
                stdev = np.std(base_dist)
                mean = np.mean(base_dist)
                noise_dist = np.abs(np.random.normal(loc=mean,scale=stdev,size=replic_data.shape[0:-1]))
                # add a source
                amp = np.amax(base_dist)
                pos = [8+randint(-2,3), 16+randint(-4,4), 22 + randint(-5, 5)]
                stdev = 5
                source_dist = Gaussian_amp(0.5*noise_dist, amp, pos, stdev) 
                # set voxels outside the head to zero
                active = np.where(base_dist.flatten()>0., 1., 0.)
                sim_data_ts = np.multiply(source_dist,np.reshape(active,source_dist.shape))
                sim_data.append(sim_data_ts)
            sim_data = np.stack(sim_data,axis=3)
            sim_nii_data = nib.Nifti1Image(sim_data,None,header=replic_obj.header.copy())
            nib.save(sim_nii_data,os.path.join(simDir,"".join(['replication_', 
                                str(split_num), '_', cfg["file_io"]["file_ext"]["nifti_file"]])))


	#also save "full" ERB file
	for i in range(0,1):
            # load ERB data for each 'replication'
            replicFile = os.path.join(replicDir, "".join(['full', cfg["file_io"]["file_ext"]["nifti_file"]]))
            replic_obj = nib.load(replicFile)
            replic_data = replic_obj.get_fdata()#[:,:,:,self.timept]
            sim_data = []
            for ts in range(0,replic_data.shape[3]):
                # generate noise
                base_dist = replic_data[:,:,:,ts]
                stdev = np.std(base_dist)
                mean = np.mean(base_dist)
                noise_dist = np.abs(np.random.normal(loc=mean,scale=stdev,size=replic_data.shape[0:-1]))
                # add a source
                amp = np.amax(base_dist)
                pos = [8+randint(-2,3), 16+randint(-4,4), 22 + randint(-5, 5)]
                stdev = 5
                source_dist = Gaussian_amp(0.5*noise_dist, amp, pos, stdev) 
                # set voxels outside the head to zero
                active = np.where(base_dist.flatten()>0., 1., 0.)
                sim_data_ts = np.multiply(source_dist,np.reshape(active,source_dist.shape))
                sim_data.append(sim_data_ts)
            sim_data = np.stack(sim_data,axis=3)
            sim_nii_data = nib.Nifti1Image(sim_data,None,header=replic_obj.header.copy())
            nib.save(sim_nii_data,os.path.join(simDir,"".join(['full', cfg["file_io"]["file_ext"]["nifti_file"]])))


    
        
def Gaussian_amp(array, amp, pos, stdev):
    # Generates a Gaussian point source in a 3D array based on specified parameters
    
    for x in range(0, array.shape[0]):
        for y in range(0, array.shape[1]):
            for z in range(0, array.shape[2]):
                array[x,y,z] = array[x,y,z] + amp*np.exp(0.5*(-(x-pos[0])**2. - (y-pos[1])**2. - (z-pos[2])**2.)/stdev**2.)
                
    return array
    
    
    
