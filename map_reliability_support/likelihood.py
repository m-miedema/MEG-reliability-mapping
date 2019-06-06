# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 03:54:41 2019

@author: Mary Miedema
"""

import os
#import mne
import time
import sys
import copy
import nibabel as nib
import numpy as np
import constrNMPy as cNM
import scipy as scp
from scipy.optimize import minimize
#from scipy.optimize import basinhopping
from scipy.integrate import quad
import pandas as pd
import logging.config

logger = logging.getLogger(__name__)


class ML_reliability:
    def __init__(self, cfg, subject_id, sim_mode, timept):
        # note: timepoint needs to be chosen based on seconds after starting time point of ERB file -- integrate better with previous ERB crop
        
        self.cfg = cfg
        self.subject_id = subject_id
        self.sim_mode = sim_mode
        self.timept = timept
        
        # unpack settings from configuration file
        self.num_replic = cfg["reliability_mapping"]["data_division"]["num_replications"]
        self.num_thresh = cfg["reliability_mapping"]["ML"]["num_thresholds"]
        self.beta0 = cfg["reliability_mapping"]["ML"]["init_beta"]
        self.max_iter = cfg["reliability_mapping"]["ML"]["max_iterations"]
        self.tolerance = cfg["reliability_mapping"]["ML"]["conv_tolerance"]
        
        self.relDir = cfg["file_io"]["dirs"]["relDir"]
        self.replicDir = os.path.join(self.relDir, subject_id, sim_mode)
        self.erbFile = os.path.join(self.replicDir,"".join(['full', cfg["file_io"]["file_ext"]["nifti_file"]]))         
        self.nnDir = os.path.join(cfg["file_io"]["dirs"]["subjectsDir"], subject_id, 'bem')
        self.nnFile = os.path.join(self.nnDir, "".join([subject_id, cfg["file_io"]["file_ext"]["nn_file"]]))
        
        # initialize other variables
        self.erb = None 
        self.erb_trans = None
        self.erb_header = None
        self.vox_in_use = None
        self.thresh = None
        self.y = None
        
        self.lambda0 = None
        self.p_a0 = None
        self.p_i0 = None
        
        self.lambd = None
        self.p_a = None
        self.p_i = None
        self.beta = self.beta0
        
        self.lambda_conv = None
        self.p_a_conv = None
        self.p_i_conv = None
        self.beta_conv = 0
        self.conv = 1000.

        self.p_eval = 0
        self.beta_eval = 0
        self.lambda_eval = 0
        
        self.rel_thresh = None # needs to be a value in self.thresh
        self.active_voxels = None
        self.reliability = None
        self.antireliability = None
        self.reliability_map = None
        
        

	# find thresholds
        self.get_thresholds()

	# NEW: generate NNs locally

        #self.gen_vox_nn()

        self.nn_df = self.gen_vox_nn() #pd.read_csv(self.nnFile) 

        
        # find number of replications each voxel is classified as active in for each threshold
        self.calc_y()
        
        # calculate an initial guess for lambda
        self.guess_lambda()
        
        # calculate an initial guess for the p_a and p_i for each threshold based on Genovese et al's findings
        self.guess_p()
        
        
    def get_thresholds(self):
        # read in ERB file
        erb_obj = nib.load(self.erbFile)
        # get data at relevant timepoint
        self.erb = erb_obj.get_fdata()[:,:,:,self.timept]
        self.erb_trans = erb_obj.affine
        self.erb_header = erb_obj.header.copy()
        self.erb_header['dim'][4] = 1 # only want a header for a single timept
        # find maximum ERB value
        erb_max = np.amax(self.erb)
        # get voxels in use
        self.vox_in_use = (np.abs(self.erb.flatten()) > 0)
        # create equally spaced thresholds ranging from 0-100% of max
        thresholds = np.linspace(0.,erb_max,num=(self.num_thresh+1))
        self.thresh = thresholds
        
    def calc_y(self):
        y_data = []
        # first read in the replications
        for split_num in range(1,self.num_replic+1):
            # load ERB data for each 'replication' at specified timepoint
            replicFile = os.path.join(self.replicDir, "".join(['replication_', 
                                str(split_num), '_', self.cfg["file_io"]["file_ext"]["nifti_file"]]))
            replic_obj = nib.load(replicFile)
            replic_data = replic_obj.get_fdata()[:,:,:,self.timept]
            # flatten ERB data and check whether active for each threshold
            y_obj = 1.* (np.broadcast_to(replic_data.flatten(), 
                        (self.num_thresh+1,replic_data.size)) > self.thresh[:, np.newaxis])
            y_data.append(y_obj)
        # sum across replications    
        self.y = sum(y_data)
        
        
    def guess_lambda(self):
        # initial guess for probability a voxel is truly active: ratio to maximum activity (full ERB map)
        self.lambda0 = (self.erb.flatten())/np.amax(self.erb)
        self.lambd = copy.deepcopy(self.lambda0) 
        
        
    def guess_p(self):
        # not quite sure if there is a simple data-driven way to do this
        # referring to Genovese et al. Table 3 as a rough guide for now:
        p = np.linspace(0.9,0.1,self.num_thresh)
        self.p_a0 = p
        self.p_i0 = np.exp(-1/p) 
        self.p_a = copy.deepcopy(self.p_a0) 
        self.p_i = copy.deepcopy(self.p_i0) 
        self.p_a_conv = np.ones(self.num_thresh)  
        self.p_i_conv = np.ones(self.num_thresh)      
        
        
    def estimate_p(self):
        
        # will estimate p_a and p_i for each threshold quasi-independently, since lower thresholds should not depend on higher thresholds
        for thr_k in range(0,self.num_thresh):

            # set up adaptive constraints so that p is positive and strictly decreasing with threshold:
            if thr_k == 0:
                p_UB = [1.,1.]
            else:
                p_UB = [self.p_a[thr_k-1],self.p_i[thr_k-1]]

            p_guess = [self.p_a[thr_k],self.p_i[thr_k]]

            # make sure initial guess for p is within constraints
            if p_guess[0] > p_UB[0]:
                p_guess[0] = 0.4*p_UB[0]
            if p_guess[1] > p_UB[1]:
                p_guess[1] = 0.4*p_UB[1]

            # use constrained Nelder-Mead for estimation
            p_opt_res = cNM.constrNM(self.p_func,p_guess,[0.,0.],p_UB,full_output=True,args=[thr_k])
            p_est = p_opt_res['xopt']
            
            # update guesses for p_a & p_i
            self.p_a[thr_k] = p_est[0]
            self.p_i[thr_k] = p_est[1]

            # calculate difference from previous guesses
            self.p_a_conv[thr_k] = np.abs(p_est[0] - p_guess[0])
            self.p_i_conv[thr_k] = np.abs(p_est[1] - p_guess[1])

    def estimate_lambda(self):
        # ICM estimation process for lambda: lambda only depends on NNs
        lambda_prev = copy.deepcopy(self.lambd)
        for vox in range(0,self.lambd.size):
            # only perform estimation if voxel in use:
            if self.vox_in_use[vox] == True:
                # now maximize local likelihood with constrained Nelder-Mead
                lambda_guess = [self.lambd[vox]]
                vox_opt_res = cNM.constrNM(self.lambda_func,lambda_guess,[0.],[1.],args=[vox])
                self.lambd[vox] = vox_opt_res['xopt']
        # output estimation progress to terminal
		lambda_msg = "Estimation complete for voxel %i of %i" % (vox, self.lambd.size)
                sys.stdout.write(lambda_msg + chr(8)*len(lambda_msg))
                sys.stdout.flush()
                if vox_opt_res['warnflag'] is not None:
                    sys.stdout.write("\n")
                    print('Potential issue with lambda optimization')
                    print(vox_opt_res)
                    print(vox)
                    sys.stdout.write("\n")
        sys.stdout.write("\n")
        # calculate change in lambda for each voxel
        self.lambda_conv = np.abs(self.lambd - lambda_prev) 
        
    def estimate_beta(self):
        # this function not currently in use; beta is held constant
        beta_guess = self.beta
        beta_opt_res = minimize(self.beta_func,beta_guess,method='Nelder-Mead',tol=0.0001,callback=self.callback_beta)
        beta_est = beta_opt_res.x
        
        print(beta_opt_res.success)
        print(beta_opt_res.nit)
        print(beta_est)
        
        # calculate difference from previous guesses
        self.beta_conv = np.abs(beta_est - beta_guess)
        # update guess for beta
        self.beta = beta_est

    def calc_convergence(self):
        # returns the largest change in all estimated variables this iteration
        return max([max(self.p_a_conv), max(self.p_i_conv), max(self.lambda_conv), self.beta_conv])
        
    def calc_reliability_measures(self):
        # iteratively estimate p_a, p_i, lambda and (optionally) beta
        start = time.time()
        num_iter = 1
        conv = self.conv
        while conv > self.tolerance and num_iter < self.max_iter:
            print "ICM Iteration: ", num_iter
            print('Beginning p estimation')
            print "Elapsed time:", time.time()-start
            self.estimate_p()
            print('Finished estimating p!')
            print(self.p_a)
            print(self.p_i)
            print('Beginning lambda estimation')
            print "Elapsed time:", time.time()-start
            #self.estimate_beta()
            self.estimate_lambda()
            logger.info("ICM estimation in progress; iteration {}".format(num_iter))
            conv = self.calc_convergence()
            print "Convergence:",conv
            num_iter += 1
	print "Finished calculation with ",num_iter," iterations"
        print "Total elapsed time:", time.time()-start
        logger.info("Reliability measure estimation completed; convergence within {}".format(conv))
        
    def map_reliability(self,t):
        # in the future, could have an option to output reliability map for each threshold instead
        # for now choose the first threshold equal to or higher than t
        t_thresh = np.argmax(self.thresh>=(t*np.amax(self.erb)))
        self.rel_thresh = self.thresh[t_thresh]
        logger.info("Reliability map generated for threshold set to {} of maximum activity".format(self.rel_thresh))
        print "Threshold for reliability maps:", self.rel_thresh
        # to do: add error message here if t not within appropriate range
        
        # find all active voxels in original ERB map        
        self.active_voxels = np.where(self.erb.flatten() >= self.rel_thresh, True, False)
        # find remaining inactive voxels
        self.inactive_voxels = np.where(0. < self.erb.flatten(), True, False)*np.where(self.erb.flatten() < self.rel_thresh, True, False)
        
        # calculate unconditional probabilities pi
        pi_a = 1
        pi_i = 1
        for k in range(0,t_thresh):
            pi_a *= self.p_a[k]
            pi_i *= self.p_i[k]

        # calculate reliability for active voxels
        rel = np.divide(pi_a*self.lambd,(pi_a*self.lambd + pi_i*(1.-self.lambd)))
        self.reliability = np.multiply(self.active_voxels, rel)
        # calculate antireliability for inactive voxels
        antirel = np.divide(((1.-pi_i)*(1.-self.lambd)),((1.-pi_a)*self.lambd + (1.-pi_i)*(1.-self.lambd)))
        self.antireliability = np.multiply(self.inactive_voxels, antirel)
        
        # use nibabel to save as nifti
        ni_rel_map = nib.Nifti1Image(np.reshape(self.reliability,self.erb.shape),None,header=self.erb_header)
        ni_antirel_map = nib.Nifti1Image(np.reshape(self.antireliability,self.erb.shape),None,header=self.erb_header)
        ni_rel_file = str(round(self.rel_thresh,4)*100)+'_rel_map.nii'
        ni_antirel_file = str(round(self.rel_thresh,4)*100)+'_antirel_map.nii'
        nib.save(ni_rel_map,os.path.join(self.replicDir,str(self.beta),ni_rel_file))
        nib.save(ni_antirel_map,os.path.join(self.replicDir,str(self.beta),ni_antirel_file))

        # to do: add arguments to toggle optional outputs
        # also save lambda and self.active_voxels for inspection
        lambda_map = nib.Nifti1Image(np.reshape(self.lambd,self.erb.shape),None,header=self.erb_header)
        nib.save(lambda_map,os.path.join(self.replicDir,'lambda_map.nii'))
        active_map = nib.Nifti1Image(np.reshape(1.*self.active_voxels,self.erb.shape),None,header=self.erb_header)
        nib.save(active_map,os.path.join(self.replicDir,'active_vox.nii'))

    def vox_nn(self,vox):
        # returns indices of nearest neighbours of vox  
        this_nn_df = self.nn_df.copy()
        vox_nn_df = this_nn_df[this_nn_df['src_index']==vox]
        return vox_nn_df['NN'].tolist()


    def gen_vox_nn(self):  
        # revised method of finding NNs; functionally equivalent to reading in previously generated NN file
                # should change this to a save file rather than calling each time a likelihood object is generated
        nn_dfs = [] 

        for vox in range(0,self.erb.size):
            # only find NNs if voxel in use:
            if self.vox_in_use[vox] == True:
                vox_unravel = np.unravel_index(vox, self.erb.shape)
                #print(vox_unravel)
                vox_list = []
                
                for i0 in range(vox_unravel[0]-1,vox_unravel[0]+2):
                    for i1 in range(vox_unravel[1]-1,vox_unravel[1]+2):
                        for i2 in range(vox_unravel[2]-1,vox_unravel[2]+2):
                            if i0 != vox_unravel[0] or i1 != vox_unravel[1] or i2 != vox_unravel[2]:
                                vox_i_ravel = np.ravel_multi_index((i0,i1,i2),self.erb.shape)
                                vox_list.append(vox_i_ravel)

            # check which ones are active
                NN_vox_list = list(set(self.vox_in_use).intersection(vox_list))
            # put results into dataframe
                vox_df = pd.DataFrame({'src_index':vox,'NN':NN_vox_list})
                nn_dfs.append(vox_df)

        nn_df = pd.concat(nn_dfs)
        #nn_df.to_csv(self.nnFile)
        return nn_df

             

######################### callback functions ##################################
        # can be used to monitor minimization progress

    def callback_p(self,p_guess):
        self.p_eval += 1
        print 'Callback', self.p_eval
        print(p_guess)

    def callback_beta(self,beta_guess):
        self.beta_eval += 1
        print(self.beta_eval)
        print(beta_guess)
        
######################### likelihood functions ################################
        # see Maitra et al, Eqs. 1, 4, 5, & 6
        
    def p_func(self,p,max_thresh):
        # Eq. 1
        log_l = 0.
        for vox in range(0,self.lambd.size):
            if self.vox_in_use[vox] == True:
                lfunc1 = self.lambd[vox]
                lfunc2 = 1. - self.lambd[vox]
                for thresh_k in range(0, max_thresh+1):

                    if thresh_k == max_thresh:
                        p_a = p[0]
                        p_i = p[1]
                    else:
                        p_a = self.p_a[thresh_k]
                        p_i = self.p_i[thresh_k]


                    lfunc1 *= scp.special.comb(self.y[thresh_k,vox],
                                self.y[thresh_k+1,vox])*(p_a**self.y[thresh_k+1,vox])*((1.-p_a)**(self.y[thresh_k,vox]-self.y[thresh_k+1,vox]))
                    lfunc2 *= scp.special.comb(self.y[thresh_k,vox],
                                self.y[thresh_k+1,vox])*(p_i**self.y[thresh_k+1,vox])*((1.-p_i)**(self.y[thresh_k,vox]-self.y[thresh_k+1,vox]))
                log_l += np.log(lfunc1 + lfunc2)
        return -log_l
    
    
    def beta_func(self,beta):
        # Eq. 5
        lfunc1 = 0.
        lfunc2 = 0.
        for vox in range(0,self.lambd.size):
            if self.vox_in_use[vox] == True:
                nn_v = self.vox_nn(vox)
                for n_vox in nn_v:
                    lfunc1 += 1./(1+(self.lambd[vox]-self.lambd[n_vox])**2.)
                integral, err = quad(self.beta_integrand, 0., 1., args = (nn_v, beta))
                lfunc2 += np.log(integral)
        lfunc1 *= beta                    
        log_pseud_l = lfunc1 - lfunc2
        return -log_pseud_l
    
    def beta_integrand(self, lambda_i, nn_v, beta):
        # Eq. 6 (modified as per email discussion)
        lambda_sum = 0
        for n_vox in nn_v:
            lambda_sum += 1./(1+(lambda_i - self.lambd[n_vox])**2.)
        return np.exp(beta*lambda_sum)
    
    def lambda_func(self,lambd_vox,vox):
        # taking the log likelihood (log of Eq. 1) for vox and its nearest neighbours
        log_l = 0.
        if self.vox_in_use[vox] == True:
            lfunc1 = lambd_vox
            lfunc2 = 1. - lambd_vox
            for thresh_k in range(0, self.num_thresh):
                lfunc1 *= scp.special.comb(self.y[thresh_k,vox],
                            self.y[thresh_k+1,vox])*(self.p_a[thresh_k]**self.y[thresh_k+1,vox])*((1.-self.p_a[thresh_k])**(self.y[thresh_k,vox]-self.y[thresh_k+1,vox]))
                lfunc2 *= scp.special.comb(self.y[thresh_k,vox],
                            self.y[thresh_k+1,vox])*(self.p_i[thresh_k]**self.y[thresh_k+1,vox])*((1.-self.p_i[thresh_k])**(self.y[thresh_k,vox]-self.y[thresh_k+1,vox]))
            log_l += np.log(lfunc1 + lfunc2)
            # penalty term (Eq. 4):
            betafunc = 0.
            nn_v = self.vox_nn(vox)
            for n_vox in nn_v:
                betafunc += 1./(1+(lambd_vox-self.lambd[n_vox])**2.)
            log_l += self.beta*betafunc
        return -log_l

