#!/usr/bin/env python
"""
Created on Mon Jan 21 21:52:03 2019

@author: Mary Miedema
"""

import os
import time
import argparse
import yaml
#import mne
import logging.config
#import warnings
#import sys
import map_reliability_support.set_up_replications as setrep
import map_reliability_support.likelihood_calc as likelihood


logger = logging.getLogger(__name__)
#mne.set_log_level("WARNING")

#warnings.simplefilter("ignore", category=DeprecationWarning)
#warnings.simplefilter("ignore", category=RuntimeWarning)

def main(configFile):
    """Top-level run script for mapping the reliability of MEG data."""
    try:
        logger.info("Opening configuration file ...")
        with open(configFile, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        logger.debug("Configuration file successfully loaded.")
    except IOError:
        logger.error("Configuration file not found.")

    logger.info("******Starting reliability mapping******")
    
    logger.info("Creating paths for output data ...")
    for section in cfg:
        for cat in cfg[section]:
            if cat == "dirs":
                dir_path = cfg[section][cat].values()[0]
                try:
                    if not os.path.exists(dir_path):
                        logger.debug("Creating {0}".format(dir_path))
                        os.makedirs(dir_path)
                except OSError as e:
                    if not os.path.isdir(dir_path):
                        raise e
            else:
                pass
            
    start_time = time.time()
    logger.info("Setting up replications & calculating nearest neighbours.")
    setrep.set_up(cfg)
    logger.info("Set-up completed.")
    logger.info("Starting maximum likelihood calculations.")
    #likelihood.calc(cfg)
    #likelihood.initialize()
    #likelihood.ICM()
    #likelihood.calc_reliability()
    #likelihood.map_reliability()
    logger.info("Maximum likelihood calculations completed.")
    logger.info("Generating reliability maps.")
    #rmap.mapdata(cfg)
    logger.info("Reliability maps created.")
    end_time = time.time()
    logger.info("TOTAL TIME = {0:.4f} seconds".format(end_time - start_time))
    logger.info("******Reliability mapping completed******")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg",help="Input name of configuration .yml file to use; defaults to config.yml",
                        default="config.yml")
    args = parser.parse_args()
    main(args.cfg)
