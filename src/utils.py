# -*- coding: utf-8 -*-
#import numpy as np
#import math
#import os
import pandas as pd

if __name__ == "utils":
    print("Loading utils...")
    
def make_sclinac(file, sclinac, mode='w'):
    """
    Makes a sclinac.dat file given sclinac config file
    
    input:
    file: str - file path ending in /sclinac.dat
    sclinac: a python list of lists were each list is a beam element configuration
    """
    with open(file, mode) as f:
        for elem in sclinac:
            for i in elem:
                f.write(f"{i} ")
            f.write("\n")
        f.write(f"0 stop ")
    return

def make_track(f, track_dat):
    """
    Makes a track.dat file given sclinac config file
    
    input:
    file: str - file path ending in /track.dat
    track_dat: a python dictionary containing track.dat settings
    """
    with open(f, 'w') as f:
        f.write('&TRAN\n')
        for elem in track_dat:
            f.write(f"  {elem}={track_dat[elem]}")
            f.write("\n")
            if elem=='dww00':
                f.write('&END\n')
                f.write('&INDEX\n')
            if elem=='iwrite_step':
                f.write('&END\n')
                f.write('&MATCH\n')
            if elem=='fit_err':
                f.write('&END')
    return

def dis2dataframe(file):
    """
    extract the distribution from a read_dis.out file and put it into a pandas dataframe.
    Input: path to a read_dis file
    Output: Pandas DataFrame with all the coordinates ["x","px","y","py","phi","dww","l"]
    """
    with open(file) as f:
        lines = f.readlines()
    inputs = []
    for line in lines:
        inputs.append(line.strip().split())
    coord = pd.DataFrame(inputs[3:], columns=["x","px","y","py","phi","dww","l"])
    coord = coord.apply(pd.to_numeric)  # make string into numeric
    return coord