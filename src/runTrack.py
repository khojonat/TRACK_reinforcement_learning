# -*- coding: utf-8 -*-
"""
Created on Thr Feb 18 10:17:25 2021
@author: Jose L. Martinez-Marin
@Coauther: Anthony Tran
"""
import os
from time import time
import subprocess
from multiprocessing import Process
import shutil
import platform

def runTrack(track_exe, track_directory, **sub_kwargs): # run TRACK using subprocess.run()
    """
    Change directory to the input files since Track code run in its own directory, 
    then return to current directory. 
    Code also return runtime.
    Parameters
    ----------
    track_exe : path
        file name of track
    track_directory : path
        path to the Track file

    Returns
    -------
    completed
        if it completed or not
    runtime
        time it took to run
    """
    oDir = os.getcwd() # return the current operating directory
    os.chdir(track_directory) # This way the Track code sees the input files when it runs
    # Track code will output its files to this directory, as if self._inPath == self._outPath

    start = time()
    completed = subprocess.run(str(os.path.join(track_directory, track_exe)), **sub_kwargs) # cast as str bc subprocess doesn't support PathLike in python3.7.6
    runtime = time() - start

    os.chdir(oDir) # restore the original directory
    return completed, runtime

def copy2folder(inputfolder, outputfolder, name):
    """
    Copy content of the inputfilder, into the output folder.
    Creates a new folder in the output folder and give custom name.
    """
    # First check if directory is present, then creates one.
    newfolder = str(outputfolder) + "/"+ name
    if not os.path.isdir(newfolder):
        os.mkdir(newfolder)
    else:
        print("Folder already exist! Folder " + str(name) + " not created.")
        
    for file in os.listdir(inputfolder):
        # construct full file path
        source = str(inputfolder) +"/"+ file
        destination = str(newfolder) +"/"+ file
        shutil.copy(source, destination)
    osname = platform.system()
    if osname == 'Linux':
        os.chmod(newfolder+'/track', 0o750)
        
def rmfolder(path, folder):
    """
    Delete folder. 
    """
    dirPath = str(path) + '/' + folder
    if not os.path.exists(dirPath):
        print(f"Folder {dirPath} don't exist...")
        return
    try:
        shutil.rmtree(dirPath)
    except:
        print('Error while deleting directory')
    return

if __name__ == "runTrack":
    print("Loading runTrack...")
