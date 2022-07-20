# -*- coding: utf-8 -*-
from pathlib import Path
import platform
import os

class Config:
  """
    The one place where you would change constant parameters. Should only need
    to adjust this file to do anything. This should also be the file where you 
    would update file paths.
    
    This main idea is that a dictionary object holds the core information of each
    file, such as track.dat, sclinac.dat, hdf5 files,
    then we can adjust the dicionary in python 
    and then recreate the whole file everytime we update the settings.
    """
  def __init__(self):
    # Common file paths and directories. Change the first two if you us a different computer.
    # Make sure all directories are created else you may get error
    self.TRACK_DIRECTORY = Path('C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/TRACK')
    self.PARENT_TRACK = Path("C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/parentTRACK")
    self.CHILD_TRACK = Path("C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/childTRACK")
    self.HDF5_DIR = Path("C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/hdf5")

    self.WORKING_DIRECTORY = Path(os.getcwd())
    
    self.TRACK_EXE = Path("TRACKv39C.exe")  # Don't change
    osname = platform.system()
    if osname == 'Linux':
        self.TRACK_EXE = Path("track")  # Don't change

    # fi_in and sclinac:
    self.fi_in = [-90.0]  # not sure what this does

    # Default params for track.dat file
    self.TRACK_DAT = {
        #TRAN
        'table_dir':'\'C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/Fields/\'',
        'rfq3d_dir':'\'C:/Users/trana/Desktop/TRACK_Development/TRACK_reinforcement_learning/Fields/rfq2-v42-trp-large-apr/\'',
        'Win':'30.5d3',
        'freqb':'12.125d6','part':0.99,
        'nqtot':1,'qq':'34.','Amass':'238.','npat':'10000','current':'0.',
        'Qdesign':'34.','Adesign':'238.',
        'db_b':0.0005,'df_4D':'180.',
        'epsnx':'0.12d0','alfax':'1.00d0','betax':'100.0d0',
        'epsny':'0.12d0','alfay':'1.00d0','betay':'100.0d0',
        'epsnz':'25.00d0','alfaz':'0.00d0','betaz':'10.0d0',
        'phmax':'180.','dwwmax':0.2,
        'x00':"0.,0.",
        'xp00':"0.,0.",
        'y00':"0.,0.",
        'yp00':"0.,0.",
        'ph00':"0.,0.",
        'dww00':"0.,0.",
        #END
        #INDEX
        'NRZ':1,'igraph':1,'iaccep':0,'isol':0,
        'iflag_dis':0,'iflag2D':0,'iflag_qq':1,'iflag_upd':13,'iflag_rms':1,
        'iint':100,'nstep_cav':50,
        'iflag_env':1,'iflag_cav':1,'iflag_ell':1,'iflag_fenv':1,'iflag_lev':0,
        'isrf':100,
        'iflag_mhb':0,'iflag_match':0,
        'iread_dis':0,'iwrite_dis':2,
        'iwrite_step':0,
        #END
        #MATCH
        'mat_type':2,'mat_opt':1,'min_typ':1,'min_opt':5,
        'min_max':1000,'min_loss':1,'min_xys':0,'min_bnd':1,
        'bnd_fac':2,'bnd_low':0,'bnd_upp':0,'beamw':0,
        'beamf':"1.4,23.,0.,1.4,23.,0.,0.,0.,0.",
        'fit_err':.01,
        #END
    }
    
    # Default setting for initial sclinac
    self.SCLINAC0 = [[1,'drift',0.001, 3.0, 3.0]]
    
    # Default setting for sclinac
    # Note, stop is automatically put each time sclinac is created
    self.SCLINAC = [[1,'drift',46.42, 3.0, 3.0],
               [68, 'mhb4', 24.0, 1.0, 140.0, 1, 5274.67, 0.0, 2, -1510.54, 0.0, 3, 917.54, 0.0, 4, -630.22, 0.0, 100],
               [-1, 'scrch'],
               [1, 'drift', 42.7, 3.0, 3.0],
               [71, 'eq3d', 3.891651480059152, 17.5, 3.0, 60],
                [72, 'eq3d', -3.6531267960581237, 17.5, 3.0 ,60],
                [1, 'drift', 89.1, 3.0, 3.0],
                [71, 'eq3d', 2.124722736160021, 17.5, 3.0, 60],
                [72, 'eq3d', -4.373542612947069, 17.5, 3.0, 60],
                [1, 'drift', 116.9, 3.0, 3.0],
                [71, 'eq3d', 3.343398023804604, 17.5, 3.0, 60],
                [72, 'eq3d', -7.209168537560333, 17.5, 3.0, 60],
                [1, 'drift', 132.4, 3.0, 3.0],
                [77, 'eq3d', 10.012, 17.145, 3.0, 60],
                [78, 'eq3d', -10.33, 27.305, 3.0, 100],
                [81, 'eq3d', 7.774, 17.145, 3.0, 60],
                [1, 'drift', 0.3175, 3.0, 3.0]]
    
    # Groups and dataset configuration for hdf5 files
    # top level are folders of the elements you are collecting data from
    # in each folders are the inputs and output folders.
    # inputs folder contains datasets and outputs also contains datasets
    # inputs and outputs must contain a tag for each location
    self.HDF5 = {'00_start':['alfax','betax','epsnx','alfay','betay','epsny'],
                 'location': ['05_eq3d','06_eq3d','08_eq3d','09_eq3d','11_eq3d','12_eq3d'],
                 'inputs': {'05_eq3d':[('voltage',2)],
                           '06_eq3d':[('voltage',2)],
                           '08_eq3d':[('voltage',2)],
                           '09_eq3d':[('voltage',2)],
                           '11_eq3d':[('voltage',2)],
                           '12_eq3d':[('voltage',2)]},
                 'outputs': {'00_start':[('x',10001),('px',10001),('y',10001),('py',10001),('l',10001)],
                             '06_eq3d':[('x',10001),('px',10001),('y',10001),('py',10001),('l',10001)],
                            '09_eq3d':[('x',10001),('px',10001),('y',10001),('py',10001),('l',10001)],
                            '12_eq3d':[('x',10001),('px',10001),('y',10001),('py',10001),('l',10001)]},
                'size':2**14,
                'index':[1,6,8,12,-1]}  # be careful about indexing. index telling which sclinac element to get distro from. index=3 mean 3 elements up to element 3 will be recorded
        
if __name__ == "config":
    print("Loading config...")
