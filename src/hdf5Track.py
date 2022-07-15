import sys
from pathlib import Path
import pandas as pd
pd.set_option("display.max_columns", None)
import time
import h5py

sys.path.insert(0, 'C:\\Users\\trana\\Desktop\\pyTRACK2/src')  # Location of PyTrack
from runTrack import *
from config import Config
from utils import *

if __name__ == "hdf5Track":
    print("Loading hdf5Track...")

cf = Config()
"""
1. Code which creates an hdf5 file
2. Creates the lattice file
3. Create the track.dat file
4. Run the simulation
5. Saves the data of the simulation in the hdf5 file
"""


def makehdf5(hdf5_path, hdf5_config, size=2**14):
    """
    Make a hdf5 file.
    This just created the empty file to for filling in.
    have to custom make this for each sclinac.
    The input and output doesn't have to exist for each beam element location
    Inputs:
    hdf5_path: str - path where hdf5_file will be stored
    location: list - beam element of where to take beam measurements. Will make a folder for each location
    inputs: dict - inputs of simulations at each location
    outputs: dict - outputs of simulations at each location
    """
    # Write file
    datebase = h5py.File(hdf5_path, "w")
    # Make group
    s = datebase.create_group("00_start")  # make start folder
    for elem in hdf5_config['00_start']:
        s.create_dataset(elem, (size,), maxshape=(None))  # make datasets for track.dat settings
    o = s.create_group('outputs')
    if '00_start' in hdf5_config['outputs']:  # if location elem is in the hdf5 output save config        
        for out in hdf5_config['outputs']['00_start']:
            o.create_dataset(out[0], (size, out[1]), maxshape=(None, out[1]))
    for elem in hdf5_config['location']:
        g = datebase.create_group(elem)
        i = g.create_group('inputs')
        o = g.create_group('outputs')
        if elem in hdf5_config['inputs']:  # if location elem is in the hdf5 input save config 
            for put in hdf5_config['inputs'][elem]:
                i.create_dataset(put[0], (size,), maxshape=(None))
        if elem in hdf5_config['outputs']:  # if location elem is in the hdf5 output save config        
            for out in hdf5_config['outputs'][elem]:
                o.create_dataset(out[0], (size, out[1]), maxshape=(None, out[1]))
    datebase.close()
    
def get_distro(folder, sclinac_config, track_config, index, cf=cf):
    """
    slinac UPDATER
    Custom method to get distribution
    Works by rewriting sclinac file up til location index
    Then run the simulation again
    
    Input:
    folder: folder of current track instance
    sclinac_config: list of lists of the sclinac
    index: index telling which sclinac element to get distro from. index=3 mean 3 elements up to element 3 will be recorded
    cf: configuration file
    
    Output:
    dist: - pandas dataFrame of distribution
    """
    sclinac = str(folder)+'/sclinac.dat'
    track = str(folder)+'/track.dat'
    distro = str(folder)+'/read_dis.out'
    # method to update sclinac.dat
    make_sclinac(sclinac, sclinac_config[:index])
    
    # If want to get distribution, must make sure TRACK makes them.
    if track_config['iwrite_dis'] != 2:
        track_config['iwrite_dis'] = 2
    if track_config['iread_dis'] != 0:
        track_config['iread_dis'] = 0
    make_track(track, track_config)
    
    _, t = runTrack(cf.TRACK_EXE, folder) #track exe and file directory of track exe
    # And get the initial distribution
    dist = dis2dataframe(distro)
    return dist

def get_beamout(folder, cf=cf):
    """
    Get beam.out file into a pandas df. Must be used after running the simulation first.
    
    Input:
    folder: folder of current track instance
    cf: configuration file
    
    Output:
    df: - pandas dataFrame of beam.out
    """
    beam = str(folder)+'/beam.out'
    df =  pd.read_csv(beam, delim_whitespace=True, header=0)
    return df

def save_inputs(hdf5_file, location, input_dic, index):
    """
    Save corresponding inputs in the hdf5 file at index location. Save input from sclinac.dat file
    Go to correct hdf5 file >> beam element location >> input folder >> dictionary >> index
    Input:
    hdf5_file: path to hdf5 file
    location: the beam element name
    input_dic: dictionary containing the input of beam element
    index: trial number
    """
    # TODO: check for null case
    with h5py.File(hdf5_file, "a") as database:
        for var in input_dic:
            dset = database[f'{location}/inputs/{var}']
            dset[index] = input_dic[var]
    return

def save_outputs(hdf5_file, sim_folder, sclinac_config, track_config, output_dic, location, index, trial):
    """
    Save corresponding inputs in the hdf5 file at index location.
    Go to correct hdf5 file >> beam element location >> output folder >> dictionary >> index
    NOTE: output requires running TRACK up to a point.
    Input:
    hdf5_file: path to hdf5 file
    sim_folder: folder where TRACK is.
    output_dic: dictionary containing the outputs of beam element for example [x,y,px,py]
    sclinac_config and track_config: config dictionary in order to setup track.
    index: index telling which sclinac element to get distro from. index=3 mean 3 elements up to element 3 will be recorded 
    trial: trial number
    """
    # TODO: check for null case
    df = get_distro(sim_folder, sclinac_config, track_config, index) # gets the coords ["x","px","y","py","phi","dww","l"]
    with h5py.File(hdf5_file, "a") as database:
        for var in output_dic:
            dset = database[f'{location}/outputs/{var[0]}']
            dset[trial] = df[var[0]]
    return

def run(run_config, sim_folder, hdf5_file, trial, cf=cf):
    """
    Custom run file to run one TRACK configurate and extract the required information 
    Note: the type of beam element must be a constant.
    1. copy defult configurate of TRACK.dat and sclinac.dat
    2. update TRACK.dat and sclinac.dat file and save inputs
    3. run TRACK and save distributions at corresponding locations.
    
    Inputs:
    run_config - dirction which contains important information about the track file, sclinac file, and locations
    sim_folder - location of simualation folder
    hdf5_file - location of hdf5 file
    trial - trial number
    """
    cur_track = cf.TRACK_DAT
    cur_sclinac = cf.SCLINAC
    cur_sclinac0 = cf.SCLINAC0  # TODO: implement initial distro
    cur_hdf5 = cf.HDF5
    track_file = f"{sim_folder}/track.dat"
    sclinac_file = f"{sim_folder}/sclinac.dat"
    
    track_settings = run_config['track']  # given the track setting we want to update, this should update the track settings
    for i in track_settings:  # just use a dictionary so it's easy to configure.
        cur_track[i] = track_settings[i] 
    
    # updating sclinac will be harder so we'll have to use indexes [element][argument][value]
    sclinac_setting = run_config['sclinac']
    for i in sclinac_setting:
        cur_sclinac[i[0]][i[1]] = i[2] 
    
    make_track(track_file, cur_track)  # update track.dat file
    locations_in = cur_hdf5['inputs']  # extract locations of inputs
    for location in locations_in:
        if len(run_config['inputs'])==0:
            break
        save_inputs(hdf5_file, location, run_config['inputs'][location], trial)  # saves input data at location
    
    make_sclinac(sclinac_file, cur_sclinac)  # update sclinac.dat file
    index = cur_hdf5['index']
    
    locations_out = [i for i in cur_hdf5['outputs']] # extract locations of outputs
    for i in range(len(locations_out)):  
        save_outputs(hdf5_file, sim_folder, cur_sclinac, cur_track, cur_hdf5['outputs'][locations_out[i]], locations_out[i], index[i], trial)
    return

def datagen(inputs, child, hdf5):
    """
    inputs:
    inputs - vectors of different inputs. Must all have the same dimensions
    child - folder where simulation will run
    hdf5 - hdf5 file. Where all the data is stored
    """
    # This dictionary contains the variables you want to change.
    # Make sure the run_config matches the input and output hdf5 configs
    # At the moment, you have to specify voltages for both the sclinac and inputs.
    if len(set(map(len,inputs)))==1:
        the_len = len(inputs[next(iter(inputs))])
    else:
        print("They are not the same length!")
        raise ValueError('not all lists have same length!')
    for i in range(the_len):
        v1 = inputs['v1']
        v2 = inputs['v2']
        v3 = inputs['v3']
        v4 = inputs['v4']
        v5 = inputs['v5']
        v6 = inputs['v6']
        run_config = {'track': {'epsnx':'0.12d0',
                                'alfax':'1.00d0',
                                'betax':'100.0d0',
                                'epsny':'0.12d0',
                                'alfay':'1.00d0',
                                'betay':'100.0d0'},
                     'sclinac': [[4,2,v1[i]],[5,2,v2[i]],[7,2,v3[i]],[8,2,v4[i]],[10,2,v5[i]],[11,2,v6[i]]],
                     'inputs':{'05_eq3d':{'voltage':v1[i]},  # first index in name, second index is value
                               '06_eq3d':{'voltage':v2[i]},
                              '08_eq3d':{'voltage':v3[i]},
                              '09_eq3d':{'voltage':v4[i]},
                              '11_eq3d':{'voltage':v5[i]},
                              '12_eq3d':{'voltage':v6[i]}}
                     }
        run(run_config, child, hdf5, i)