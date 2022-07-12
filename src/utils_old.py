# -*- coding: utf-8 -*-
import numpy as np
import math
import os
from shutil import copy, move
from datetime import datetime
from config import Config
import pandas as pd

if __name__ == "utils":
    print("Loading utils...")

def random_generator(n_samples):
    cf = Config()

    alfax_list = np.random.uniform(-cf.factor2*cf.alfax,
                                   cf.factor2*cf.alfax, size=n_samples).tolist()
    alfay_list = np.random.uniform(-cf.factor2*cf.alfay,
                                   cf.factor2*cf.alfay, size=n_samples).tolist()
    alfaz_list = np.random.uniform(-cf.factor2*cf.alfaz,
                                   cf.factor2*cf.alfaz, size=n_samples).tolist()

    epsnx_list = np.abs(np.random.uniform(
        cf.factor1*cf.epsnx, cf.factor2*cf.epsnx, size=n_samples)).tolist()
    epsny_list = np.abs(np.random.uniform(
        cf.factor1*cf.epsny, cf.factor2*cf.epsny, size=n_samples)).tolist()
    epsnz_list = np.abs(np.random.uniform(
        cf.factor1*cf.epsnz, cf.factor2*cf.epsnz, size=n_samples)).tolist()

    betax_list = np.abs(np.random.uniform(
        cf.factor1*cf.betax, cf.factor2*cf.betax, size=n_samples)).tolist()
    betay_list = np.abs(np.random.uniform(
        cf.factor1*cf.betay, cf.factor2*cf.betay, size=n_samples)).tolist()
    betaz_list = np.abs(np.random.uniform(
        cf.factor1*cf.betaz, cf.factor2*cf.betaz, size=n_samples)).tolist()

    db_b_list = np.abs(np.random.uniform(
        cf.db_b_min, cf.db_b_max, size=n_samples)).tolist()
    df_4D_list = np.abs(np.random.uniform(
        cf.df_4D_min, cf.df_4D_max, size=n_samples)).tolist()

    isrf = np.random.randint(0, high=2147483646, size=1)

    alfa = {'x': alfax_list, 'y': alfay_list, 'z': alfaz_list}
    beta = {'x': betax_list, 'y': betay_list, 'z': betaz_list}
    epsn = {'x': epsnx_list, 'y': epsny_list, 'z': epsnz_list}
    bunch = {'db_b': db_b_list, 'df_4D': df_4D_list}

    input_dict = {'alfax': alfax_list,
                  'alfay': alfay_list,
                  'alfaz': alfaz_list,
                  'betax': betax_list,
                  'betay': betay_list,
                  'betaz': betaz_list,
                  'espnx': epsnx_list,
                  'espny': epsny_list,
                  'espnz': epsnz_list,
                  'db_b': db_b_list,
                  'df_40': df_4D_list,
                  'isrf': isrf}

    return input_dict

"""
def generate_coordinates(fdpath, fname='ini_dis.dat'):
    oDir = os.getcwd()  # return the current operating directory
    os.chdir(fdpath)
    subprocess.call('IniDis_To_Coord')

    # this file is unformatted binary FORTRAN output
    ini_disUnf_p = os.path.join(fdpath)
    # converter IniDis_to_Coord.exe outputs the file coord.inp to dirToRead w/ this implementation
    iniDisConvFunc(ini_disUnf_p)
    os.chdir(oDir)  # restore the original directory
"""

def files2track(simDir, trackDir):
    """
    Take all .inp and .dat files from simDir and copy to trackDir
    
    Parameters
    ----------
    simDir
        input directory
    trackDir
        output director
    """
    toCopy = sorted(list(simDir.glob('*.inp'))
                    + list(simDir.glob('*.dat')))
    for file in toCopy:
        copy(file, trackDir)

def files2folder(simDir, trackDir):
    """
    Take all .inp, .dat, .out, .out. files from trackDir and move to simDir
    
    Parameters
    ----------
    simDir
        input directory
    trackDir
        output director
    """
    toMove = sorted(list(trackDir.glob('*.out'))
                    + list(trackDir.glob('*.inp'))
                    + list(trackDir.glob('*.out.*'))
                    + list(trackDir.glob('*.dat')))
    for file in toMove:
        move(str(file), simDir)

def buildList(*args, current=[()]):
    if len(args) > 0:
        a = args[0]
        new = []
        for t in current:
            for i, val in enumerate(a):
                if len(t) == 0:  # empty tuple
                    new.append((f"-{i}", val))
                else:
                    new.append((t[0]+f"-{i}", *t[1:], val))
        # print(new)
        return buildList(*args[1:], current=new)

    return current

def currentSim(path, prefix='sim_'):
    list_ = [int(filename[4:])
             for filename in os.listdir(path) if filename.startswith(prefix)]
    list_.append(0)
    return max(list_)

def createDir(dir):
    try:
        os.mkdir(dir)
    except:
        raise '{} already exists or could not be created.'.format(dir)
    return

def logging2txt(folder, name, simsInfo=[], **kargs):
    file = os.path.join(folder, f'{name}.log')
    start = kargs.get('start')
    sd = kargs.get('n', 1)
    runtime = kargs.get('runtime')
    npat = kargs.get('npat', 0)
    done = kargs.get('done', 1)
    mode = kargs.get('mode', 'a')
    tnpat = sd*npat
    with open(file, mode) as f:
        f.write(f"Simulations started: {start}\n")
        f.write(f"completed {sd} simulations, listed below, \n")
        f.write(f"for a total of {tnpat} particles.\n")
        #f.write(f"simulations remaining: {len(sims)} \n")
        f.write(f"Average TRACK runtime: {sum(runtime)/len(runtime)} s\n")
        f.write(f"Simulations finished: {done}\n")
        f.write(f"=== Completed Simulations ===\n")
        for s in simsInfo:
            f.write(f"{s}\n")
        f.write("=============================\n\n")

def defSimInfo(**kargs):
    now = datetime.now()
    n = 1 + kargs.get('n', 0)
    currentSim = kargs.get('path')
    rt = kargs.get('runtime')
    input_vars = kargs.get('input_vars')

    simInfo = ''
    simInfo += f' n: {n} -'
    simInfo += f' start: {now} -'
    simInfo += f' path: {currentSim} -'
    simInfo += f' runtime: {rt} seconds -'
    simInfo += f'\n\t input_vars: {input_vars}'
    return simInfo

def clearFile(file):
    """
    Used to remove a file
    """
    if os.path.isfile(file):  # remove file if exist
        os.remove(file)
    
def drift(file, n, d_elem, rapx, rapy, nstep=0):
    """
    Drift element
    Parameters
    ----------
    n - any integer number
    d_elem [cm] - length of the drift space
    Rx [cm] = horizontal half aperture (X direction)
    Ry [cm] - vertical half aperture (Y direction)
    n step - (optional) number of steps for integration allong the -drift space.
    """
    if not os.path.isfile(file): ## If file do not exist, start at the top
        with open(file, 'w') as f:
            f.write(str(n)+" drift "+str(d_elem)+" " +str(rapx)+" "+str(rapy))
        if (nstep != 0):
            with open(file, 'a') as f:
                f.write(" "+str(nstep))
        return
    with open(file, 'a') as f:
        f.write("\n" + str(n) + " drift " + str(d_elem) + " " + str(rapx) + " " + str(rapy))
        if (nstep != 0):
            f.write(" "+str(nstep))
        return
    
def mhb4(file, ntype, d_elem, rap, ph0, harm1, V1, phsh1, harm2, V2, phsh2, harm3, V3, phsh3, harm4, V4, phsh4, nstep=0):
    """
    *4 Harmonic Buncher
    Parameters
    ----------
    ntype    = el_n   ! Type of the MHB to read the field distribution

    d_elem = el_p1  ! total lengh of the MHB [cm]

    rap         = el_p2  ! aperture of the MHB [cm]

    ph0        = el_p3  ! Initial phase set of the MHB   [deg]

    harm1   = el_nn1 ! harmonic number of the first frequency

    V1          = el_p4  ! inter-electrode potential of the first harmonic [V]

    phsh1    = el_p5  ! phase shift with respect to the device phase [V]

    harm2   = el_nn2 ! harmonic number of the second frequency

    V2          = el_p6  ! inter-electrode potential of the second harmonic [V]

    phsh2    = el_p7  ! phase shift with respect to the device phase [V]

    harm3   = el_nn3 ! harmonic number of the third frequency

    V3          = el_p8  ! inter-electrode potential of the first harmonic [V]

    phsh3    = el_p9  ! phase shift with respect to the device phase [V]

    harm4   = el_nn4 ! harmonic number of the fourth frequency

    V4          = el_p10  ! inter-electrode potential of the fourth harmonic [V]

    phsh4    = el_p11  ! phase shift with respect to the device phase [V]

    nstep     = el_nn5 ! number of step for integration per device
    """
    if not os.path.isfile(file): 
        with open(file, 'w') as f:
            f.write(str(ntype) + " mhb4 " + str(d_elem) + " " + str(rap) + " " + str(ph0)
                       +" "+str(harm1)+" "+str(V1)+" "+str(phsh1)
                       +" "+str(harm2)+" "+str(V2)+" "+str(phsh2)
                       +" "+str(harm3)+" "+str(V3)+" "+str(phsh3)
                       +" "+str(harm4)+" "+str(V4)+" "+str(phsh4))
        if (nstep != 0):
            with open(file, 'a') as f:
                f.write(" "+str(nstep))
        return
    with open(file, 'a') as f:
        f.write("\n" + str(ntype) + " mhb4 " + str(d_elem) + " " + str(rap) + " " + str(ph0)
                       +" "+str(harm1)+" "+str(V1)+" "+str(phsh1)
                       +" "+str(harm2)+" "+str(V2)+" "+str(phsh2)
                       +" "+str(harm3)+" "+str(V3)+" "+str(phsh3)
                       +" "+str(harm4)+" "+str(V4)+" "+str(phsh4))
        if (nstep != 0):
            f.write(" "+str(nstep))
        return
    
def eq3d(file, n, Vf, d_elem, rap, nstep=0):
    """
    3D-Field-Map Electrostatic Quadrupole

    Parameters
    ----------
    n extension number in the files eh_EM.#n

    Vf [V] inter-electrode voltage*

    d_elem [cm] total length of the quad

    Ra [cm] aperture radius

    nstep number of integration steps for the device
    """
    if not os.path.isfile(file): ## If file do not exist, start at the top
        with open(file, 'w') as f:
            f.write(str(n)+" eq3d "+str(Vf)+" " +str(d_elem)+" "+str(rap))
        if (nstep != 0):
            with open(file, 'a') as f:
                f.write(" "+str(nstep))
        return
    with open(file, 'a') as f:
        f.write("\n" + str(n)+" eq3d "+str(Vf)+" " +str(d_elem)+" "+str(rap))
        if (nstep != 0):
            f.write(" "+str(nstep))
        return

def strip(file, n, xWtarg, dWtarg, dxtarg, dds):
    """
    stripper element

    Parameters
    ----------
    n - number of the stripper film or foil and extension number of the file stripper.#0n

    xWtarg - energy loss (keV/u)

    dWtarg - RMS energy straggling in keV/u, rms value calculated by the external code,
        inTRACK we assume the distribution is Gaussian.

    dxtarg - RMS scattering in X' and Y', in radians.

    dds - fluctuation of the target thickness (relative unit), peak-to-peak amplitude. 
        The number of charge states and the percentage of particle distribution in different
        charge states, as well as average beam energy after the stripper must be given in file stripper.#0n.
    """
    if not os.path.isfile(file): ## If file do not exist, start at the top
        with open(file, 'w') as f:
            f.write(str(n)+" strip "+str(xWtarg)+" " +str(dWtarg)+" "+str(dxtarg)+" "+str(dds))
        return
    with open(file, 'a') as f:
        f.write("\n" + str(n)+" strip "+str(xWtarg)+" " +str(dWtarg)+" "+str(dxtarg)+" "+str(dds))
        return

def stop(file):
    """
    Stops simulation at this given location along the z
    """
    if not os.path.isfile(file): ## If file do not exist, start at the top
        with open(file, 'w') as f:
            f.write(str(0)+" stop")
        return
    with open(file, 'a') as f:
        f.write("\n"+str(0)+" stop")
        return

def scrch(file, n):
    """
    -1 scrch

    1 scrch

    -1 : the code writes all necessary information to temporary file scratch.#01.

    +1 : the code reads all data and starts simulation from the intermediate
    """
    if not os.path.isfile(file): ## If file do not exist, start at the top
        with open(file, 'w') as f:
            f.write(str(n)+" scrch")
        return
    with open(file, 'a') as f:
        f.write("\n"+str(n)+" scrch")
        return
    
def updateQuads(file, v1, v2, v3, v4, v5, v6):
    """
    slinac UPDATER
    Custom method to update Quads
    """
    clearFile(file)
    drift(file, n=1, d_elem=46.42, rapx=3.0, rapy=3.0)
    mhb4(file, ntype=68, d_elem=24., rap=1.,  ph0=140., harm1=1, V1=5274.67, phsh1=0., harm2=2,
         V2=-1510.54, phsh2=0., harm3=3, V3=917.54, phsh3=0., harm4=4, V4=-630.22, phsh4=0., nstep=100)
    scrch(file, n=-1)
    drift(file, n=1, d_elem=42.7, rapx=3.0, rapy=3.0)
    eq3d(file, n=71, Vf=v1, d_elem=17.5, rap=3.0, nstep=60)
    eq3d(file, n=72, Vf=v2, d_elem=17.5, rap=3.0, nstep=60)
    drift(file, n=1, d_elem=89.1, rapx=3.0, rapy=3.0)
    eq3d(file, n=71, Vf=v3, d_elem=17.5, rap=3.0, nstep=60)
    eq3d(file, n=72, Vf=v4, d_elem=17.5, rap=3.0, nstep=60)
    drift(file, n=1, d_elem=116.9, rapx=3.0, rapy=3.0)
    eq3d(file, n=71, Vf=v5, d_elem=17.5, rap=3.0, nstep=60)
    eq3d(file, n=72, Vf=v6, d_elem=17.5, rap=3.0, nstep=60)
    drift(file, n=1, d_elem=132.4, rapx=3.0, rapy=3.0)
    eq3d(file, n=77, Vf=10.012, d_elem=17.145, rap=3.0, nstep=60)
    eq3d(file, n=78, Vf=-10.330, d_elem=27.305, rap=3.0, nstep=100)
    eq3d(file, n=81, Vf=7.774, d_elem=17.145, rap=3.0, nstep=60)
    drift(file, n=1, d_elem=0.3175, rapx=3.0, rapy=3.0)
    stop(file)

def track2dataframe(file):
    """
    Input: track.dat file path
    Output: pandas DataFrame
    TODO: code breaks if we have an error in the file such as incomplete file. Need to fix
    """
    def is_eof(f):
      cur = f.tell()    # save current position
      f.seek(0, os.SEEK_END)
      end = f.tell()    # find the size of file
      f.seek(cur, os.SEEK_SET)
      return cur == end

    df = pd.DataFrame()
    with open(file, 'r') as f:
        while not is_eof(f):  # while not end of file
            line = f.readline()  # read next line
            line = line.strip()  # stip the white space
            if (len(line) != 0 and line[0]=='&'):  # If read the first &, start to read the inner block
                while (line!='&END'):  # Don't stop til you reach end mark
                    line = f.readline()
                    line = line.strip()
                    if (len(line) == 0 or line[0]=='&'): # only read lines that are not the end or empty.
                        continue
                    params = line.split("=")  # split along equal sign
                    if (len(params)==2):  #simple assignment
                        #df[params[0].strip()]=[params[1].strip()]
                        block = params[0]     
                        value = params[1].split(',')
                        if (len(value)==1):  # Case for each parameter has only one value
                            df[block.strip()]=[value[0].strip()]
                        else:  # Case when a parameter has more than one value
                            for j in range(len(value)):  # minus one because the last value must be the next name
                                df[block.strip()+str(j)]=[value[j].strip()]
                    else:
                        for i in range(len(params)-1):  # minus one because value is +1
                            block = params[i].split(',')     
                            value = params[i+1].split(',')
                            if (len(value)<=2):  # Case for each parameter has only one value
                                df[block[-1].strip()]=[value[0].strip()]
                            else:  # Case when a parameter has more than one value
                                for j in range(len(value)-1):  # minus one because the last value must be the next name
                                    df[block[-1].strip()+str(j)]=[value[j].strip()]
                                if (i+1==len(params)-1):  # if we are the last element, incude the last value
                                    df[block[-1].strip()+str(j+1)]=[value[j+1].strip()]
    return df

def dataframe2track(file, df):
    """
    Input: dataframe with track.dat parameters
    Output: track.dat file path
    """
    with open(file, 'w') as f:
        f.writelines("&TRAN")
        f.writelines("\n  table_dir=" + df.table_dir)
        f.writelines("\n  rfq3d_dir=" + df.rfq3d_dir)
        f.writelines("\n  Win="+df.Win)
        f.writelines("\n  freqb ="+df.freqb)
        f.writelines("\n  part ="+df.part)
        f.writelines("\n  nqtot="+df.nqtot)
        f.writelines("\n  qq="+df.qq)
        f.writelines("\n  Amass="+df.Amass)
        f.writelines("\n  npat="+df.npat)
        f.writelines("\n  current="+df.current)
        f.writelines("\n  Qdesign="+df.Qdesign)
        f.writelines("\n  Adesign="+df.Adesign)
        f.writelines("\n  db_b="+df.db_b)
        f.writelines("\n  df_4D="+df.df_4D)
        f.writelines("\n  epsnx="+df.epsnx)
        f.writelines("\n  alfax="+df.alfax)
        f.writelines("\n  betax ="+df.betax )
        f.writelines("\n  epsny ="+df.epsny )
        f.writelines("\n  alfay ="+df.alfay )
        f.writelines("\n  betay ="+df.betay )
        f.writelines("\n  epsnz ="+df.epsnz )
        f.writelines("\n  alfaz ="+df.alfaz )
        f.writelines("\n  betaz ="+df.betaz )
        f.writelines("\n  phmax="+df.phmax)
        f.writelines("\n  dwwmax="+df.dwwmax)
        f.writelines("\n  x00="+df.x000 +','+df.x001)
        f.writelines("\n  xp00="+df.xp000 +','+df.xp001)
        f.writelines("\n  y00="+df.y000 +','+df.y001)
        f.writelines("\n  yp00="+df.yp000 +','+df.yp001)
        f.writelines("\n  ph00="+df.ph000 +','+df.ph001)
        f.writelines("\n  dww00="+df.dww00)
        f.writelines("\n&END")
        f.writelines("\n&INDEX")
        f.writelines("\n  NRZ="+df.NRZ)
        f.writelines("\n  igraph="+df.igraph)
        f.writelines("\n  iaccep="+df.iaccep)
        f.writelines("\n  isol="+df.isol)
        f.writelines("\n  iflag_dis="+df.iflag_dis)
        f.writelines("\n  iflag2D="+df.iflag2D)
        f.writelines("\n  iflag_qq="+df.iflag_qq)
        f.writelines("\n  iflag_upd="+df.iflag_upd)
        f.writelines("\n  iflag_rms="+df.iflag_rms)
        f.writelines("\n  iint="+df.iint)
        f.writelines("\n  nstep_cav="+df.nstep_cav)
        f.writelines("\n  iflag_env="+df.iflag_env)
        f.writelines("\n  iflag_cav="+df.iflag_cav)
        f.writelines("\n  iflag_ell="+df.iflag_ell)
        f.writelines("\n  iflag_fenv="+df.iflag_fenv)
        f.writelines("\n  iflag_lev="+df.iflag_lev)
        f.writelines("\n  isrf="+df.isrf)
        f.writelines("\n  iflag_mhb="+df.iflag_mhb)
        f.writelines("\n  iflag_match="+df.iflag_match)
        f.writelines("\n  iread_dis="+df.iread_dis)
        f.writelines("\n  iwrite_dis="+df.iwrite_dis)
        f.writelines("\n  iwrite_step="+df.iwrite_step)
        f.writelines("\n&END")
        f.writelines("\n&MATCH")
        f.writelines("\n  mat_typ="+df.mat_typ)
        f.writelines("\n  mat_opt="+df.mat_opt)
        f.writelines("\n  min_typ="+df.min_typ)
        f.writelines("\n  min_opt="+df.min_opt)
        f.writelines("\n  min_max="+df.min_max)
        f.writelines("\n  min_los="+df.min_los)
        f.writelines("\n  min_xys="+df.min_xys)
        f.writelines("\n  min_bnd="+df.min_bnd)
        f.writelines("\n  bnd_fac="+df.bnd_fac)
        f.writelines("\n  bnd_low="+df.bnd_low)
        f.writelines("\n  bnd_upp="+df.bnd_upp)
        f.writelines("\n  beamw="+df.beamw)
        f.writelines("\n  beamf="+df.beamf0+','+df.beamf1+','+df.beamf2+
                     ','+df.beamf3+','+df.beamf4+','+df.beamf5+
                     ','+df.beamf6+','+df.beamf7+','+df.beamf8)
        f.writelines("\n  fit_err="+df.fit_err)
        f.writelines("\n&END")
    
# ---Start Code from Wade Fisher---
# define our underlying PDF to sample
def pdfFunc(xvar, sd):
    """
    Currently creates a simple guassian distribution function
    TODO: add other distribution functions
    """
    mean = 0
    stddev = sd
    retval = 1/np.sqrt(2*math.pi)
    retval *= math.exp(-1*math.pow(xvar-mean,2)/(2*math.pow(stddev,2)))
    return retval

# define our proposal function
def propFunc(xvar, sd):
    stddev = sd
    retval = np.random.normal(xvar,stddev)
    return retval

# define our acceptance function
def acceptPoint(x1, x2, sd, rand):
    threshold = np.minimum(1,pdfFunc(x2, sd)/(pdfFunc(x1, sd)+1e-9))
    if rand < threshold:
        return x2
    else:
        return x1
    
def mcmc(n, seed, sd):
    """
    creates a 1-D dsitribution following the current propFunc which is currently a guassina distribution
    TODO: allow other distributions
    TODO: add seeds implmentations
    Input:
    n - number of particles
    seed - seed for random number
    sd - standard deviation for guassian distribution

    Output:
    (i, xi)
    """
    seq1 = []
    seed1 = 0

    x1 = seed1
    seq1.append([0,x1])

    for i in range(1,n):    
        x1 = acceptPoint(x1,propFunc(x1, sd), sd, np.random.uniform(0,1))    
        seq1.append([i,x1])
    return zip(*seq1) # The * unpacks all arguments into zip(x1,y1) 

#---End Code from Wade Fisher---

def create_guassian_distribution(n,mean,sdx, sdpx, sdy, sdpy, dww_lower, dww_upper):
    """
    Given the number of particles and the parameters for each coordinates, return
    a dataframe with the distribution for all particles
    """
    x1, x = mcmc(n, mean, sdx) 
    x2, px = mcmc(n, mean, sdpx)
    x3, y = mcmc(n, mean, sdy)  
    x4, py = mcmc(n, mean, sdpy)
    phi = np.random.uniform(-np.pi,np.pi,n)
    dww = np.random.uniform(dww_lower,dww_upper,n)
    zero = np.zeros(n, dtype=np.int8)  # all particle at inside
    data=pd.DataFrame(np.array([x,px,y,py,phi,dww,zero]).T,columns=['x','px','y','py','phi','dww','l'])
    data.l=data.l.astype(int)
    return data

def dis2dataframe(file):
    """
    extract the distribution from a read_dis file and put it into a pandas dataframe.
    Input: path to a read_dis file
    Output: Pandas DataFrame with all the coordinates
    """
    with open(file) as f:
        lines = f.readlines()
    inputs = []
    for line in lines:
        inputs.append(line.strip().split())
    coord = pd.DataFrame(inputs[3:], columns=["x","px","y","py","phi","dww","l"])
    coord = coord.apply(pd.to_numeric)  # make string into numeric
    return coord

def df2dis(path, df, Win, nqtot, qq):
    """
    Given a data frame with x,px,y,py,phi,dww,l
    Note: L must be an int
    and also information from track.dat file such as Win, nqtot, qq, and something else
    make a read_dis file that track can read.
    Input:
    -----
    path: file path
    dataframe: pandas datafraem obeject with col x,px,y,py,phi,dww, and l
    Win: energy
    nqtot: number of particle types
    qq: charge

    Output: read_dis file at the file path
    """
    file = str(path) + "\\read_dis.dat"
    data = zip(df['x'],df['px'],df['y'],df['py'],df['phi'],df['dww'],df['l'])
    with open(file, 'w') as f:
        f.writelines("  {:.15E}       {}".format(Win, nqtot))
        f.writelines("\n   "+str(df.shape[0]))
        f.writelines('\n  {:.15E}'.format(qq)) 
        f.writelines('\n  {:.15E}  {:.15E}  {:.15E}  {:.15E}  {:.15E}  0.809242037859003E-02       0'.format(0,0,0,0,0))
        for x,px,yt,py,phi,dww,l in data:
            f.writelines("\n  {:.15E} {:.15E} {:.15E} {:.15E}  {:.15E}  {:.15E}       {}".format(x,px,yt,py,phi,dww,l))
            