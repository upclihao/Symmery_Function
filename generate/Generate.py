__coding__ = "utf-8"
__author__ = "Hao L"
__email__ = "929752788@qq.com"
__date__ = "2018-12-3"
__version__ = "0.1"

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

    
def get_atom_NUM(Path):
    """
    Get molecular atomic number in molecular log file
    
    Parameters:
    ----------
    Path: file path to the log file
    
    Returns:
    ----------
    out: atomic number
    
    Example:
    ----------
    >>> get_atom_NUM('s00001.log')
    27
    """
    f = open(Path)
    f_lines = f.readlines()
    for i,f_line in enumerate(f_lines):
        if f_lines[i].startswith(' NAtoms=') and f_lines[i+1].startswith('    '):
            NUM = f_lines[i].split()[1]
    f.close()
    return int(NUM)

def get_coord(Path):
    """
    Get molecular Cartesian coordinates from log file
    
    Parameters:
    ----------
    Path: file path to the log file
    
    Returns:
    ----------
    out: An array object code molecular Cartesian coorfinates
    
    Example:
    ----------
    >>> get_coord('s0001.log')
    array[[1,0,0,0],[6,1,0,0],...,]
    """
    xyz = np.zeros((get_atom_NUM(Path),4))
    f = open(Path)
    f_lines = f.readlines()[::-1]
    for i,f_line in enumerate(f_lines):
        if f_line.startswith(' Rotational constants (GHZ)') \
        and f_lines[i+1].startswith(' ---------------------------------------'):
            xyz_lines = f_lines[i+1+get_atom_NUM(Path):i+1:-1]
            break
    for j,xyz_line in enumerate(xyz_lines):
        xyz[j,0] = xyz_line.split()[1]
        xyz[j,1] = xyz_line.split()[3]
        xyz[j,2] = xyz_line.split()[4]
        xyz[j,3] = xyz_line.split()[5]    
    f.close()
    return xyz

def get_func_para(Path):
    """
    Get radius and angular symmery function parameters
    from file 
    
    Parameters:
    ----------
    Path: file path to the file
    
    Returns:
    ----------
    out1: radius symmery function parameters in array object
    out2: angular symmery function parameters in array object
    
    Example:
    ----------
    >>> get_func_para('NMA-H2O.txt')
    array[[1,0.00003,0,6.5],....],
    array[[1,1,0.00003,-1,1,6.5]]
    """
    file = open(Path)
    f_lines = file.readlines()
    for i,f_line in enumerate(f_lines):
        if f_line.startswith('#G=2'):
            index_1 = i
        elif f_line.startswith('#G2END-'):
            index_2 = i
        elif f_line.startswith('#G=4'):
            index_3 = i
        elif f_line.startswith('#G4END-'):
            index_4 = i
    G2PARA = np.zeros((index_2-index_1-2,4))
    G4PARA = np.zeros((index_4-index_3-2,6))
    for i,f_line in enumerate(f_lines[index_1+2:index_2]):
        G2PARA[i,0] = f_line.split()[0]
        G2PARA[i,1] = f_line.split()[1]
        G2PARA[i,2] = f_line.split()[2]
        G2PARA[i,3] = f_line.split()[3]
    for i,f_line in enumerate(f_lines[index_3+2:index_4]):
        G4PARA[i,0] = f_line.split()[0]
        G4PARA[i,1] = f_line.split()[1]
        G4PARA[i,2] = f_line.split()[2]
        G4PARA[i,3] = f_line.split()[3]
        G4PARA[i,4] = f_line.split()[4]
        G4PARA[i,5] = f_line.split()[5]
    file.close()
    return G2PARA,G4PARA

def get_distance(Matrix_1,Matrix_2):
    """
    Get distance between atom_1 and atom_2
    
    Parameters:
    ----------
    Matrix_1: An vector code atom_1's coordinates
    Matrix_2: An vector code atom_2's coordinates
    
    Returns:
    ----------
    out: distance between vector_1 and vector_2
    
    Example:
    ----------
    >>> get_distance([0,0,0,0],[0,1,0,0,])
    1
    
    Notes:
    ----------
    The first number of input vector often code atmoic index number which ignored
    in this function
    """
    return np.linalg.norm(Matrix_1[1:]-Matrix_2[1:])   

def cutoff_func(R,Rc):
    """
    Calculate the cutoff function value for radiu R
    Ref. J. Behler, International Journal of Quantum Chemistry 115, 1032 (2015). (6)
    
    Parameters:
    ----------
    R: Distance number
    Rc: Cut off radiu
    
    Return:
    ----------
    out: Cut off function value of radiu R in condition cut off radiu Rc
    
    Example:
    ----------
    >>> cutoff_func(0,6.5)
    1
    """
    Rc = Rc
    if R <= Rc:
        fc = 0.5 * (np.cos(np.pi * R / Rc) + 1)
    else:
        fc = 0
    return fc

def G2func(Eta,R,Rs,Fc):
    """
    Calculate radius symmery function value 
    Ref. J. Behler, International Journal of Quantum Chemistry 115, 1032 (2015). (9)
    
    Parameters:
    ----------
    Eta,Rs: Symmery function parameters from parameter file
    R: Distance between atom i and j
    Fc: Cut off function value of radiu R
    
    Return:
    ----------
    out: Radius symmery function value
    """
    return (np.exp(-1 * Eta * (R - Rs) ** 2)) * Fc

def G4func(Zeta,Lambda,Cos,Eta,R1,R2,R3,Fc1,Fc2,Fc3):
    """
    Calculate angular symmery function value 
    Ref. J. Behler, International Journal of Quantum Chemistry 115, 1032 (2015). (10)
    
    Parameters:
    ----------
    Zeta,Lambda,Eta: Symmery function parameters from parameter file
    Cos: Cos value with angle jik
    R1,R2,R3: Distance between ij, ik, jk
    Fc1,Fc2,Fc3: Cut off function values of R1,R2,R3
    
    Returns:
    ----------
    out: Angular symmery function value 
    """
    return (2 ** (1 - Zeta)) * ((1 + Lambda * Cos) ** Zeta) * \
            np.exp(-1 * Eta * (R1 ** 2 + R2 ** 2 + R3 ** 2)) * \
            Fc1 * Fc2 * Fc3    

def get_cosangle(Matrix_C,Matrix_1,Matrix_2):
    """
    Calculate cos angle between <ijk
    
    Parameters:
    ----------
    Matrix_C: Central atomic coordinates
    Matrix_1,Matrix_2: Coordinates of atom j and k
    
    Returns:
    ----------
    out: Cos value with angle jik
    
    Example:
    ----------
    >>> get_cosangle([1,0,0,0],[7,0,0,1],[6,1,0,0])
    -1
    
    Notes:
    ----------
    The first number of input vector often code atmoic index number which ignored
    in this function
    """
    v1 = Matrix_1[1:] - Matrix_C[1:]
    v2 = Matrix_2[1:] - Matrix_C[1:]
    DOT = np.dot(v1,v2)    
    return DOT/np.linalg.norm(v1)/np.linalg.norm(v2)

def get_mol_radisym(Matrix,Params):
    """
    Get radius symmery function values of an molecule to describe its structure
    
    Parameters:
    ----------
    Matrix: Molecular coordinates array
    Params: Radius symmery function parameters array
    
    Returns:
    ----------
    out: Radius symmery function array of molecule
    """
    sym = np.zeros((Matrix.shape[0],Params.shape[0]))
    for i,coord in enumerate(Matrix):
        for j,Param in enumerate(Params):
            n = 0
            for k in np.delete(range(Matrix.shape[0]),i):
                if Matrix[k,0] == Param[0]:
                    r = get_distance(coord,Matrix[k])
                    fc = cutoff_func(r,Param[-1])
                    n += G2func(Param[1],r,Param[2],fc)
            sym[i,j] = n
    return sym

def get_mol_angsym(Matrix,Params):
    """
    Get molecular angular symmery function values 
    
    Parameters:
    ----------
    Matrix: Molecular coordinates array
    Params: Angular symmery function parameters array
    
    Returns:
    ----------
    out: Angular symmery function array of molecule
    """
    sym = np.zeros((Matrix.shape[0],Params.shape[0]))
    for i,coord in enumerate(Matrix):
        for j,Param in enumerate(Params):
            n = 0
            for k in itertools.combinations(np.delete(range(Matrix.shape[0]),i),2):                
                if ((Matrix[k[0],0] == Param[0] and Matrix[k[1],0] == Param[1]) or \
                   (Matrix[k[0],0] == Param[1] and Matrix[k[1],0] == Param[0])):
                    r0 = get_distance(coord,Matrix[k[0]])
                    r1 = get_distance(coord,Matrix[k[1]])
                    r2 = get_distance(Matrix[k[0]],Matrix[k[1]])
                    cos = get_cosangle(coord,Matrix[k[0]],Matrix[k[1]])
                    fc0 = cutoff_func(r0,Param[-1])
                    fc1 = cutoff_func(r1,Param[-1])
                    fc2 = cutoff_func(r2,Param[-1])
                    n += G4func(Param[4],Param[3],cos,Param[2],r0,r1,r2,fc0,fc1,fc2)
            sym[i,j] = n
    return sym

def get_mol_sym(Path_Mol,Path_Param):
    """
    Get molecular symmery function values to describe its structure
    
    Parameters:
    ----------
    Path_Mol: File path to log file
    Path_Param: File path to symmery function parameter file
    
    Returns:
    ----------
    out: Symmery function values of molecule in array object
    """
    Radi_Params, Ange_Params = get_func_para(Path_Param)
    MOL = get_coord(Path_Mol)
    Radi = get_mol_radisym(MOL,Radi_Params)
    Ange = get_mol_angsym(MOL,Ange_Params)
    return np.hstack((Radi,Ange))

def get_energy(Path):
    """
    Get energy from log file
    
    Parameters:
    ----------
    Path: File path to log file
    
    Returns:
    ----------
    out: Energy number
    
    Example:
    ----------
    >>> get_energy('s00000.log')
    -17026.006770207
    
    Notes:
    ----------
    The output energy in this function is in eV but not in Hartrees
    """
    with open(Path) as f:
        f_lines = f.readlines()
        for f_line in f_lines:
            if f_line.startswith(' SCF Done:'):
                E = np.float64(f_line.split()[4])
    return E * 27

def batch_generate(Path_1,Path_2):
    """
    Generate symmery function descriptors for vast molecules
    
    Parameters:
    ----------
    Path_1: File path to input file
    Path_2: File path to symmery function parameter file
    
    Returns:
    ----------
    out1: An array with descriptor for each molecule in input file
    out2: An array with energy for each molecule in input file
    
    Example:
    ----------
    >>> batch_generate(path_1+ 'input.txt',path_1+'NMA_H2O.txt')
    [[[5.32890071e+00 1.98285091e+00 6.49030512e-01 ... 0.00000000e+00
       5.85846492e-02 5.21274806e-03]
      ...
      [3.62506538e+00 7.46162658e-01 4.25800346e-01 ... 0.00000000e+00
       1.44153917e-02 1.12073199e-02]]]
    """
    f = open(Path_1)
    f_lines = f.readlines()
    dat_sym = np.zeros((len(f_lines),27,212))
    dat_eng = np.zeros(len(f_lines))
    for f_line in f_lines:
        index = int(f_line[-10:-5])
        dat_sym[index] = get_mol_sym(f_line.strip('\n'),Path_2)
        dat_eng[index] = get_energy(f_line.strip('\n'))
    return dat_sym,dat_eng
