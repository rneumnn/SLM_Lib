"""
Package for processing spacetime-beams related data
"""
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar, Union
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import re
from .Visualizing import showImage
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLMResults
from dataclasses import dataclass
import time

MOTOR_STEP_SIZE = 50e-9 #m
C0 = 299792458  #m/s
DType = TypeVar("DType", bound=np.generic)
#types
DataArray = Annotated[npt.NDArray[DType],Literal["t", "X", "Y"]]
TimeShapedArray = Annotated[npt.NDArray[DType], Literal["t"]]

#structures
@dataclass
class OGVMeasurement:
    # structure to store OGV Meassurement data
    OGV: float
    OGVError: float
    LOGV: float 
    LOGVError: float
    avgRadius: float
    avgRadiusError: float
    radii: TimeShapedArray[np.float64]
    radiiError: TimeShapedArray[np.float64]
    anglesMeassured: TimeShapedArray[np.float64]
    intensities: TimeShapedArray[np.float64]
    times: TimeShapedArray[np.float64]
    angleFit: RLMResults
    angles: TimeShapedArray[np.float64]

@dataclass
class STPulseData:
    data: DataArray[np.float64]
    times: TimeShapedArray[np.float64]
    motorPos: TimeShapedArray[np.float64]

    def safeToFile(self, path:Union[str, Path]) -> None:
        """
        Saves the spacetime pulse data to a file.

        Parameters
        ----------
        path : str | Path
            The path where the data should be saved.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".npz":
            path = path.with_suffix(".npz")
        if not ParameterKeys.N_SLICES in path.stem:
            path.with_name(path.stem+f"{ParameterKeys.N_SLICES}={self.data.shape[0]}") # add number of slices to the filename
        if not os.path.isdir(path.parent):
            os.makedirs(path.parent)
        np.savez_compressed(path, data=self.data, times=self.times, motorPos=self.motorPos)
    
    def loadFromFile(path:Union[str, Path]) -> "STPulseData":
        """
        Loads spacetime pulse data from a file.

        Parameters
        ----------
        path : str | Path
            The path to the file containing the spacetime pulse data.

        Returns
        -------
        STPulseData
            An instance of STPulseData containing the loaded data.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".npz":
            path = path.with_suffix(".npz")
        with np.load(path) as data:
            return STPulseData(data=data["data"], times=data["times"], motorPos=data["motorPos"])
    
@dataclass
class CircleFit:
    center: tuple[float, float]
    radius: float
    error: float

# ParameterKeys
class ParameterKeys:
    TOPOLOGICALCHARGE = r"tc"
    DELTA_L = r"dl"
    N_FREQUENCYBINS = r"n"
    SLM_POSITION = r"SLM"
    TOTAL_DL = r"totaldl"
    CAM_POSITION = r"camPos"
    MOTOR_POSITION = r"motorPos"
    N_SLICES = r"nSlices"

    def listKeys():
        """
        Returns a list of all keys in the ParameterKeys class.
        
        Returns
        -------
        list
            A list of strings representing the keys.
        """
        return [ParameterKeys.TOPOLOGICALCHARGE,
                ParameterKeys.DELTA_L,
                ParameterKeys.N_FREQUENCYBINS,
                ParameterKeys.SLM_POSITION,
                ParameterKeys.TOTAL_DL,
                ParameterKeys.CAM_POSITION,
                ParameterKeys.MOTOR_POSITION,
                ParameterKeys.N_SLICES]
    
    def extractParameter(parameter, filename) -> int:
        """
        Extracts the parameter name from a given filename.
        """

        pattern = parameter + r"\s*(?:=\s*)?(-?\d+)"

        match = re.search(pattern, filename)
        value = None
        if match:
            value = int(match.group(1))
        return value
    
    def getAllParams(filename) -> dict[str, int]:
        """
        Extracts all parameters from a given filename.

        Parameters
        ----------
        filename : str
            The filename from which to extract parameters.

        Returns
        -------
        dict[str, float]
            A dictionary containing the extracted parameters.
        """
        params = {}
        for key in ParameterKeys.listKeys():
            value = ParameterKeys.extractParameter(key, filename)
            if value is not None:
                params[key] = value
        return params
    
    def dictToName(params:dict[str, int]) -> str:
        """
        Converts a dictionary of parameters to a string representation.

        Parameters
        ----------
        params : dict[str, int]
            A dictionary containing parameter names and their values.

        Returns
        -------
        str
            A string representation of the parameters.
        """
        return "_".join([f"{key}={value}" for key, value in params.items() if value is not None])

def _searchForSTPulseData(folder:Path) -> Path:
    """
    Searches for spacetime pulse data in a folder.

    Parameters
    ----------
    folder : Path
        The folder to search in.
    Raises
    ------
    FileNotFoundError
        If no spacetime pulse data is found in the folder.
    FileExistsError
        If multiple spacetime pulse data files are found in the folder.
    Returns
    -------
    filepath : Path
        The spacetime pulse data found in the folder.
    """
    files = os.listdir(folder)
    files = [f for f in files if f[-4:] == ".npz"]
    if len(files) == 0:
        raise FileNotFoundError(f"No spacetime pulse data found in {folder}")
    elif len(files) > 1:
        raise FileExistsError(f"Multiple spacetime pulse data files found in {folder}: {files}")
    filePath = folder/Path(files[0])
    return filePath

def _cropSTPulseData(data:STPulseData, mask) -> STPulseData:
    """
    Crops the spacetime pulse data using a mask.

    Parameters
    ----------
    data : STPulseData
        The spacetime pulse data to crop.
    mask : np.ndarray
        A boolean mask to apply to the data.

    Returns
    -------
    STPulseData
        The cropped spacetime pulse data.
    """
    new = [field[mask] if field is not None else None for field in (data.data, data.times, data.motorPos)]
    return STPulseData(*new)

def _checkDataType(arg):
    if isinstance(arg, np.ndarray):
        return arg 
    elif (isinstance(arg, Path) or isinstance(arg, str)):
            return np.load(arg)
    else:
        raise TypeError(f"Wrong type of input argument. Must be type either {type(str)}, {type(Path)} or {type(np.ndarray)}!")
    
def reconstructData(file, 
                     cx:int, cy:int, A:int, B:int, thickness:int, angle_deg:float = 0,
                     crop_pos = ((500,500),1000), 
                     plot_all:bool=False, 
                     plot_reconstruction:bool=False, 
                     fft_crop_radius:float = -1, 
                     crop_shift:tuple[int,int] = (0,0),
                     subtract_ref = None) -> np.ndarray:
    """
    Reconstructs the full electric field (phase and amplitude) from off-axis holography interference data.

    Parameters
    ----------
    file : str | Path | np.ndarray
        Path to the file to reconstruct or the raw data as np.ndarray.
    cx : int
        Center x-position of the cropping ellipse.
    cy : int
        Center y-position of the cropping ellipse.
    A : int
        Ellipse total width.
    B : int
        Ellipse total height.
    thickness : int
        Ellipse thickness.
    angle_deg : float, optional (default=0)
        Rotation angle of the ellipse in degrees.
    crop_pos : tuple, optional (default=((500,500),1000))
        Crop position (x, y) and width: ((posX, posY), width) of the data crop.
    plot_all : bool, optional (default=False)
        If True, plot all reconstruction steps.
    plot_reconstruction : bool, optional (default=False)
        If True, plot the reconstructed phase and amplitude.
    fft_crop_radius : float, optional (default=-1)
        Crop width of the masked FFT. If -1, automatically chosen.
    crop_shift : tuple[int, int], optional (default=(0,0))
        Shift coordinates for the ellipse center.
    subtract_ref : str | Path | np.ndarray, optional (default=None)
        Data to subtract from the input.

    Returns
    -------
    data_proc : np.ndarray
        The reconstructed complex field.
    """
    theta = angle_deg*(np.pi/180)
    rout = max((A,B))/2 + thickness
    # ellipse equation with "radius" varying 't' from the input parameters of the function. returns np.ndarray. Values = 0 describing the specified ellipse
    ellipse = lambda t:(np.power((X-cx)*np.cos(theta)+(Y-cy)*np.sin(theta),2)/((A+t)/2)**2)+(np.power(-(Y-cy)*np.cos(theta)+(X-cx)*np.sin(theta),2)/((B+t)/2)**2)-1 

    # read data
    data = _checkDataType(file).astype(np.int16)
    # plt.figure()
    # plt.imshow(data)
    # plt.title("loaded data")
    # plt.show()
    if subtract_ref is not None:
        ref = _checkDataType(subtract_ref)
        sub = data-ref
        if plot_all:
            fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
            fig.set_size_inches(12,3)
            axs[0].set_title("Interference raw")
            axs[1].set_title("no Interference raw")
            axs[2].set_title("subtracted")
            a = axs[0].imshow(data, origin = "lower")
            fig.colorbar(a, ax=axs[0],location='bottom')
            b = axs[1].imshow(ref, origin = "lower")
            fig.colorbar(b, ax=axs[1],location='bottom')
            c = axs[2].imshow(sub,cmap = "plasma", origin = "lower")
            fig.colorbar(c, ax=axs[2],location='bottom')
            plt.show(block = False)
        data = sub

    center = crop_pos[0]
    width = crop_pos[1]
    data = data[int(center[0]-width/2):int(center[0]+width/2),
                int(center[1]-width/2):int(center[1]+width/2)]
    # create coordinates
    nx, ny = data.shape
    X, Y = np.meshgrid( np.arange(nx), np.arange(ny), indexing="ij" )
    
    # crop radius for fft
    r_crop_fft = int(rout*fft_crop_radius)
    if fft_crop_radius == -1:
        r_crop_fft = np.min( [ np.abs( nx-cx ), np.abs( cx ), np.abs( ny-cy ), np.abs( cy ) ] )

    # fft
    PIXEL_SIZE = 3.3
    fft = np.fft.fftshift( np.fft.fft2(data) )
    kx = np.fft.fftshift( np.fft.fftfreq( nx, PIXEL_SIZE ) ) * 2*np.pi
    ky = np.fft.fftshift( np.fft.fftfreq( ny, PIXEL_SIZE ) ) * 2*np.pi
    fft_proc = fft.copy()

    #crop fft ellipse
    mask = (0 < ellipse(0)) & (0 > ellipse(thickness))
    # plt.figure()
    # plt.contour(ellipse(0))
    # plt.colorbar()
    # plt.title("ellipse array")
    # plt.show()
    # plt.figure()
    # plt.imshow(mask)
    # plt.colorbar()
    # plt.title("mask array")
    # plt.show()

    fft_proc[ np.logical_not( mask ) ] = 0
    fft_proc = fft_proc[cx-r_crop_fft:cx+r_crop_fft, cy-r_crop_fft:cy+r_crop_fft]#fft_proc[cx+crop_shift[0]-r_crop_fft:cx+crop_shift[0]+r_crop_fft, cy+crop_shift[1]-r_crop_fft:cy+crop_shift[1]+r_crop_fft] #fft_proc[cx-L:cx+L,  cy-L:cy+L ]
    if not fft_proc.shape[0] == fft_proc.shape[1]:
        raise IndexError(f"shape of the fft crop array is wrong: {fft_proc.shape}")
    # plt.figure()
    # plt.imshow(abs(fft_proc),cmap="plasma")
    # plt.show()
    ##reconstruct data
    data_proc = np.fft.ifft2( np.fft.fftshift( fft_proc ) )
    if plot_reconstruction:
        # plot
        fig, axs = plt.subplots( 1, 2, figsize=(12,5), dpi=100 )     
        plt.sca( axs[0] )
        plt.imshow( np.abs(data_proc).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Intensity" )
        plt.sca( axs[1] )
        plt.imshow( np.angle(data_proc).T, 
                   cmap="hsv",
                   origin="lower",
                   vmin=-np.pi,
                   vmax=np.pi,
                   alpha=np.abs(data_proc).T/np.max(np.abs(data_proc))
                )
        plt.colorbar()
        plt.title( "Reconstructed Phase" )
        plt.show(block=False)
 
    if plot_all:
        # plot
        fig, axs = plt.subplots( 2, 3, figsize=(18,10), dpi=100 )
        axs = axs.flatten()
 
        plt.sca( axs[0] )
        plt.imshow( np.abs(data).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Input data" )
 
        plt.sca( axs[1] )
        plt.imshow( np.abs(fft).T, 
                   #extent=[ np.min(kx), np.max(kx), np.min(ky), np.max(ky) ],
                   cmap="jet", origin="lower", norm=mpl.colors.LogNorm(), aspect='equal'
                )
        plt.colorbar()
        plt.title( "FFT" )
        #plt.xlim( [cx-1.1*rout, cx+1.1*rout] )
        #plt.ylim( [cy-1.1*rout, cy+1.1*rout] )
        plt.sca( axs[2] )
        plt.imshow( np.abs(fft).T, cmap="jet", origin="lower", norm=mpl.colors.LogNorm() )
        plt.colorbar()
        plt.title( "FFT Magnified" )
        plt.xlim( [cx+crop_shift[0]-1.1*rout, cx+crop_shift[0]+1.1*rout] )
        plt.ylim( [cy+crop_shift[1]-1.1*rout, cy+crop_shift[1]+1.1*rout] )
        for ax in axs[1:3]:
            ax.scatter( cx+crop_shift[0], cy+crop_shift[1], c='k', s=10 )
            circle1 = mpl.patches.Ellipse([cx+crop_shift[0], cy+crop_shift[1]], A, B, angle=angle_deg, fill=False, color='k', ls='--', lw=0.5)
            circle2 = mpl.patches.Ellipse([cx+crop_shift[0], cy+crop_shift[1]], A+thickness, B+thickness, angle=angle_deg, fill=False, color='k', ls='--', lw=0.5)
            ax.add_artist(circle1)
            ax.add_artist(circle2)
        plt.scatter( cx+crop_shift[0], cy+crop_shift[1], c='k', s=10 ) 
        plt.sca( axs[3] )
        plt.imshow( np.abs(data_proc).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Intensity" )
        plt.sca( axs[4] )
        plt.imshow( np.angle(data_proc).T, cmap="hsv", origin="lower", vmin=-np.pi, vmax=np.pi, alpha=np.abs(data_proc).T/np.max(np.abs(data_proc)) )
        plt.colorbar()
        plt.title( "Reconstructed Phase" )
        plt.sca( axs[5] )
        plt.imshow( np.angle(data_proc).T*np.sqrt(np.abs(data_proc)), cmap="seismic", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Field" )
        plt.show(block=False)
 
    return data_proc

def reconstruct_data_gaussfilt(file, cx, cy, rin, rout, plot_all=False, plot_reconstruction=False):
    """
    Reconstructs the electric field using a Gaussian mask in Fourier space.

    Parameters
    ----------
    file : str | Path
        Path to the data file.
    cx : int
        Center x-position in Fourier space.
    cy : int
        Center y-position in Fourier space.
    rin : float
        Inner radius of the mask in Fourier space.
    rout : float
        Outer radius of the mask in Fourier space.
    plot_all : bool, optional (default=False)
        If True, plot all reconstruction steps.
    plot_reconstruction : bool, optional (default=False)
        If True, plot the reconstructed phase and amplitude.

    Returns
    -------
    data_proc : np.ndarray
        The reconstructed complex field.
    """
    # parameters
        # file: path to the data file
        # cx, cy: center of the signal in fourier space
        # rin, rout: inner and outer radius of the mask in fourier space
        # plot_all: plot all the steps of the reconstruction
        # plot_reconstruction: plot the reconstructed data
 
    # read data
    data = np.load(file)
    nx, ny = data.shape
    X, Y = np.meshgrid( np.arange(nx), np.arange(ny), indexing="ij" )
    L = np.min( [ np.abs( nx-cx ), np.abs( cx ), np.abs( ny-cy ), np.abs( cy ) ] )
 
    # fft
    fft = np.fft.fftshift( np.fft.fft2(data) )
 
    # process fft
    fft_proc = fft.copy()
    r = np.sqrt( (X-cx)**2 + (Y-cy)**2 )
    mask = np.exp(-((r-np.mean([rout,rin]))**2)/10**2)
    fft_proc = np.multiply(fft,mask)
    
 
    # reconstruct data
    data_proc = np.fft.ifft2( np.fft.fftshift( fft_proc ) )
    
    if plot_reconstruction:
        # plot
        fig, axs = plt.subplots( 1, 2, figsize=(12,5), dpi=100 )
        
        plt.sca( axs[0] )
        plt.imshow( np.abs(data_proc).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Intensity" )
 
        plt.sca( axs[1] )
        plt.imshow( np.angle(data_proc).T, cmap="hsv", origin="lower", vmin=-np.pi, vmax=np.pi, alpha=np.abs(data_proc).T/np.max(np.abs(data_proc)) )
        plt.colorbar()
        plt.title( "Reconstructed Phase" )
 
        plt.show()
 
    if plot_all:
        # plot
        fig, axs = plt.subplots( 2, 3, figsize=(18,10), dpi=100 )
        axs = axs.flatten()
 
        plt.sca( axs[0] )
        plt.imshow( np.abs(data).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Input data" )
 
        plt.sca( axs[1] )
        plt.imshow( np.abs(mask).T, cmap="jet", origin="lower")#, norm=mpl.colors.LogNorm() )
        plt.colorbar()
        plt.title( "Fourier mask" )
        plt.xlim( [cx-1.1*rout, cx+1.1*rout] )
        plt.ylim( [cy-1.1*rout, cy+1.1*rout] )
        
        plt.sca( axs[2] )
        plt.imshow( np.abs(fft).T, cmap="jet", origin="lower", norm=mpl.colors.LogNorm() )
        plt.colorbar()
        plt.title( "FFT Magnified" )
        plt.xlim( [cx-1.1*rout, cx+1.1*rout] )
        plt.ylim( [cy-1.1*rout, cy+1.1*rout] )
 
        for ax in axs[1:3]:
            ax.scatter( cx, cy, c='k', s=10 )
            circle1 = plt.Circle( [cx, cy], rin, fill=False, color='k', ls='--', lw=0.5 )
            circle2 = plt.Circle( [cx, cy], rout, fill=False, color='k', ls='--', lw=0.5 )
            ax.add_artist(circle1)
            ax.add_artist(circle2)
 
 
        plt.scatter( cx, cy, c='k', s=10 )
        circle1 = plt.Circle( [cx, cy], rin, fill=False, color='k', ls='--', lw=0.5 )
        circle2 = plt.Circle( [cx, cy], rout, fill=False, color='k', ls='--', lw=0.5 )
        plt.gca().add_artist(circle1)
        plt.gca().add_artist(circle2)
 
 
        plt.sca( axs[3] )
        plt.imshow( np.abs(data_proc).T, cmap="jet", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Intensity" )
 
        plt.sca( axs[4] )
        plt.imshow( np.angle(data_proc).T, cmap="hsv", origin="lower", vmin=-np.pi, vmax=np.pi, alpha=np.abs(data_proc).T/np.max(np.abs(data_proc)) )
        plt.colorbar()
        plt.title( "Reconstructed Phase" )
 
        plt.sca( axs[5] )
        plt.imshow( np.angle(data_proc).T, cmap="seismic", origin="lower" )
        plt.colorbar()
        plt.title( "Reconstructed Field" )
 
        plt.show()
 
    return data_proc



def phaselockField(field):
    """
    Shifts the phase of every image in the input field so that a reference pixel (with max intensity in the first image)
    has the same phase value across all images.

    Parameters
    ----------
    field : np.ndarray
        3D complex array of Fourier transformed measurement data with shape (x, y, n), where n is the number of images.

    Returns
    -------
    nfield : np.ndarray
        3D complex array with the same shape as `field`, phase-locked across the third axis.
    """
    default_phase_index = np.unravel_index(np.argmax(np.abs(field[:,:,0]), axis=None), field[:,:,0].shape)
    default_phase = np.angle(field[*default_phase_index,0])
    # print(f"Default Phase index:{default_phase_index} corresponding phase value: {default_phase}")
    nfield = np.zeros(field.shape, dtype=complex)
    for w in range(nfield.shape[2]):
        dPhi = default_phase - np.angle(field[*default_phase_index, w])
        nfield[:,:,w] = field[:,:,w]*np.exp(dPhi*1j)
        # fig, ax = plt.subplots(1,2)
        # im1 = ax[0].imshow(np.abs(nfield[:,:,w]),cmap = "plasma")
        # ax[0].set_title("Amplitude fft")
        # im2 = ax[1].imshow(np.angle(nfield[:,:,w]), cmap = "hsv")
        # ax[1].set_title("Phase fft")
        # fig.colorbar(im1,shrink = .49)
        # fig.colorbar(im2, shrink = .49)
        # fig.show()
    return nfield

def manualFFT(E_omega:np.ndarray, omegas:np.ndarray, t, applyPhaselock = False):
    """ 
    Does the manual fft of the given hypervolume of reconstructed Phase data E(x,y,omega).
    Each array along the 3rd axis represents the reconstruction E_i for a specific omega_i, which are given in ´omegas´.
    
    :param E_omega: Reconstructed phase data = E(x,y,omega)
    :type E_omega: numpy.ndarray with shape=(x,y, size(omegas))

    :param omegas: sample frequencies for the manual fft (refering to the measurement, e.g. the wavelengths of bandpass-filter)
    :type omegas: numpy.ndarray with shape=(x)

    :param applyPhaselock: Phaselocks each E_i(x,y,omega_i) refereing to one pixel of E_1
    :type applyPhaselock: bool
    """
    if applyPhaselock:
        E_omega = phaselockField(E_omega)
    res = 0+0j
    for i in range(len(omegas)):
        res = res + E_omega[:,:,i] * np.exp(1j * omegas[i] * t)
    return res

def getMotorPos(name)->float|None:
    """
    Extracts the motor position from a filename.

    Parameters
    ----------
    name : str
        Filename containing the motor position.

    Returns
    -------
    motor_pos : float or None
        Motor position in meters, or None if not found.
    """
    match = re.search(r"motorPos\s*(-?\d+)", name)
    motor_pos = None
    if match:
        motor_pos = int(match.group(1))
        motor_pos = motor_pos * MOTOR_STEP_SIZE
    return motor_pos

def calculateTiming(motorPos:float|np.ndarray)->float|np.ndarray:
    """
    Calculates the timing from the motor position.

    Parameters
    ----------
    motorPos : float | np.ndarray
        Motor position(s) in meters.

    Returns
    -------
    timing : float | np.ndarray
        Timing value(s) corresponding to the motor position(s).
    """
    return motorPos/C0*2

def normalizeTimestamps(times:np.ndarray)-> np.ndarray:
    """
    Normalizes timestamps so that the minimum value is zero.

    Parameters
    ----------
    times : np.ndarray
        Array of timestamps.

    Returns
    -------
    normalized_times : np.ndarray
        Array of normalized timestamps.
    """
    return times - (times.min())


def reconstructFromFolder(folderPath:Path, cx, cy, A, B, thickness, angle,
                         crop_pos, removeWrongExposure=True, significance=1.5,
                         shift=None, subtract_ref:Path=None, phaselock=0, plot_rec=False, plot_all=False, plot_exp_hist=False, fft_crop_radius=6):
    """
    Reconstructs the field from all data in a folder and saves the results.
    Uses parameters as in `reconstruct_data`.

    Parameters
    ----------
    folderPath : Path
        Path to the folder containing data files.
    cx, cy, A, B, thickness, angle, crop_pos : see reconstruct_data
        Parameters for the reconstruction.
    removeWrongExposure : bool, optional (default=True)
        If True, removes exposures with double illumination.
    significance : float, optional (default=1.5)
        Significance threshold for outlier removal.
    shift : tuple or None, optional (default=None)
        If not None, must contain shift values for each file ((x_i), (y_i)).
    subtract_ref : Path | np.ndarray | None, optional (default=None)
        Data to subtract from each input.
    phaselock : bool, optional (default=0)
        If True, phase-locks all reconstructed fields.
    plot_rec : bool, optional (default=False)
        If True, plot the reconstructed field.
    plot_all : bool, optional (default=False)
        If True, plot all reconstruction steps.
    plot_exp_hist : bool, optional (default=False)
        If True, plot exposure histogram.
    fft_crop_radius : float, optional (default=6)
        Crop radius for FFT.

    Returns
    -------
    None or np.ndarray
        Returns None if not phase-locking, otherwise returns the phase-locked data.
    """
    
    files = os.listdir(folderPath)
    fileNames = [f[:-4] for f in files if f[-4:] == ".npy"]
    try:
        p = _searchForSTPulseData(folderPath)
    except (FileNotFoundError, FileExistsError):
        stTyped = False
    else:
        stTyped = True
    data = makeVolumeArray(folderPath)

    if removeWrongExposure:
        data, outliers = removeDoubleLitted(data, significance, plot_exp_hist)
        if not stTyped:
            fileNames = np.asarray(fileNames)[outliers]
        else:
            fileNames = [f"{p.stem}_{str(n+1)}" for n in range(len(data.data))]

    dataNew = None
    savePath = folderPath/Path("reconstructionData")
    for i, slice in enumerate(data.data):
        file = fileNames[i]
        print(file)
        if shift is None:
            shift_i = (0,0)
        else:
            shift_i = (shift[0][i],shift[1][i])
        rec = reconstructData(data.data[i,:,:],
                            cx, cy, A, B, thickness, angle,
                            crop_pos, plot_all=plot_all,
                    plot_reconstruction = plot_rec,fft_crop_radius=fft_crop_radius,crop_shift=shift_i,
                    subtract_ref = subtract_ref)
        if i == 0:
                dataNew = np.zeros(shape=(len(fileNames),*rec.shape), dtype=complex)
        dataNew[i,:,:] = rec
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        if not phaselock:
            np.savetxt(savePath/Path(f"{file}_{data.motorPos[i]}_reconstruction_abs.csv"),np.abs(rec), delimiter=";")
            #np.savetxt(savePath/Path(f"{file}_reconstruction_angle.csv"),np.angle(rec), delimiter=";")
        else:
            if i == 0:
                dataNew = np.zeros(shape=(len(fileNames),*rec.shape), dtype=complex)
            dataNew[i,:,:] = rec
        i = i + 1
    if not phaselock:
        STPulseData(dataNew, data.times, data.motorPos).safeToFile(savePath/f"{p.stem}_reconstruction_complex")
        return
    dataNew = phaselockField(dataNew)
    if not os.path.isdir(savePath/Path("phaselocked")):
        os.makedirs(savePath/Path("phaselocked"))
    for j, fileName in enumerate(fileNames):
        rec = dataNew[:,:,j]
        np.savetxt(savePath/Path("phaselocked")/Path(f"{fileName}_{data.motorPos[j]}_reconstruction_phaselocked_abs.csv"),np.abs(rec), delimiter=";")
        #np.savetxt(savePath/Path("phaselocked")/Path(f"{fileName}_reconstruction_phaselocked_angle.csv"),np.angle(rec), delimiter=";")
    STPulseData(data=dataNew, times=data.times, motorPos=data.motorPos).safeToFile(savePath/Path("phaselocked")/Path("STPulseData_reconstruction_phaselocked_complex"))

def fitCircle(
        points:np.ndarray, 
        centerArray:np.ndarray, 
        sigmaFit=1, show_statistics=False
) -> CircleFit:
    """
    Fits a circle to a set of 2D points and estimates the center and radius.

    The function:
      - Uses the maximum of the provided centerArray as the initial center estimate.
      - Computes distances from all points to the estimated center.
      - Filters out outlier points based on the specified sigma range.
      - Calculates the mean and standard deviation of the filtered distances as the radius and its error.
      - Optionally displays fitting statistics and visualizations.

    Parameters
    ----------
    points : np.ndarray
        2D array of shape (2, n) containing the (x, y) coordinates of the points to fit.
    centerArray : np.ndarray
        2D array representing the intensity or likelihood map for the center; the maximum is used as the initial center.
    sigmaFit : float, optional (default=1)
        Number of standard deviations for filtering outlier points in the fit.
    show_statistics : bool, optional (default=False)
        If True, plot the circle fit and histograms.

    Returns
    -------
    circleFit : CircleFit
        CircleFit object containing the estimated center (x, y), radius, and fit error.
    """
    estimated_center_i  = np.asarray(np.flip(np.unravel_index(np.argmax(centerArray),centerArray.shape)))
    # print(f"Center: {estimated_center_i}")
    #fit radius
    distances = np.linalg.norm(points - estimated_center_i.reshape((2,1)), axis=0)
    mean, std =np.nanmean(distances), np.nanstd(distances)
    distances_noOL = distances[((distances>mean-sigmaFit*std) & (distances < mean + sigmaFit*std))]
    # print(distances_noOL)
    if show_statistics:
        plt.figure()
        bins= 30
        plt.hist([distances,distances_noOL],bins)
        plt.title("histogram distances")
        plt.show()
    estimated_radius_i, error = np.nanmean(distances_noOL), np.nanstd(distances_noOL)

    print(f"Estimated Initial Center: {estimated_center_i}")
    print(f"Estimated Initial Radius: {estimated_radius_i}")

    #recursive fit: Not implemented yet
    def rec(points:np.ndarray, center, radius):
        l = points[:,points[0] < center[0]]
        r = points[:,points[0] >= center[0]]
        up = points[:,points[1] < center[1]]
        low = points[:,points[1] >= center[1]]
        plt.figure()
       
        plt.gca().set_aspect("equal")
        plt.show()
        # functions for circle
        y_lower = center[1]+np.sqrt(
            radius**2-np.power(low-center[0],2)
        )
        y_upper = center[1]-np.sqrt(
            radius**2-np.power(up-center[0],2)
        )
        x_right = center[0]+np.sqrt(
            radius**2-np.power(r-center[1],2)
        )
        print(radius**2-np.power(l-center[1],2))
        x_left = center[0]-np.sqrt(
            radius**2-np.power(l-center[1],2)
        )
        # calc difference in y
        dl = np.nanmean(x_left-l,)
        dr = np.nanmean(x_right-r)
        dup = np.nanmean(y_upper-up)
        dlow = np.nanmean(y_lower-low)
        dx = np.nanmean((dl,dr))
        dy = np.nanmean((dup,dlow))
        new_center = np.asarray((center[0]-dx,center[1]-dy))

        plt.figure()
        plt.plot(low,y_lower,marker = "^", label = "lower", ls = "",color="red")
        plt.plot(up,y_upper,marker="v", label = "upper", ls = "", color = "yellow")
        plt.scatter(x_right,r,marker = ">", color = "green")
        plt.scatter(x_left,l,marker = "<", color = "purple")
        plt.scatter(*l, marker = "+", color = "purple")
        plt.scatter(*r, marker = "+", color = "green")
        plt.scatter(*up, marker = "x",color = "yellow")
        plt.scatter(*low, marker = "x", color = "red")
        plt.scatter(*new_center,marker="o", color = "blue")
        plt.gca().set_aspect("equal")
        plt.show
    #rec(points, estimated_center_i, estimated_radius_i)
    estimated_center, estimated_radius = estimated_center_i, estimated_radius_i
    return CircleFit(estimated_center, estimated_radius, error)

def findRadius(
        img: np.ndarray, 
        centerXY: tuple[int, int], 
        filterR_px: int = 30, 
        threshhold: float = 0.8, 
        sigma_Fit: float = 1, 
        showStatistics: bool = False
) -> CircleFit:
    """
    Estimates the radius and center of a circular feature (e.g., a lightspring) in a 2D image.

    The function:
      - Masks out the central region to avoid the 0th order.
      - Selects high-intensity pixels above a threshold.
      - Fits a circle to the selected points, optionally filtering outliers based on distance from the estimated center.
      - Optionally displays fitting statistics and visualizations.

    Parameters
    ----------
    img : np.ndarray
        2D CCD image data, shape (x, y).
    centerXY : tuple[int, int]
        Initial estimate for the center of the circle (x, y).
    filterR_px : int, optional (default=30)
        Radius of the inner mask (in pixels) to exclude the central region.
    threshhold : float, optional (default=0.8)
        Fraction of the maximum intensity to use as the threshold for selecting points.
    sigma_Fit : float, optional (default=1)
        Number of standard deviations for filtering outlier points in the fit.
    showStatistics : bool, optional (default=False)
        If True, plot fitting statistics and visualizations.

    Returns
    -------
    circleFit : CircleFit
        CircleFit object containing the estimated center (x, y), radius (pixels), and fit error (pixels).
    """
    #filter the 0th order
    crop = img.copy()
    if crop.dtype == complex:
        crop = np.abs(crop)
    X, Y = np.meshgrid(np.linspace(1,img.shape[1],img.shape[1]),np.linspace(1,img.shape[0],img.shape[0]))
    c_mask = np.sqrt(np.power(X-centerXY[0],2)+np.power(Y-centerXY[1],2))<=filterR_px
    crop[c_mask] = 0
    c_array = img.copy()
    c_array[~c_mask] = 0
    max_mask = crop > threshhold*np.nanmax(crop)
    max_points = np.indices(crop.shape)
    max_points = np.asarray(((max_points[1][max_mask]),(max_points[0][max_mask])))
    # print(max_points)
    crop[~max_mask] = 0
    circleFit = fitCircle(max_points.reshape(2,(max_points.shape[1])),c_array,sigma_Fit, showStatistics)
    if showStatistics:
        plt.figure()
        plt.imshow(crop, origin = "lower")
        plt.scatter(*max_points, marker="x", color = "red")
        plt.scatter(*circleFit.center)
        c = plt.Circle(circleFit.center,circleFit.radius, color = "green", ls = ":",fill=False)
        ax = plt.gca()
        ax.add_patch(c)
        ax.add_patch(plt.Circle(circleFit.center, filterR_px, color = "yellow", ls = ":",fill=False))
        #rad = plt.Circle(estimated_center)
        plt.show()
    return circleFit

def calculateRadius(
        circle: CircleFit, 
        magnification=1, 
        pixelSize_mm=3.45e-3
) -> CircleFit:
    """
    Converts a fitted circle's radius and error from pixels to meters.

    Parameters
    ----------
    circle : CircleFit
        CircleFit object containing center coordinates, radius (in pixels), and error (in pixels).
    magnification : float, optional
        Magnification factor for converting pixels to real-world units (default: 1).
    pixelSize_mm : float, optional
        Pixel size in millimeters (default: 3.45e-3 mm).

    Returns
    -------
    circle_meters : CircleFit
        CircleFit object with center unchanged, but radius and error converted to meters.
    """
    circleMeters = CircleFit(circle.center, circle.radius*magnification*pixelSize_mm*1e-3, circle.error*magnification*pixelSize_mm*1e-3)
    return circleMeters

def getCurrentAngleLS(
        lsSLice:np.ndarray, 
        threshhold, 
        center:tuple[int,int], 
        maskRadius:int, 
        showPlot=False, 
        showFit=False
) -> tuple[float, float, float]:
    """
    Calculates the angle to the x-axis (counterclockwise) of a lightspring slice.

    Parameters
    ----------
    lsSLice : np.ndarray
        Single temporal slice of reconstruction data, shape (n, n).
    threshhold : float
        Threshold for intensity values considered for the fit.
    center : tuple[int, int]
        Center of the lightspring.
    maskRadius : int
        Radius of the center mask.
    showPlot : bool, optional (default=False)
        If True, show fitting statistics.
    showFit : bool, optional (default=False)
        If True, show fitting plot.

    Returns
    -------
    max_angle : float
        Angle at maximum intensity.
    weighted_squared_angle : float
        Angle at squared weighted average.
    mass_point_angle : float
        Angle at center of mass (linear weighted average).
    """
    x,y = np.meshgrid(np.linspace(0,lsSLice.shape[0]-1,lsSLice.shape[0]), 
                      np.linspace(0,lsSLice.shape[1]-1,lsSLice.shape[1]))
    if np.isnan(maskRadius):
        mask = np.zeros(lsSLice.shape,dtype=bool)
    else:
        mask = np.ones(lsSLice.shape,dtype=bool)
        mask[np.sqrt(np.power(x-center[0],2)+np.power(y-center[1],2))<=maskRadius] = 0
    if type(center) == np.ndarray:
        center = tuple(center)

    sliceMasked = np.abs(np.copy(lsSLice))
    sliceMasked[~mask] = 0
    cMax =np.max(sliceMasked)
    indexVals = np.unravel_index(np.where(sliceMasked > cMax*threshhold),mask.shape)[1] 
    #consider to unite with the circle-radius fit to sort out points with higher radius
    if indexVals.size == 0:
        return np.nan, np.nan, np.nan
    def getAngle(point):
        #return angle e [0,2pi]
        dY=x[point]-x[center]
        dX=y[point]-y[center]
        angle = np.arctan2(dY,dX)
        #map so that angle[0,2pi]
        if angle < 0:
            angle = np.interp(angle,[-np.pi,0],[np.pi,2*np.pi])
        return angle
   
    #get weighted average of these points
    nVals = np.ravel(sliceMasked)[np.ravel_multi_index(indexVals,sliceMasked.shape)]
    nVals = np.interp(
        nVals,
        [nVals.min(), nVals.max()],
        [0,1]
    )
    nVals2 = np.power(nVals,2)
    if showPlot:
        plt.figure()
        plt.hist(nVals, bins=10)
        plt.title("nVals")

    massPoint = (round(np.average(indexVals[1], weights=nVals)), round(np.average(indexVals[0],weights=nVals)))
    weightedSquared = (round(np.average(indexVals[1], weights=nVals2)), round(np.average(indexVals[0],weights=nVals2)))
    maxPoint =  np.unravel_index(np.flip(np.where(sliceMasked == np.max(sliceMasked))), mask.shape)[1].flatten()
    maxPoint = (maxPoint[0],maxPoint[1])
    if showFit:
        plt.figure()
        plt.title("fit lightspring intensity")
        plt.imshow(np.abs(sliceMasked),origin="lower")
        plt.scatter(indexVals[1],indexVals[0],marker="x", color ="r")
        plt.scatter(*massPoint,marker="+", color = "y", label = f"weighted squared ({round(getAngle(massPoint)*(180/np.pi),1)}°)")
        plt.scatter(*maxPoint, marker="+", color = "lime", label = f"maximum ({round(getAngle(maxPoint)*(180/np.pi),2)}°)")
        plt.colorbar()
        plt.legend()
        plt.show()
    return getAngle(maxPoint), getAngle(weightedSquared), getAngle(massPoint)

def calculateLOGV_theory(
        radius_m:float, 
        radiusError_m:float, 
        dOmega=2*np.pi*C0/778e-9 - 2*np.pi*C0/806e-9, 
        dl=7
) -> tuple[float, float]:
    """
    Calculates the theoretical LOGV (Linear Optical Group Velocity).

    Parameters
    ----------
    radius : float
        Radius in meters.
    dOmega : float, optional
        Difference in angular frequency.
    dl : float, optional
        Difference in wavelength.

    Returns
    -------
    logv : float
        Theoretical LOGV value.
    logv_error : float
        Theoretical LOGV value.
    """
    return radius_m*(dOmega/dl), radiusError_m*(dOmega/dl)

def makeVolumeArray(data) -> STPulseData | np.ndarray:
    """
    Combines a list of n equal-sized ndarrays of shape (x, y) into one array of shape (n, x, y).

    Parameters
    ----------
    data : Path | list[np.ndarray]
        Path to data folder or list of arrays to combine.

    Returns
    -------
    volume : np.ndarray
        3D data array of shape (n, x, y).
    times : np.ndarray, optional
        If data is a Path, also returns timings as a 1D array.
    """
    def fuseArrays(aList):
        return np.stack(aList)
    if isinstance(data, (list, tuple)):
        if isinstance(data[0], np.ndarray):
            return fuseArrays(data)
    elif isinstance(data, (Path)):
        try:
            path = _searchForSTPulseData(data)
            print(f"STPulseData found at\n{path}\nLoading...")
            return loadSTPulseData(path)
        except (FileNotFoundError, FileExistsError) as e:
            d = []
            times = []
            motorPos = []
            files = sorted([f for f in os.listdir(data) if f[-4:] == ".npy"], key = lambda x: getMotorPos(x), reverse=True)
            for file in files:
                d.append(np.load(data/file))
                motorPos.append(getMotorPos(file)) if getMotorPos(file) is not None else motorPos.append(np.nan)
                times.append(calculateTiming(getMotorPos(file)))
            return STPulseData(fuseArrays(d), np.asarray(times), np.asarray(motorPos))
    
def removeDoubleLitted(
        data: np.ndarray | STPulseData,
        significance=1.5, 
        plot=False
) -> tuple[np.ndarray | STPulseData, np.ndarray]:
    """
    Removes exposures with double illumination (double-litted) based on summed intensity outliers.

    For each exposure (slice), the total intensity is summed. Exposures with summed intensity significantly above the median (by a factor of `significance`) are considered double-illuminated and removed.

    Parameters
    ----------
    data : np.ndarray or STPulseData
        3D data array of shape (n, x, y) or STPulseData object containing such an array.
    significance : float, optional (default=1.5)
        Threshold factor: exposures with summed intensity > significance × median are removed.
    plot : bool, optional (default=False)
        If True, plot a histogram of summed intensities.

    Returns
    -------
    filtered_data : np.ndarray or STPulseData
        Data with double-illuminated exposures removed.
    mask : np.ndarray
        Boolean mask indicating which exposures were kept (True = kept, False = removed).

    Raises
    ------
    TypeError
        If input is not a numpy ndarray or STPulseData object.
    ValueError
        If input data is not 3D.
    """
    if isinstance(data, STPulseData):
        dataP = data.data
    elif isinstance(data, np.ndarray):
        dataP = data
    else:
        raise TypeError("Input data must be a numpy ndarray or STPulseData object")
    if dataP.ndim != 3:
        raise ValueError("Input data must be a 3D array of shape (n, x, y)")
    summed = np.sum(np.sum(dataP,1),1)
    if plot:
        plt.figure()
        plt.hist(summed, bins = 20)
        plt.show()
    mask = summed > significance*np.median(summed)
    if isinstance(data, STPulseData):
        return _cropSTPulseData(data, ~mask), ~mask
    return data[~mask], ~mask

def cropByTimestamp(data:STPulseData, tMin:float, tMax:float)->STPulseData:
    """
    Crops the STPulseData object to include only slices with timestamps within the specified range.

    Parameters
    ----------
    data : STPulseData
        Data object containing a 3D array (n, x, y) and corresponding timestamps.
    tMin : float
        Minimum timestamp to keep.
    tMax : float
        Maximum timestamp to keep.

    Returns
    -------
    cropped_data : STPulseData
        New STPulseData object containing only the slices and metadata where tMin <= time <= tMax.
    """
    index = np.where((data.times<=tMax)&(data.times>=tMin))
    return _cropSTPulseData(data, index)

def makeIsosurfaceMask(
        data: np.ndarray | STPulseData, 
        level: float, 
        epsilon: float = 1
) -> np.ndarray:
    """
    Creates a boolean mask that identifies voxels within a specified intensity range around a relative isosurface level in a 3D volume.

    If a STPulseData object is provided, the mask is computed from its `.data` attribute.

    Parameters
    ----------
    data : np.ndarray or STPulseData
        3D volume array of shape (n, x, y) or STPulseData object containing such an array.
    level : float
        Relative isosurface level as a fraction of the maximum value [0, 1].
    epsilon : float, optional (default=1)
        Tolerance range around the isosurface level. Voxels with |value - threshold| < epsilon are included.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of the same shape as the input data, where True indicates voxels near the specified isosurface level.

    Raises
    ------
    TypeError
        If `data` is not a numpy ndarray or STPulseData object.
    ValueError
        If `data` is not 3D or if `level` is not in [0, 1].
    """
    if type(data) is STPulseData:
        data = data.data
    if not isinstance(data, np.ndarray):
        raise TypeError("Parameter `data` must be a numpy ndarray or STPulseData object")
    if data.ndim != 3:
        raise ValueError("Parameter `data` must be a 3D array of shape (n, x, y)")
    if not (0 <= level <= 1):
        raise ValueError("Parameter `level` must be in the range [0, 1]")

    threshold = data.max() * level
    mask = np.abs(data - threshold) < epsilon
    return mask

def fitAngleData(data:np.ndarray, times:np.ndarray) -> tuple[float, float, RLMResults]:
    """
    Fits the angular data to a robust linear model (y = m*x + b) using statsmodels RLM.

    This function unwraps the angle data, removes NaN values, and fits a robust linear model to angle vs. time.
    It returns the slope (rate of angular change), intercept, and the fitted model object.

    Parameters
    ----------
    data : STPulseData
        Data object containing:
            - data: 1D array of angle values (radians), shape (t,)
            - times: 1D array of timestamps, shape (t,)

    Returns
    -------
    slope : float
        Slope of the robust linear fit (angular velocity).
    intercept : float
        Intercept of the robust linear fit.
    model_fit : statsmodels.robust.robust_linear_model.RLMResults
        The fitted robust linear model object.
    """
    x = times
    y = np.unwrap(data)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    x_with_const = sm.add_constant(x)
    rlm_model = sm.RLM(y, x_with_const)
    rlm_results = rlm_model.fit()
    intercept, slope = rlm_results.params

    return slope, intercept, rlm_results

def getRadialIntensity(
        data: STPulseData,
        center: tuple[int, int], 
        cropRadius: int, 
) -> np.ndarray:
    """
    Calculates the maximum intensity within a circular region for each slice of a 3D dataset.

    For each time slice (2D image), this function:
      - Applies a circular mask centered at the specified coordinates with the given radius.
      - Computes the maximum intensity value within the masked region.
      - Optionally plots the masked intensity profile for each slice.

    Parameters
    ----------
    data : STPulseData
        Data object containing a 3D array (n, x, y), where n is the number of slices.
    center : tuple[int, int]
        Center coordinates (x, y) of the circular mask.
    cropRadius : int
        Radius of the circular mask in pixels.

    Returns
    -------
    intensityMax : np.ndarray
        1D array of shape (n,), where each element is the maximal intensity within the mask for one slice.
    """
    n = data.data.shape[0]
    intensityMax = np.zeros(n)
    for i in range(n):
        x, y = np.meshgrid(
            np.linspace(0, data.data.shape[1] - 1, data.data.shape[1]),
            np.linspace(0, data.data.shape[2] - 1, data.data.shape[2])
        )
        mask = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= cropRadius
        radial_profile = data.data[i].copy()
        radial_profile[~mask] = 0
        maxI = np.max(radial_profile)
        intensityMax[i] = maxI
    return intensityMax

def cropByIntensity(
        data: np.ndarray|STPulseData, 
        intensity: np.ndarray, 
        threshhold = 0.5
) -> tuple[np.ndarray|STPulseData, np.ndarray]:
    """
    Returns only those slices of the data, timestamps, and intensity values where the intensity exceeds a specified fraction of the maximum intensity.

    Parameters
    ----------
    data : STPulseData
        Data object containing a 3D array (n, x, y) and corresponding timestamps.
    intensity : np.ndarray
        1D array of intensity values, shape (n,).
    threshhold : float, optional
        Fraction of the maximum intensity to use as the cutoff (default: 0.5).

    Returns
    -------
    cropped_data : STPulseData
        Data, times, and motor positions for slices where intensity > threshhold * max(intensity).
    cropped_intensity : np.ndarray
        Intensity values for the selected slices.
    """
    maxI = np.max(intensity)
    index = np.where(intensity > maxI*threshhold)
    if isinstance(data, STPulseData):
        croppedData = _cropSTPulseData(data, index)
    else:
        croppedData = data[index]
    return croppedData, intensity[index]

def fitAnglesAndIntensity(
    data: STPulseData,
    center: tuple[int, int],
    cropRadius: int,
    threshhold: float = 0.8,
    crop: bool = True,
    intensityThreshhold:float = 0.5
) -> tuple[sm.robust.robust_linear_model.RLMResults, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Calculates the angular position and intensity for each time slice in a 3D dataset, and fits the angle data to a robust linear model.

    For each slice:
      - Computes the angular position using getCurrentAngleLS (squared weighted method).
      - Calculates the maximum radial intensity using getRadialIntensity.
      - Optionally filters out slices with intensity below a threshold.
      - Fits the resulting angle vs. time data using a robust linear model (statsmodels RLM).

    Parameters
    ----------
    data : STPulseData
        Data object containing a 3D array (n, x, y) and corresponding timestamps.
    center : tuple[int, int]
        Center coordinates (x, y) for angle and intensity calculation.
    cropRadius : int
        Mask radius in pixels for angle and intensity calculation.
    threshhold : float, optional
        Threshold for angle calculation (passed to getCurrentAngleLS, default: 0.8).
    crop : bool, optional
        If True, only slices with intensity above the specified threshold are used for fitting (default: True).
    intensityThreshhold : float, optional
        Intensity threshold for filtering slices (default: 0.5).

    Returns
    -------
    angleFit : statsmodels.robust.robust_linear_model.RLMResults
        Robust linear model fit result for angle vs. time.
    intensities : np.ndarray
        1D array of maximum intensities for each slice, shape (n,).
    angles : np.ndarray
        1D array of calculated angles (squared weighted) for each slice, shape (n,).
    times (optional) : np.ndarray
        Returned only if cropping is enabled; filtered timestamps.
    """
    n = data.data.shape[0]
    angles = np.nan*np.zeros(n)

    for i in range(n):
        angles[i] = getCurrentAngleLS(data.data[i], threshhold, center, cropRadius)[1]
    # Get intensity profile for each slice
    intensities = getRadialIntensity(data, center, cropRadius)
    angles = np.array(angles)
    if crop:
        angleStruct = STPulseData(angles, data.times, data.motorPositions)
        angles, intensities = cropByIntensity(angleStruct, intensities, intensityThreshhold) 
    # Fit angles vs times
        angleFit = fitAngleData(angles.data, angles.times)[2]
    else:
        angleFit = fitAngleData(angles, data.times)[2]
    # If cropping is enabled, return the filtered angles and times
    if crop:
        return angleFit, intensities, angles.data, angles.times
    return angleFit, intensities, angles, None


def getOGV(lightspringData:STPulseData, estimatedCenter:tuple[int,int], maskRadius:int, magnification=1,
           threshhold=0.8, sigma_Fit=1, filterbyIntensity=False, intensityThreshhold=0.5,
           showStatistics=False, showFit=False, angleFittingAlgo=1)-> OGVMeasurement:
    """
    Analyze a 3D lightspring dataset to extract the Optical Group Velocity (OGV) and related physical parameters.

    For each time slice, this function:
      - Fits a circle to estimate the lightspring's radius and center.
      - Calculates the angular position of the lightspring using the specified algorithm.
      - Converts the radius from pixels to meters using the given magnification.
      - Optionally filters slices by intensity for robust fitting.
      - Fits the angular position as a function of time to determine the OGV.

    Parameters
    ----------
    lightspringData : STPulseData
        Object containing a 3D array (t, x, y) and corresponding timestamps.
    estimatedCenter : tuple[int, int]
        Initial estimate for the lightspring center (x, y).
    maskRadius : int
        Radius for the circular mask used in calculations (pixels).
    magnification : float, optional
        Magnification factor for converting pixels to meters (default: 1).
    threshhold : float, optional
        Intensity threshold for feature detection (default: 0.8).
    sigma_Fit : float, optional
        Sigma range for radius fitting (default: 1).
    filterbyIntensity : bool, optional
        If True, only use slices with intensity above a threshold for fitting.
    intensityThreshhold : float, optional
        Intensity threshold for filtering slices (default: 0.5).
    showStatistics : bool, optional
        If True, display fitting statistics.
    showFit : bool, optional
        If True, plot fit results for each slice.
    angleFittingAlgo : int, optional
        Algorithm for angle calculation: 0=max, 1=squared weighted, 2=center of mass (default: 1).

    Returns
    -------
    OGVMeasurement
        Data class containing:
            OGV (float): Fitted optical gauge velocity (rad/m).
            OGVError (float): Error of the OGV fit.
            LOGV (float): Local optical gauge velocity (OGV × average radius).
            LOGVError (float): Error of the LOGV value.
            avgRadius (float): Average radius (meters).
            avgRadiusError (float): Error of the average radius.
            radii (np.ndarray): Radii for each slice (meters).
            radiiError (np.ndarray): Error for each radius (meters).
            anglesMeassured (np.ndarray): Angular positions for each slice (radians).
            intensities (np.ndarray): Maximum intensity for each slice.
            times (np.ndarray): Timestamps used for fitting.
            angleFit (statsmodels result): Fitted angle vs. time model.
            angles (np.ndarray): Angles used for fitting (radians).
    """
    data = np.copy(lightspringData.data)
    times = np.copy(lightspringData.times)
    meassured = np.zeros((data.shape[0],2)) # axis1: 0: r, 1: angle
    c_errors = np.zeros((data.shape[0]))
    #fitting the rawdata
    for i in range(meassured.shape[0]):
        fittedCircle = findRadius(data[i,:,:], estimatedCenter, maskRadius, threshhold, sigma_Fit, showStatistics)
        angle = getCurrentAngleLS(data[i,:,:], threshhold, fittedCircle.center, fittedCircle.radius*0.65, showStatistics, showFit)[angleFittingAlgo]
        realCircle = calculateRadius(fittedCircle, magnification=magnification)
        meassured[i,0]=realCircle.radius
        c_errors[i]=realCircle.error
        meassured[i,1]=angle
    #process data
    avgRadius = np.nanmean(meassured[:,0])
    avgCircleError = np.nanmean(c_errors)
    anglefit, intensities, angles, timesFit = fitAnglesAndIntensity(lightspringData, estimatedCenter, maskRadius, threshhold, crop=filterbyIntensity, intensityThreshhold=intensityThreshhold)
    if filterbyIntensity:
        times = timesFit
    #calculate OGV
    ogv = anglefit.params[1]
    ogv_error = anglefit.bse[1]
    logv = ogv*avgRadius
    logv_error = np.sqrt((ogv*avgCircleError)**2 + (avgRadius*ogv_error)**2)
    return OGVMeasurement(ogv, ogv_error, logv, logv_error, 
                          avgRadius, avgCircleError, 
                          meassured[:,0], c_errors,
                          meassured[:,1], intensities, times, anglefit, angles)


def extractParametersFromFilename(filename: str) -> dict:
    """
    Extracts parameters from a filename formatted as 'parameter1_value1_parameter2_value2...'.

    Parameters
    ----------
    filename : str
        The filename from which to extract parameters.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and their corresponding values.
    """
    parameters = {}
    params = {}
    parts = filename.split('_')
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            params[parts[i]] = parts[i + 1]
    return params

def convertFolderDataToSTPulseData(
        folderPath: Path, 
        deleteOld: bool = False
) -> None:
    """
    Reconstructs data from a folder containing .npy files and saves the results.

    Parameters
    ----------
    folderPath : Path
        Path to the folder containing .npy files.
    deleteOld : bool, optional
        If True, deletes the original .npy files after reconstruction (default: False).
    Raises
    ------
    FileNotFoundError
        If no .npy files are found in the specified folder.
    """
    files = os.listdir(folderPath)
    files = [f for f in files if f.endswith(".npy")]
    if files == []:
        raise FileNotFoundError(f"No .npy files found in {folderPath}.")
    date = os.path.getmtime(folderPath/files[0])
    date = time.strftime("%y%m%d", time.gmtime(date))
    params = ParameterKeys.getAllParams(files[0])
    if ParameterKeys.MOTOR_POSITION in params:
        del params[ParameterKeys.MOTOR_POSITION]
    params = ParameterKeys.dictToName(params)
    data = makeVolumeArray(folderPath)
    data.safeToFile(folderPath/("STPulseData_"+params+"_"+f"{ParameterKeys.N_SLICES}={data.data.shape[0]}"+"_"+date))
    if deleteOld:
        for file in files:
            os.remove(folderPath/file)

def loadSTPulseData(path) -> STPulseData:
    """
    Loads a STPulseData object from a specified path.

    Parameters
    ----------
    path : Path
        Path to the STPulseData file.

    Returns
    -------
    STPulseData
        Loaded STPulseData object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    return STPulseData.loadFromFile(path)