"""
Package for processing spacetime-beams related data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import re
from .Visualizing import showImage
import statsmodels.api as sm

MOTOR_STEP_SIZE = 50e-9 #m
C0 = 299792458  #m/s
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
        plt.imshow( np.angle(data_proc).T, cmap="seismic", origin="lower" )
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

def getMotorPos(name):
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

def calculateTiming(motorPos):
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

def normalizeTimestamps(times:np.ndarray):
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
    files = [f for f in files if f[-4:] == ".npy"]
    fileNames = [f[:-4] for f in files]
    minPos = max(getMotorPos(x) for x in files) #max() because its going further in -x direcion
    rawData, times = makeVolumeArray(folderPath)
    if removeWrongExposure:
        rawData, outliers = removeDoubleLitted(rawData, significance, plot_exp_hist)
        fileNames = np.asarray(fileNames)[outliers]
        files = np.asarray(files)[outliers]
    data = None
    for i, file in enumerate(fileNames):
        print(file)
        if shift is None:
            shift_i = (0,0)
        else:
            shift_i = (shift[0][i],shift[1][i])
        rec = reconstructData(rawData[i,:,:],
                            cx, cy, A, B, thickness, angle,
                            crop_pos, plot_all=plot_all,
                    plot_reconstruction = plot_rec,fft_crop_radius=fft_crop_radius,crop_shift=shift_i,
                    subtract_ref = subtract_ref)
        savePath = folderPath/Path("reconstructionData")
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        if not phaselock:
            np.save(savePath/Path(file+"_reconstruction_complex"),rec)
            np.savetxt(savePath/Path(f"{file}_reconstruction_abs.csv"),np.abs(rec), delimiter=";")
            np.savetxt(savePath/Path(f"{file}_reconstruction_angle.csv"),np.angle(rec), delimiter=";")
        else:
            if i == 0:
                data = np.zeros(shape=(*rec.shape,len(files)), dtype=complex)
            data[:,:,i] = rec
        i = i +1
    if not phaselock:
        return
    data = phaselockField(data)
    if not os.path.isdir(savePath/Path("phaselocked")):
        os.makedirs(savePath/Path("phaselocked"))
    for j, fileName in enumerate(fileNames):
        rec = data[:,:,j]
        np.save(savePath/Path("phaselocked")/Path(fileName+"_reconstruction_phaselocked_complex"),rec)
        np.savetxt(savePath/Path("phaselocked")/Path(f"{fileName}_reconstruction_phaselocked_abs.csv"),np.abs(rec), delimiter=";")
        np.savetxt(savePath/Path("phaselocked")/Path(f"{fileName}_reconstruction_phaselocked_angle.csv"),np.angle(rec), delimiter=";")

def fitCircle(points:np.ndarray, cArray:np.ndarray, sigmaFit=1, show_statistics=False) -> tuple[tuple[int, int], float]:
    """
    Fits a circle to circularly distributed data points and returns the center and radius.

    Parameters
    ----------
    points : np.ndarray
        Data points of shape (2, n).
    cArray : np.ndarray
        Filtered data array, where only points considered for the fit have values > 0, shape (x, y).
    sigmaFit : float, optional (default=1)
        Range of data points considered for the fit in sigma of the distance distribution.
    show_statistics : bool, optional (default=False)
        If True, plot the circle fit and histograms.

    Returns
    -------
    estimated_center : tuple[int, int]
        Estimated center coordinates (x, y).
    estimated_radius : float
        Estimated radius.
    """
    estimated_center_i  = np.asarray(np.flip(np.unravel_index(np.argmax(cArray),cArray.shape)))
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
    estimated_radius_i = np.nanmean(distances_noOL)

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
    return estimated_center, estimated_radius

def findRadius(img: np.ndarray, centerXY: tuple[int, int], filterR_px: int = 30, threshhold: float = 0.8, sigma_Fit: float = 1, showStatistics: bool = False):
    """
    Finds the central radius of the lightspring from a CCD image.

    Parameters
    ----------
    img : np.ndarray
        CCD image data, shape (x, y).
    centerXY : tuple[int, int]
        Estimated center of the circle (x, y).
    filterR_px : int, optional (default=30)
        Radius of the inner circle filter in pixels.
    threshhold : float, optional (default=0.8)
        Threshold for data to be considered for the fit.
    sigma_Fit : float, optional (default=1)
        Range of data points considered for the fit in sigma of the distance to the estimated center.
    showStatistics : bool, optional (default=False)
        If True, plot fitting statistics.

    Returns
    -------
    radius : float
        Estimated radius in pixels.
    center : tuple[int, int]
        Estimated center in pixels.
    """
    #filter the 0th order
    crop = img.copy()
    if crop.dtype == complex:
        crop = np.abs(crop)
    X, Y = np.meshgrid(np.linspace(1,img.shape[1],img.shape[1]),np.linspace(1,img.shape[0],img.shape[0]))
    # print(img.shape)
    # print(X.shape)
    c_mask = np.sqrt(np.power(X-centerXY[0],2)+np.power(Y-centerXY[1],2))<=filterR_px
    crop[c_mask] = 0
    # plt.figure()
    # plt.imshow(crop)
    # plt.show()
    c_array = img.copy()
    c_array[~c_mask] = 0
    max_mask = crop > threshhold*np.nanmax(crop)
    max_points = np.indices(crop.shape)
    max_points = np.asarray(((max_points[1][max_mask]),(max_points[0][max_mask])))
    # print(max_points)
    maxima = crop[max_mask]
    crop[~max_mask] = 0
    center, radius = fitCircle(max_points.reshape(2,(max_points.shape[1])),c_array,sigma_Fit, showStatistics)
    if showStatistics:
        plt.figure()
        plt.imshow(crop, origin = "lower")
        plt.scatter(*max_points, marker="x", color = "red")
        plt.scatter(*center)
        c = plt.Circle(center,radius, color = "green", ls = ":",fill=False)
        ax = plt.gca()
        ax.add_patch(c)
        ax.add_patch(plt.Circle(center, filterR_px, color = "yellow", ls = ":",fill=False))
        #rad = plt.Circle(estimated_center)
        plt.show()
    return radius, center

def calculateRadius(radius_px, pixelSize_mm=3.45e-3):
    """
    Calculates the radius in meters from pixels.

    Parameters
    ----------
    radius_px : float
        Radius in pixels.
    pixelSize_mm : float, optional (default=3.45e-3)
        Pixel size in millimeters.

    Returns
    -------
    radius_m : float
        Radius in meters.
    """
    return radius_px*pixelSize_mm*1e-3

def getCurrentAngleLS(lsSLice:np.ndarray, threshhold, center:tuple[int,int], maskRadius:int, showPlot=False, showFit=False) -> tuple[float, float]:
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
    return (getAngle(maxPoint), getAngle(weightedSquared), getAngle(massPoint))

def getOGV(lightspringData:np.ndarray, times:np.ndarray, estimatedCenter:tuple[int,int], maskRadius:int,
           threshhold=0.8, sigma_Fit=1, fromIsosurface=False, isoLevel=.5, isoTolerance=10,
           showStatistics=False, showFit=False, angleFittingAlgo=1):
    """
    Calculates the OGV (Optical Gauge Velocity) and radius for each slice of the `lightspringData`, as well as the
    average OGV and average radius.

    Parameters
    ----------
    lightspringData : np.ndarray
        3D array representing the light spring data with shape (t, x, x).
    times : np.ndarray
        1D array of timestamps for each time slice.
    estimatedCenter : tuple[int, int]
        Coordinates (x, y) of the estimated center.
    maskRadius : int
        Radius of the circular mask.
    threshhold : float, optional (default=0.8)
        Threshold for data filtering.
    sigma_Fit : float, optional (default=1.0)
        Sigma range for radius fitting.
    fromIsosurface : bool, optional (default=False)
        If True, use isosurface mask for angle calculation.
    isoLevel : float, optional (default=0.5)
        Relative isosurface level.
    isoTolerance : float, optional (default=10)
        Tolerance for isosurface level.
    showStatistics : bool, optional (default=False)
        If True, plot fitting statistics.
    showFit : bool, optional (default=False)
        If True, plot fit for each slice.
    angleFittingAlgo : int, optional (default=1)
        Algorithm for angle calculation (0=max, 1=squared weighted, 2=center of mass).

    Returns
    -------
    averageOGV : float
        Average optical gauge velocity [rad/m].
    averageRadius : float
        Average radius [m].
    OGV_array : np.ndarray
        OGV for each slice [rad/m].
    radius_array : np.ndarray
        Radius for each slice [m].
    angle_array : np.ndarray
        Angular position for each slice [rad].
    """
    data = np.copy(lightspringData)
    if fromIsosurface:
        mask = makeIsosurfaceMask(lightspringData, isoLevel, isoTolerance)
        data[~mask] = 0
        angleFittingAlgo = 2 #center of mass
        m = np.any(mask,axis= (1,2))
    meassured = np.zeros((data.shape[0],2)) # axis1: 0: r, 1: angle
    for i in range(meassured.shape[0]):
        # if fromIsosurface and not m[i]:
        #     meassured[i,0]=np.nan
        #     meassured[i,1]=np.nan
        #     continue
        r, center = findRadius(data[i,:,:], estimatedCenter, maskRadius, threshhold, sigma_Fit, showStatistics)
        angle = getCurrentAngleLS(data[i,:,:], threshhold, center, r*0.65, showStatistics, showFit)[angleFittingAlgo]
        r = calculateRadius(r)
        meassured[i,0]=r
        meassured[i,1]=angle
    #put fit here instead of gradient
    ogv = np.gradient(meassured[:,1], times)
    avgOGV = np.nanmean(ogv)
    avgRadius = np.nanmean(meassured[:,0])
    return avgOGV, avgRadius, ogv, meassured[:,0], meassured[:,1]

def calculateLOGV_theory(radius, dOmega=2*np.pi*C0/778e-9 - 2*np.pi*C0/806e-9, dl=7):
    """
    Calculates the theoretical LOGV (Local Optical Gauge Velocity).

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
    """
    return radius*(dOmega/dl)

def makeVolumeArray(data) -> np.ndarray:
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
        d = []
        times = []
        files = sorted([f for f in os.listdir(data) if f[-4:] == ".npy"], key = lambda x: getMotorPos(x), reverse=True)
        for file in files:
            d.append(np.load(data/file))
            times.append(calculateTiming(getMotorPos(file)))
        return fuseArrays(d), np.asarray(times)
    
def removeDoubleLitted(data, significance=1.5, plot=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes exposures with double illumination based on summed intensity.

    Parameters
    ----------
    data : np.ndarray
        3D data array of shape (z, x, y).
    significance : float, optional (default=1.5)
        Significance threshold for outlier removal.
    plot : bool, optional (default=False)
        If True, plot histogram of summed intensities.

    Returns
    -------
    filtered_data : np.ndarray
        Data with double-illuminated exposures removed.
    mask : np.ndarray
        Boolean mask indicating which exposures were kept.
    """
    summed = np.sum(np.sum(data,1),1)
    if plot:
        plt.figure()
        plt.hist(summed, bins = 20)
        plt.show()
    mask = summed > significance*np.median(summed)
    return data[~mask], ~mask

def cropByTimestamp(data:np.ndarray, times:np.ndarray, tMin, tMax):
    """
    Crops data and timestamps to the specified time window.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (t, ...).
    times : np.ndarray
        Array of timestamps, shape (t,).
    tMin : float
        Minimum timestamp to keep.
    tMax : float
        Maximum timestamp to keep.

    Returns
    -------
    cropped_data : np.ndarray
        Cropped data array.
    cropped_times : np.ndarray
        Cropped timestamps.
    """
    index = np.where((times<=tMax)&(times>=tMin))
    return data[index], times[index]       

def makeIsosurfaceMask(data:np.ndarray, level:float, epsilon: float = 1):
    """
    Creates a boolean mask to extract an isosurface at a specified level from a volume array.

    Parameters
    ----------
    data : np.ndarray
        Volume array of shape (n, x, y).
    level : float
        Relative isosurface level [0, 1].
    epsilon : float, optional (default=1)
        Tolerance range around the isosurface level.

    Returns
    -------
    mask : np.ndarray
        Boolean mask where the isosurface is located.
    """

    if not (0 <= level <= 1):
        raise ValueError("Parameter `level` must be in the range [0, 1]")

    threshold = data.max() * level
    mask = np.abs(data - threshold) < epsilon
    return mask

def fitAngleData(data: np.ndarray, times: np.ndarray):
    """
    Fits the angle data to a robust linear model (y = m*x + b) using statsmodels and returns the slope and intercept.

    Parameters
    ----------
    data : np.ndarray
        1D array of angle data, shape (t,).
    times : np.ndarray
        1D array of timestamps, shape (t,).

    Returns
    -------
    slope : float
        Slope of the robust linear fit.
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

def getRadialIntensity(data: np.ndarray, center: tuple[int, int], cropRadius: int, showPlot: bool = False) -> np.ndarray:
    """
    Calculates the radial intensity profile of a 3D array (e.g., time or wavelength series of 2D images).

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n, x, y), where n is the number of slices.
    center : tuple[int, int]
        Center coordinates (x, y).
    cropRadius : int
        Mask radius in pixels.
    showPlot : bool, optional (default=False)
        If True, plot the radial intensity profile for each slice.

    Returns
    -------
    intensityMax : np.ndarray
        1D array of shape (n,), where each element is the maximal intensity of one slice.
    """
    n = data.shape[0]
    intensityMax = np.zeros(n)
    for i in range(n):
        x, y = np.meshgrid(
            np.linspace(0, data.shape[1] - 1, data.shape[1]),
            np.linspace(0, data.shape[2] - 1, data.shape[2])
        )
        mask = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= cropRadius
        radial_profile = data[i].copy()
        radial_profile[~mask] = 0
        maxI = np.max(radial_profile)
        intensityMax[i] = maxI
    return intensityMax

def fitAnglesAndIntensity(
    data: np.ndarray,
    times: np.ndarray,
    center: tuple[int, int],
    cropRadius: int,
    threshhold: float = 0.8,
    crop: bool = True
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Calculates the angle and intensity for each time slice in a 3D data array, and fits the angle data to a robust linear model.

    For each slice in the input data:
      - Computes the angle using getCurrentAngleLS (using squared weighting).
      - Calculates the maximum radial intensity using getRadialIntensity.
      - Optionally filters out slices with intensity below half the maximum.
      - Fits the resulting angle vs. time data using a robust linear model (statsmodels RLM).

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n, x, y), where n is the number of slices.
    times : np.ndarray
        1D array of timestamps, shape (n,).
    center : tuple[int, int]
        Center coordinates (x, y) for angle and intensity calculation.
    cropRadius : int
        Mask radius in pixels for angle and intensity calculation.
    threshhold : float, optional
        Threshold for angle calculation (passed to getCurrentAngleLS).
    crop : bool, optional
        If True, only slices with intensity above half the maximum are used for fitting.

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
    n = data.shape[0]
    angles = np.nan*np.zeros(n)

    for i in range(n):
        angles[i] = getCurrentAngleLS(data[i], threshhold, center, cropRadius)[1]
    # Get intensity profile for each slice
    intensities = getRadialIntensity(data, center, cropRadius, showPlot=False)
    angles = np.array(angles)
    if crop:
        angles, times, intensities = cropByIntensity(angles, times, intensities) 
    # Fit angles vs times
    angleFit = fitAngleData(angles, times)[2]
    if crop:
        return angleFit, intensities, angles, times
    return angleFit, intensities, angles

def cropByIntensity(data: np.ndarray, times: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selects and returns only those slices of the data, timestamps, and intensity values where the intensity exceeds half of the maximum intensity.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n, x, y), where n is the number of slices.
    times : np.ndarray
        1D array of timestamps, shape (n,).
    intensity : np.ndarray
        1D array of intensity values, shape (n,).

    Returns
    -------
    cropped_data : np.ndarray
        Data slices where intensity > 0.5 * max(intensity).
    cropped_times : np.ndarray
        Corresponding timestamps for the selected data.
    cropped_intensity : np.ndarray
        Corresponding intensity values for the selected data.
    """
    maxI = np.max(intensity)
    index = np.where(intensity > maxI*0.5)
    return data[index], times[index], intensity[index]