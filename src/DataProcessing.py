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

MOTOR_STEP_SIZE = 50e-9 #m
C0 = 299792458  #m/s
def _checkDataType(arg):
    if isinstance(arg, np.ndarray):
        return arg 
    elif (isinstance(arg, Path) or isinstance(arg, str)):
            return np.load(arg)
    else:
        raise TypeError(f"Wrong type of input argument. Must be type either {type(str)}, {type(Path)} or {type(np.ndarray)}!")
    
def reconstruct_data(file, 
                     cx:int, cy:int, A:int, B:int, thickness:int, angle_deg:float = 0,
                     crop_pos = ((500,500),1000), 
                     plot_all:bool=False, 
                     plot_reconstruction:bool=False, 
                     fft_crop_radius:float = -1, 
                     crop_shift:tuple[int,int] = (0,0),
                     subtract_ref = None) -> np.ndarray:
    """
    Reconstructs the full electric field(phase and amplitude) out of the off-axis-holography interference data.
    First the fft of the data is taken. An ellipse is defined by `cx, cy, A, B` to crop out the part of the fft reffering to the interference term MR* or M*R.
    This crop gets centered in frequency space and gets inverse transformed to get the phase and amplitude of the field.
    The data array is cropped to be a square. This is defined by `crop_pos`.

    Parameters
        :param file: Path to the file to reconstruct or the raw data as np.ndarray
        :type file: str | Path | numpy.ndarray
        :param cx: Center x-position of the cropping ellipse
        :type cx: int
        :param cy: Center y-position of the cropping ellipse
        :type cy: int
        :param A: Ellipse total width
        :type A: int
        :param B: Ellipse total height
        :type B: int
        :param thickness: Ellipse thickness
        :type thickness: int
        :param angle_deg: default: 0
        Rotation angle of the ellipse
        :type angle_deg: float
        :param crop_pos: default: ((500,500),1000)
        Crop position (x,y) and width: ((posX,posY),width) of the data crop.
        :type crop_pos: tuple(tuple(int,int), int)
        :param plot_all: default: False
        Plot the reconstruction steps
        :type plot_all: bool
        :param plot_reconstruction: default: False
        Plot the reconstructed phase and amplitude
        :type plot_reconstruction: bool
        :param fft_crop_radius: default: -1
        Defines the crop width of the masked fft used for the reconstruction step. Width is in units of the maximal extend of the ellipse.
        If value is -1 the radius is automatically chosen to be the maximal possible width keeping it squared. 
        :type fft_crop_radius: float
        :param crop_shift: default: (0,0)
        Adds shift coordinates to the center of the ellipse to compensate a shift of the interference ellipse in the data. Used for multiple reconstructions.
        :type crop_shift: tuple(int,int)
        :param subtract_ref: default: None
        Subtracting a data array from the input.
        :type subtract_ref: str | Path | numpy.ndarray of size (file.size)
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

def reconstruct_data_gaussfilt( file, cx, cy, rin, rout, plot_all=False, plot_reconstruction=False ):
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
    Shifts the phase of every image relating on one reference pixel (pixel with max intensity in the first image), 
    which is set to the same phasevalue in every imagecontained in field (images stored along 3rd axis).
    
    :param field: Array of fourier transformed measurement data 
    :type field: 3dim complex numpy.ndarray
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
    extract the motorposition from the filename
    
    Parameters
        :param name: filename
        :type name: str
    """
    match = re.search(r"motorPos\s*(-?\d+)", name)
    motor_pos = None
    if match:
        motor_pos = int(match.group(1))
        motor_pos = motor_pos * MOTOR_STEP_SIZE
    return motor_pos

def calculateTiming(motorPos):
    """
    Parameters
    :param motorPos: Motorposition in m
    :type motorPos: float | numpy.ndarray
    """
    return motorPos/C0*2

def normalizeTimestamps(times:np.ndarray):
    return times - (times.min())


def reconstructFromFolder(folderPath:Path, cx, cy, A, B, thickness, angle,
                    crop_pos, removeWrongExposure = True, significance = 1.5,
                    shift = None, subtract_ref:Path = None, phaselock = 0, plot_rec = False, plot_all = False, plot_exp_hist = False, fft_crop_radius=6):
    """
    reconstructs the field from the data in the given folder 
    and safes it as two files for real and imaginary part to import it to matlab.
    Other parameters see `reconstruct_data`!

    :param shift: if not None: must contain value for each file for x and y: ((x_i),(y_i))
    :type shift: None | (float[n_files],float[n_files])

    :param phaselock: phaselocks all reconstructed E_i at one pixel
    :type phaselock: bool

    :param tilt_correction: enables tilt correction to compensate the angle of the OAH
    :type phaselock: bool default:False
    
    :param subtract_ref: defines subtraction data
    :type subtract_ref: Path|numpy.ndarray|None default:None
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
        rec = reconstruct_data(rawData[i,:,:],
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

def fitCircle(points:np.ndarray, cArray:np.ndarray, sigmaFit = 1,show_statistics=False)->tuple[tuple[int,int],float]:
    """
    Fits a circle to circular distributed data points. Returning center(x,y) and radius. For correct usage see `findRadius`!

    Parameters
        :param points: Data
        :type points: np.ndarray of shape (2,n)
        :param cArray: Filtered data array, where only points considered for the fit are have values > 0.
        :type cArray: np.ndarray of shape (x,y)
        :param sigmaFit: default:1 
        Defines the range of datapoints considered for the fit in sigma of the distribution of the distance to the estimated center.
        :type sigmaFit: float 
        :param show_statistics: default:False
        Plotting the circle fit and histograms
        :type show_statistics: bool
        
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

def findRadius(img:np.ndarray, centerXY:tuple[int,int], filterR_px = 30, threshhold = 0.8, sigma_Fit = 1, showStatistics = False):
    """
    Finding the central radius of the lightspring from the ccd image. The function takes an pre estimated center and a filter radius to filter the 0th order data.
    Also using the 0th order maximum to find the estimated center of the circle for `fitCircle`. All data below `threshold` is set to 0 before passing to `fitCircle`.

    Parameters
        :param img: ccd image data
        :type img: np.ndarray of shape(x,y)

        :param centerXY: estimated center of the circle
        :type centerXY: tuple[int,int]

        :param filterR_px: default: 30
        radius of the inner circle filter given in pixel
        :type filterR_px: int

        :param threshhold: default: 0.8 Threshhold for the data to be consired for the fit.
        :type threshhold: float

        :param sigmaFit: default:1 
        Defines the range of datapoints considered for the fit in sigma of the distribution of the distance to the estimated center.
        :type sigmaFit: float 

        :param showStatistics: default: False Plot fitting statistics
        :type showStatistics: bool
    <br>
    :return: radius in pixels, center in pixels
    :rtype: tuple[float, float]
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

def calculateRadius(radius_px, pixelSize_mm = 3.45e-3):
    """
    calculates the radius (in m) from pixels
    """
    return radius_px*pixelSize_mm*1e-3

def getCurrentAngleLS(lsSLice:np.ndarray, threshhold, center:tuple[int,int], maskRadius:int, showPlot = False, showFit = False)->tuple[float,float]:
    """
    Gets the angle to the x-axis (counterclockwhise) of the lightspring slice. The coordinatesystem is centered at `center`.

    Parameters
        :param lsSLice: single temporal slice of reconstruction data of lightspring
        :type lsSLice: numpy.ndarray of shape(n,n)
        :param threshhold: threshhold of radius intensitys that will be considered for the fit
        :type threshhold: float
        :param center: center of the lightspring (use fitCircle is recommended)
        :type center: tuple[int,int]
        :param maskRadius: radius of the center mask. used to neglect intensitys inside the spring (i.e. 0th order)
        :type maskRadius: int
        :param showPlot: False
        show fitting statistic
        :type showPlot: bool
        :param showFit: False
        show fitting plot
        :type showFit: bool
    _________________________
        :return: tuple of angles based on diffrent calculation algorythms:
        (maximumValue, squared weighted average)
        :rtype: tuple[float]

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
            threshhold = 0.8, sigma_Fit = 1,
            fromIsosurface = False, isoLevel = .5, isoTolerance = 10,
            showStatistics = False, showFit = False, angleFittingAlgo = 1):
    """
    Calculates the OGV (Optical Gauge Velocity) and radius for each slice of the `lightspringData`, as well as the
    average OGV and average radius. These values can be used to compute the LOGV (Local Optical Gauge Velocity), among other metrics.

    The OGV is computed by differentiating the angle obtained from the `getCurrentAngleLS` function using the
    corresponding time values provided in `times`. Radii are computed using the `findRadius` function.

    Parameters
    ----------
    lightspringData : np.ndarray
        3D array representing the light spring data with shape (t, x, x), where `t` is the number of time slices.
    
    times : np.ndarray
        1D array of shape (t,) containing the timestamp for each time slice of `lightspringData`.
    
    estimatedCenter : tuple[int, int]
        Coordinates (x, y) of the estimated center of the light spring in each slice.

    maskRadius : int
        Radius of the circular mask applied around the estimated center for region-of-interest selection.
    
    threshhold : float, optional (default=0.8)
        Threshold used to filter the data before fitting. Only values above `threshhold * max(data)` are considered.

    sigma_Fit : float, optional (default=1.0)
        Standard deviation range (in sigma units) of distances from the estimated center used for fitting the radius.

    fromIsosurface : bool, optional (default=True)
        If True, the angle is calculated based on an isosurface mask with level defined by `trheshold`. Otherwise, raw data is used.

    isoLevel : float [0;1]
        relative isosurface level

    isoTolerance : float
        Tolerance range around the isosurface level

    showStatistics : bool, optional (default=False)
        If True, plots statistical information about the fitting process.

    showFit : bool, optional (default=False)
        If True, plots the fit for each slice.

    angleFittingAlgo : int, optional (default=1)
        Only considered if `fromIsosurface` is False!
        Algorithm used to determine the current angle from the light spring:
            0 = Maximum intensity location
            1 = Squared weighted average (recommended)
            2 = Center of mass (linear weighted average)

    Returns
    -------
    averageOGV : float
        The average optical gauge velocity over all time slices [rad/m].

    averageRadius : float
        The average radius computed over all time slices [m].

    OGV_array : np.ndarray
        Array of shape (t,) containing the OGV for each slice [rad/m].

    radius_array : np.ndarray
        Array of shape (t,) containing the radius for each slice [m].

    angle_array : np.ndarray
        Array of shape (t,) containing the angular position for each slice [rad].
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

def calculateLOGV_theory(radius, dOmega = 2*np.pi*C0/778e-9 - 2*np.pi*C0/806e-9 , dl = 7):
    return radius*(dOmega/dl)

def makeVolumeArray(data)->np.ndarray:
    """
    Combines a list of n equal sized ndarrays of size (x,y) to one array of size (n,x,y).

    Parameters
        :param data: Path or data list to combine to 3D array.
        :type data: Path | list[np.ndarray]
        <br>
        :return: 3 dim data array. If data is type(Path) also return the timings.
        :rtype: numpy.ndarray(type(any), size(n,x,y)) | tuple(ndarray(type(any), size(n,x,y)), ndarray(type(float), size(n)))
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
    
def removeDoubleLitted(data, significance = 1.5, plot = False)->tuple[np.ndarray, np.ndarray]:
    """
    
    Parameters
        :param data: 3D data array
        :type data: np.ndarray of size(z,x,y)
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
    Crops given data (shape(t)) with timestamps `times` (shape(t)) by the given timestamp boundarys corresponing to the data.

    :return: cropped data, cropped times
    :rtype: tuple[np.ndarray(shape(tnew,), np.ndarray(shape(tnew)))]
    """

    index = np.where((times<=tMax)&(times>=tMin))
    return data[index], times[index]       

def makeIsosurfaceMask(data:np.ndarray, level:float, epsilon: float = 1):
    """
    Makes a boolean mask to extract an isosurface specified by the given level [0;1] from a volume array given in `data` in the range of `level`+-`epsilon`

    Parameters
    ----------
    data : np.ndarray
        Volume array of shape (n, x, y)

    level : float [0,1] 
        relative isosurface level

    epsilon : float
        Tolerance range around the isosurface level
    Returns
    -------
    np.ndarray
        A boolean mask where the isosurface is approximately located
    """

    if not (0 <= level <= 1):
        raise ValueError("Parameter `level` must be in the range [0, 1]")

    threshold = data.max() * level
    mask = np.abs(data - threshold) < epsilon
    return mask
    