"""
Package for processing spacetime-beams related data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import re

def reconstruct_data( file, cx, cy, rin, rout, crop_pos, plot_all=False, 
                     plot_reconstruction=False, fft_crop_radius:float = -1, circle_shift:tuple[int,int] = (0,0),
                     phaseshift_pos:int = None, subtract_ref = None) -> np.ndarray:
    """
    Reconstructs the full electric field(phase and amplitude) out of the off-axis-holography interference data.
    First the fft of the data is taken. An ellipse is defined by `cx, cy, rin, rout` to crop out the part of the fft reffering to the interference term MR* or M*R.
    This crop gets centered in frequency space and gets inverse transformed to get the phase and amplitude of the field.

    Parameters
        :param file: Path to the file to reconstruct or the raw data as np.ndarray
        :type file: str|Path|numpy.ndarray
    
        :param cx: 
        :type cx: _type_
    
        :param cy: 
        :type cy: _type_
    
        :param rin: 
        :type rin: _type_
    
        :param rout: 
        :type rout: _type_
    
        :param crop_pos: 
        :type crop_pos: _type_
    
        :param plot_all: False
        :type plot_all: 
        
        :param plot_reconstruction: False
        :type plot_reconstruction: 
        
        :param fft_crop_radius: -1
        :type fft_crop_radius: 
        
        :param circle_shift: (0,0)
        :type circle_shift: 
        
        :param phaseshift_pos: None
        :type phaseshift_pos: 
        
        :param subtract_ref: None
        :type subtract_ref: 
        
    """
    # parameters
        # file: path to the data file
        # cx, cy: center of the signal in fourier space
        # rin, rout: inner and outer radius of the mask in fourier space
        # crop_pos: tuple[tuple[int,int], int] Position to crop the raw input data
        # plot_all: plot all the steps of the reconstruction
        # plot_reconstruction: plot the reconstructed data
        # fft_crop_radius: positive float:  defines the radius with which the fft should be cropped
        # circle_shift: shifts the circle center by (x,y)
        # phaseshift_pos: None|int 


    # read data
    plt.close("all")
    if (isinstance(file, Path) or isinstance(file, str)):
        data = np.load(file).astype(np.int16)
        
    elif isinstance(file, np.ndarray):
        data = file
    else:
        raise TypeError("Wrong type for argument file")
    # plt.figure()
    # plt.imshow(data)
    # plt.title("loaded data")
    # plt.show()
    if subtract_ref is not None:
        if isinstance(subtract_ref, np.ndarray):
            data = data - subtract_ref 
        elif (isinstance(file, Path) or isinstance(file, str)):
            ref = np.load(subtract_ref)
            sub = data - ref
            sub[sub < 0] = 0
            print(sub.dtype, ref.dtype, data.dtype)
            fig, axs = plt.subplots(1,3)
            a = axs[0].imshow(data)
            b = axs[1].imshow(ref)
            c = axs[2].imshow(sub)
            plt.colorbar(a, ax = axs[0])
            plt.colorbar(b, ax = axs[1])
            plt.colorbar(c, ax = axs[2])
            plt.show()
            data = sub

    center = crop_pos[0]
    width = crop_pos[1]
    data = data[int(center[0]-width/2):int(center[0]+width/2),
                int(center[1]-width/2):int(center[1]+width/2)]
    nx, ny = data.shape
    #print(f"{nx}, {ny}")
    X, Y = np.meshgrid( np.arange(nx), np.arange(ny), indexing="ij" )
    
    r_crop_fft = int(rout*fft_crop_radius)
    if fft_crop_radius == -1:
        r_crop_fft = np.min( [ np.abs( nx-cx ), np.abs( cx ), np.abs( ny-cy ), np.abs( cy ) ] )

    # fft
    PIXEL_SIZE = 3.3
    fft = np.fft.fftshift( np.fft.fft2(data) )
    kx = np.fft.fftshift( np.fft.fftfreq( nx, PIXEL_SIZE ) ) * 2*np.pi
    ky = np.fft.fftshift( np.fft.fftfreq( ny, PIXEL_SIZE ) ) * 2*np.pi
    # process fft
    fft_proc = fft.copy()
    # phaseshift
    if not phaseshift_pos is None:
        phase = np.zeros(fft_proc.shape, dtype=complex)
        a = 100
        for i in range(phase.shape[0]):
            for j in range(phase.shape[1]):
                r = np.sqrt((i-(phase.shape[0]/2))**2+(j-(phase.shape[1]/2))**2)
                phase[i][j] = ((a*r**2*phaseshift_pos/(2*299792458)) % (2*np.pi)) *1j
        plt.figure()
        plt.imshow(np.imag(phase))
        plt.colorbar()
        plt.title("applied phaseshift")
        plt.show()
        fft_proc = fft_proc * phase
    
    r = np.sqrt( (X-(cx+circle_shift[0]))**2 + (Y-(cy+circle_shift[1]))**2 )
    mask = (r > rin) & (r < rout)
    fft_proc[ np.logical_not( mask ) ] = 0
    fft_proc = fft_proc[cx-r_crop_fft:cx+r_crop_fft, cy-r_crop_fft:cy+r_crop_fft]#fft_proc[cx+circle_shift[0]-r_crop_fft:cx+circle_shift[0]+r_crop_fft, cy+circle_shift[1]-r_crop_fft:cy+circle_shift[1]+r_crop_fft] #fft_proc[cx-L:cx+L,  cy-L:cy+L ]
    if not fft_proc.shape[0] == fft_proc.shape[1]:
        raise IndexError(f"shape of the fft crop array is wrong: {fft_proc.shape}")
    # plt.figure()
    # plt.imshow(abs(fft_proc),cmap="plasma")
    # plt.show()
    # reconstruct data
    data_proc = np.fft.ifft2( np.fft.fftshift( fft_proc ) )
    #print(data_proc)
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
        plt.xlim( [cx+circle_shift[0]-1.1*rout, cx+circle_shift[0]+1.1*rout] )
        plt.ylim( [cy+circle_shift[1]-1.1*rout, cy+circle_shift[1]+1.1*rout] )
 
        for ax in axs[1:3]:
            ax.scatter( cx+circle_shift[0], cy+circle_shift[1], c='k', s=10 )
            circle1 = plt.Circle( [cx+circle_shift[0], cy+circle_shift[1]], rin, fill=False, color='k', ls='--', lw=0.5 )
            circle2 = plt.Circle( [cx+circle_shift[0], cy+circle_shift[1]], rout, fill=False, color='k', ls='--', lw=0.5 )
            ax.add_artist(circle1)
            ax.add_artist(circle2)
 
 
        plt.scatter( cx+circle_shift[0], cy+circle_shift[1], c='k', s=10 )
        circle1 = plt.Circle( [cx+circle_shift[0], cy+circle_shift[1]], rin, fill=False, color='k', ls='--', lw=0.5 )
        circle2 = plt.Circle( [cx+circle_shift[0], cy+circle_shift[1]], rout, fill=False, color='k', ls='--', lw=0.5 )
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

def reconstructFromFolder(folderPath:Path, cx, cy, rin, rout, crop_pos,
                    shift = None, subtract_ref:Path = None, phaselock = 0, tilt_correction = False, plot_rec = False, plot_all = False, fft_crop_radius=6):
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
    def get_motorPos(name):
        match = re.search(r"motorPos\s*(-?\d+)", name)
        motor_pos = None
        if match:
            motor_pos = int(match.group(1))
            motor_pos = motor_pos * 58e-9 #stepsize
        return motor_pos
    
    files = os.listdir(folderPath)
    files = [f for f in files if f[-4:] == ".npy"]
    minPos = max(get_motorPos(x) for x in files) #max() because its going further in -x direcion
    i = 0
    data = None
    for file in files:
        fileName = file[0:-4]
        print(fileName)
        motorPos = None
        if tilt_correction:
            motorPos = abs(get_motorPos(fileName)-minPos)
            print(motorPos)
        if shift is None:
            shift_i = (0,0)
        else:
            shift_i = (shift[0][i],shift[1][i])
        rec = reconstruct_data(folderPath/Path(file), cx, cy, rin, rout, crop_pos, plot_all=plot_all,
                    plot_reconstruction = plot_rec,fft_crop_radius=fft_crop_radius,circle_shift=shift_i,
                    phaseshift_pos=motorPos,subtract_ref = subtract_ref)
        savePath = folderPath/Path("reconstructionData")
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        if not phaselock:
            np.save(savePath/Path(fileName+"_reconstruction_complex"),rec)
            np.savetxt(savePath/Path(f"{fileName}_reconstruction_abs.csv"),np.abs(rec), delimiter=";")
            np.savetxt(savePath/Path(f"{fileName}_reconstruction_angle.csv"),np.angle(rec), delimiter=";")
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
    for j, file in enumerate(files):
        fileName = file[0:-4]
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
    mean, std =np.mean(distances), np.std(distances)
    distances_noOL = distances[((distances>mean-sigmaFit*std) & (distances < mean + sigmaFit*std))]
    # print(distances_noOL)
    if show_statistics:
        plt.figure()
        bins= 30
        plt.hist([distances,distances_noOL],bins)
        plt.title("histogram distances")
        plt.show()
    estimated_radius_i = np.mean(distances_noOL)

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
        
    """
    #filter the 0th order
    crop = img.copy()
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
    max_mask = crop >= threshhold*np.max(crop)
    max_points = np.indices(crop.shape)
    max_points = np.asarray(((max_points[1][max_mask]),(max_points[0][max_mask])))
    # print(max_points)
    maxima = crop[max_mask]
    crop[~max_mask] = 0
    center, radius = fitCircle(max_points.reshape(2,(max_points.shape[1])),c_array,sigma_Fit, showStatistics)

    plt.figure()
    plt.imshow(crop)
    plt.scatter(*max_points, marker="x", color = "red")
    plt.scatter(*center)
    c = plt.Circle(center,radius, color = "green", ls = ":",fill=False)
    ax = plt.gca()
    ax.add_patch(c)
    ax.add_patch(plt.Circle(center, filterR_px, color = "yellow", ls = ":",fill=False))
    #rad = plt.Circle(estimated_center)
    plt.show()
    return radius

def calculateRadius(radius_px, pixelSize_mm = 3.45e-3):
    return radius_px*pixelSize_mm

def calculateOGV(radius, dOmega = 2*np.pi*300000/28e-9, dl = 7):
    return radius*(dOmega/dl)
