import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from imageio import imread
import re

SLM_WIDTH_PX = 1280
SLM_HEIGHT_PX = 1024
SLM_PIXEL_LENGTH = 12.5e-6 #m

DEFAULT_VOLTAGE_CALIBRATION = np.asarray([255/(2*np.pi),0])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def create_checkerboard(height = SLM_HEIGHT_PX, width = SLM_WIDTH_PX, square_size = 4, value_black=0, value_white=2*np.pi-1e-10):
    """
    Creates a checkerboard pattern with customizable square size.
    
    Parameters
        :param height: Height of the output pattern in pixels (default: SLM_HEIGHT_PX)
        :type height: int
        :param width: Width of the output pattern in pixels (default: SLM_WIDTH_PX)
        :type width: int
        :param square_size: Size of each square in pixels (default: 4)
        :type square_size: int
        :param value_black: Value for black squares (default: 0)
        :type value_black: float
        :param value_white: Value for white squares (default: 2π)
        :type value_white: float
    
    Returns
        :return: 2D checkerboard pattern array
        :rtype: np.ndarray
    
    Example
        >>> cb = create_checkerboard(1024, 1280, 32)  # 32x32 pixel squares
    """
    checkerboard = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                checkerboard[i, j] = value_black
            else:
                checkerboard[i, j] = value_white
    return checkerboard

class Hologram:
    def __init__(self, widthPx = SLM_WIDTH_PX , heightPx = SLM_HEIGHT_PX, 
                 pixelLength = SLM_PIXEL_LENGTH):
        self.width = widthPx
        self.height = heightPx
        self.pixelSize = pixelLength
        self.x, self.y = np.meshgrid(np.arange(-np.floor(self.width/2),np.floor(self.width/2)),
            np.arange(-np.floor(self.height/2),np.floor(self.height/2)))
        self.rho, self.phi = cart2pol(self.x, self.y)
        self.rho = self.rho*pixelLength
        self.holoBase = np.zeros((self.height,self.width))
        self.holo = self.holoBase
        self.phaseCalibration = np.zeros((self.height,self.width))
        self.voltageCalibration = DEFAULT_VOLTAGE_CALIBRATION
        self.zernikeHolo = np.zeros((self.height,self.width))
        self.boundaries:np.ndarray = None
        self.OAMs:np.ndarray = None
        self.frequencyRadiusMap = None
        self.omega_max = 840e-9
        self.omega_min = 774.405e-9
        self.rMin = 5.63 
        self.rMax = 6.05
        return
    
    def _updateModes(self, boundaries, OAMs):
        self.boundaries = boundaries
        self.OAMs = OAMs

    def _applyPhaseCalibration(self):
        self.holo = (self.holo+self.phaseCalibration)%(2*np.pi)
        #self.plotHolo()
    
    def _applyVoltageCalibration(self):
        """
        Apply the voltage calibration to the hologram. 
        Apply this as the last step before finalizing the hologram!
        The fit is intended to return a value in the interval [0;255] for given values from [0;2pi].
        """
        self.holo = np.polyval(self.voltageCalibration,self.holo%(2*np.pi))/(255/(2*np.pi))

    def _applyZernikeCorrection(self):
        self.holo = (self.holo+self.zernikeHolo)%(2*np.pi)

    def _finalizeHologram(self):
        self._applyPhaseCalibration()
        self._applyZernikeCorrection()
        self._applyVoltageCalibration()
    
    def _linearInterpolRadius(self, n:int, OAM_start:int, deltaL = 1):
        omegas = np.linspace(self.omega_min,self.omega_max,n+1)
        radia = np.interp(omegas,
                          [self.omega_min, self.omega_max],
                          [self.rMin, self.rMax])
        OAM = np.flip(np.linspace(OAM_start,OAM_start+(n-1)*deltaL,n))
        return radia, OAM

    def flipHolo(self, axis:int = 1):
        self.holo = np.flip(self.holo, axis=axis)
        self._updateModes(self.boundaries, -self.OAMs)
        self._finalizeHologram()
        return self.holo

    ## Hologram functions
    def vortex(self,OAM:int, rMin = None, rMax = None):
        if rMin is None: rMin = 0
        if rMax is None: rMax = np.max(self.rho)
        holo = self.holoBase
        holo[(self.rho >= rMin) & (self.rho<=rMax)] = self.phi[(self.rho >= rMin) & (self.rho<=rMax)]*OAM % (2*np.pi)
        self.holo = holo
        self._updateModes(np.asarray((rMin,rMax)), np.asarray(([OAM])))
        self._finalizeHologram()
        return self.holo

    def lightspring(self, OAM:int, deltaL:int, nSample:int, rMin = None, rMax = None):
        """
        Creates a hologram for a lightspring specified by the input parameters. 
        A number of `nSamples` different OAMs is generated with a difference `deltaL` between each OAM.
        The OAMs are applied from the smallest to the largest radius.
        
        Parameters
            :param OAM: start value for the OAM
            :type OAM: uint
            :param deltaL: difference between each OAM modulation of the spring
            :type deltaL: int         
            :param nSample: number of OAM modes
            :type nSample: uint        
            :param rMin: minimum radius [m] of the OAM zones
            :type rMin: float            
            :param rMax: maximum radius [m] of the OAM zones
            :type rMax: float
            
        """
        if rMin is None: rMin = 0
        if rMax is None: rMax = np.max(self.rho)
        holo = self.holoBase
        boundaries = np.linspace(rMin,rMax,nSample+1)
        OAMs = np.linspace(OAM,OAM+(nSample-1)*deltaL,nSample)
        for i, oam in enumerate(OAMs):
            holo[(self.rho >= boundaries[i]) & (self.rho<boundaries[i+1])] = self.phi[(self.rho >= boundaries[i]) & (self.rho<boundaries[i+1])]*oam
        self.holo = holo
        self._updateModes(boundaries, OAMs)
        self._finalizeHologram()
        return self.holo
    
    def customOAMFunction(self, boundaries, OAMs)->np.ndarray:
        """
        Creates a hologram with custom shaped OAM distribution. Maps the OAM(i) to the space between boundaries(i:i+1).

        Parameters
            :param boundaries: boundaries in units of rho for the given OAM_i
            :type boundaries: np.ndarray of type float and shape(n,)        
            :param OAMs: OAMs for each space between given boundaris
            :type OAMs: np.ndarray of type int, shape (n-1,)
        
        """
        if not (np.shape(boundaries)[0] == np.shape(OAMs)[0]+1):
            raise IndexError(msg="len of 'boundaries' must be len('OAMs')+1 !" )
        holo = self.holoBase
        for i, oam in enumerate(OAMs):
            holo[(self.rho >= boundaries[i]) & (self.rho<boundaries[i+1])] = self.phi[(self.rho >= boundaries[i]) & (self.rho<boundaries[i+1])]*oam
        self.holo = holo
        self._updateModes(boundaries,OAMs)
        self._finalizeHologram()
        return self.holo
    
    def parabolaOAM(self, n:int, OAM_start:int, OAM_max:int):
        """
        Makes parabolic profile of the OAM with the vertex at middle radius.

        Parameters
            :param n: Number of modes. Must be odd integer!
            :type n: int       
            :param OAM_start: Minimum OAM of the parabola.
            :type OAM_start: int       
            :param OAM_max: Maximum OAM of the parabola.
            :type OAM_max: int
        
        """

        if n%2 == 0:
            raise ValueError("Parameter 'n' must be an odd integer number!")

        def parabola(x):
            h = np.ceil(n/2)
            return ((OAM_max-OAM_start)/(n-h)**2)*(x-h)**2+OAM_start
        
        omegas = np.linspace(self.omega_min,self.omega_max,n+1)
        boundaries = np.interp(omegas,[self.omega_min,self.omega_max],[self.rMin, self.rMax])
        OAMs = np.ceil(parabola(np.linspace(1,n,n)))
        return self.customOAMFunction(boundaries, OAMs)

    def stepFunctionOAM(self, n:int, step_pos:int,
                        OAM_start:int, OAM_gap:int,
                        delta_l1:int = -1, delta_l2:int = 1):
        """
        Makes a stepfunction profile for the OAM.
        
        Parameters
            :param n: Number of OAM modes.
            :type n: uint
            :param step_pos: Position of the mode containing the step.
            :type step_pos: uint  
            :param OAM_start: OAM the first part of step function begins with.
            :type OAM_start: int    
            :param OAM_gap: Size of the OAM gap.
            :type OAM_gap: int
            :param delta_l1: Difference between the OAM modes of the first part of the step function.
            :type delta_l1: int default = -1            
            :param delta_l2: Difference between the OAM modes of the second part of the step function.
            :type delta_l2: int default = 1
        """

        boundaries, OAM1 = self._linearInterpolRadius(n, OAM_start+OAM_gap, delta_l1)
        OAM2 = self._linearInterpolRadius(step_pos, OAM_start, delta_l2)[1]
        OAM1[step_pos:] = OAM2
        OAM = OAM1
        return self.customOAMFunction(boundaries, OAM)

    def applyGratingToSections(self, boundaries:np.ndarray = None, holoIn:np.ndarray = None, g:float = 1, dphi:np.ndarray=None, plot=False):
        if boundaries is None:
            boundaries = self.boundaries
        if holoIn is None:
            holoIn = self.holo
        holoOut = np.copy(holoIn)
        n = boundaries.shape[0]-1
        if dphi is None:
            dphi = 2*np.pi/n
            dphi = dphi*np.linspace(0,n-1,n)
        elif not (dphi.shape[0] == n):
                raise IndexError(f"length of 'dphi' must equal 'n'! shape(lenght) == ({int(n)},) but is {dphi.shape}")
        for i in range(n):
            grating = (2*np.pi*(np.cos(dphi[i])*g*self.x/self.width+np.sin(dphi[i])*g*self.y/self.width))%(2*np.pi)
            if plot:
                plt.figure()
                plt.title(f"Grating {i+1}/{n}: phi = {dphi[i]*(180/np.pi)}°")
                a = plt.imshow(grating)
                plt.colorbar(a)
                plt.show()
            new = holoIn[(self.rho>boundaries[i])&(self.rho<boundaries[i+1])] + grating[(self.rho>boundaries[i])&(self.rho<boundaries[i+1])]
            holoOut[(self.rho>boundaries[i])&(self.rho<boundaries[i+1])] = new
        holoOut = holoOut%(2*np.pi)
        self.holo = holoOut
        return self.holo, dphi
    
    def calibrationHolo(self, rMin:float, rMax:float, thickness_mm:float = 0.4, nSample:int = 5, OAM:int = 60, deltaOAM:int = -1, g = 30):
        if deltaOAM == -1:
            deltaOAM = round(OAM/4,0)
        outerBoundaries = np.linspace(rMax,rMax+thickness_mm,nSample+1)
        innerBoundaries = np.linspace(rMin-thickness_mm,rMin,nSample+1)
        oMask = (self.rho>outerBoundaries[0])&(self.rho<outerBoundaries[-1])
        iMask = (self.rho>innerBoundaries[0])&(self.rho<innerBoundaries[-1])
        holo = self.holoBase
        innerHolo = ((OAM-deltaOAM)*self.phi)%(2*np.pi)
        outerHolo = (OAM*self.phi)%(2*np.pi)
        holo[oMask] = outerHolo[oMask]
        holo[iMask] = innerHolo[iMask]
        outHolo, dphi = self.applyGratingToSections(outerBoundaries,holo,g)
        outHolo = self.applyGratingToSections(innerBoundaries,outHolo,g, dphi = np.flip(dphi))[0]
        print("Calibration: each sample of the radius has thickness {:.2f} mm;\ninner radius maximum: {:.2f};\nouter radius minimum: {:.2f}".format(thickness_mm/nSample,rMin,rMax))



    def setHoloBase(self, mode = 0):
        def makeCheck(square_size):
            """
            Creates a checkerboard pattern with alternating squares.
            
            Parameters
                :param square_size: Size of each square in pixels
                :type square_size: int
            
            Returns
                :return: 2D array with checkerboard pattern (0 and 2π alternating)
                :rtype: np.ndarray
            """
            check = np.zeros((self.height, self.width))
            for i in range(self.height):
                for j in range(self.width):
                    # Determine if this pixel belongs to a black or white square
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        check[i, j] = 0
                    else:
                        check[i, j] = 2 * np.pi
            return check
        if mode == 0:
            self.holoBase = np.zeros((self.height,self.width))
        elif mode == 1:
            self.holoBase = makeCheck(1)
        else:
            return
    
    def resetHolo(self):
        self.holoBase = np.zeros((self.height,self.width))

    ## Visualization
    def plotModes(self, show:bool = True):
        fig = plt.figure()
        plt.title("OAM Modes")
        if self.frequencyRadiusMap is None:
            for i in range(self.OAMs.size):
                plt.plot((self.boundaries[i],self.boundaries[i+1]), [self.OAMs[i]]*2,
                        ls="-",color="b", marker ="")
            plt.xlabel("slm radius /mm")
        #else ergänzen
        plt.ylabel("topological charge")
        plt.grid(True,"major",color = "grey", lw = .5, ls = ":")
        if show: plt.show()
        return fig
    
    def plotHolo(self, show:bool=True):
        fig = plt.figure()
        plt.title("Hologram")
        a = plt.imshow(self.holo)
        plt.colorbar(a)
        plt.xlabel("screen width /px")
        plt.ylabel("screen height /px")
        a = plt.gca()
        a.set_aspect("equal")
        if show: plt.show()
        return fig

    ## Calibration
    def resetPhaseCalibration(self):
        self.phaseCalibration = np.zeros((self.height,self.width))

    def resetVoltageCalibration(self):
        self.voltageCalibration = DEFAULT_VOLTAGE_CALIBRATION
    
    def convertHolo(self):
        converted = self.holo*(255/(2*np.pi))
        return converted.astype(np.uint8)
    
    def loadPhaseCalibrationFile(self, path:Path, plot:bool = False):
        """
        Loads the phase calibration file for the slm.
        """
        #(r"C:\Users\Voxel\Desktop\Data_analysis_LS\Correction_patterns/CAL_LSH0803840_785nm.bmp")
        calibration = imread(path)
        resize = np.zeros((self.height,self.width))
        cy,cx = calibration.shape
        dx = self.width-cx
        dy = self.height-cy
        x1 = int(round(dx/2))
        y1 = int(round(dy/2))
        resize[y1:y1+cy,x1:x1+cx] = calibration/255*np.pi*2
        if plot:
            plt.figure()
            a = plt.imshow(resize)
            plt.colorbar(a)
            a = plt.gca()
            a.set_aspect("equal")
            plt.show()
        self.phaseCalibration = resize

    def loadVoltageCalibrationFile(self, path:Path):
        def extractParameters(text):
            bracket_match = re.search(r"\[([^\]]+)\]", text)
            if bracket_match:
                content = bracket_match.group(1)
                # Step 2: Find all floating-point numbers in that content
                numbers = re.findall(r"-?\d+\.\d+", content)
                numbers = np.asarray([float(n) for n in numbers])
                self.voltageCalibration = numbers
                return True
            return False
        
        with open(path) as file:
            lines = file.readlines()
        for line in lines:
            if extractParameters(line):
                print(f"Voltage calibration parameters loaded successfuly!\n{self.voltageCalibration}")
                break

    def setZernikeCorrection(self, pist:float = 0,
                          xTilt:float = 0, yTilt:float = 0,
                          defocus:float = 0, 
                          astic0:float = 0, astic45:float = 0,
                          verComa:float = 0, horComa:float = 0,
                          verTref:float = 0, horTref:float = 0,
                          spherical:float = 0):
        """
        from current setup (may 2025):
        pist = 0.; x_tilt = -0.; y_tilt = -0.; dfoc = 0; ver_ast=0.03; ob_ast = 0.00; h_coma = 0.00; v_coma = 0.0; v_tref = 0.; ob_tref = 0.; spher = 0.00;
        """
        holo = np.ones((self.height,self.width))*pist
        holo = holo + xTilt * (self.rho*np.cos(self.phi)) 
        holo = holo + yTilt * (self.rho*np.sin(self.phi))
        holo = holo + defocus * (2*np.power(self.rho,2)-1)
        holo = holo + astic0 * (np.power(self.rho,2)*np.cos(2*(self.phi)))
        holo = holo + astic45 * (np.power(self.rho,2)*np.sin(2*(self.phi)))
        holo = holo + horComa * ((3*np.power(self.rho,2)-2)*self.rho*np.cos(self.phi))
        holo = holo + verComa * ((3*np.power(self.rho,2)-2)*self.rho*np.sin(self.phi))
        holo = holo + verTref * (np.power(self.rho,3)*np.sin(3*(self.phi)))
        holo = holo + horTref * (np.power(self.rho,3)*np.cos(3*(self.phi)))
        holo = holo + spherical * (6*np.power(self.rho,4) - 6*np.power(self.rho,2) + 1)
        self.zernikeHolo = holo
        return holo




if __name__ == '__main__':
    screen_id = 0
    is_color = True
    a = Hologram()
    img = a.vortex(20)
    a.setHoloBase(0)
    img = a.lightspring(40,-3,7,5,6)
    plt.figure()
    ar = plt.imshow(img)
    plt.colorbar(ar)
    plt.show()
    a.plotModes()
    #showImage(1)


"""
Further Hologram Functions to implement

Lightspring with centered OAM
"""