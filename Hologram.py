import numpy as np
import cv2
import screeninfo
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from imageio import imread
import re

SLM_WIDTH_PX = 1280
SLM_HEIGHT_PX = 1024
SLM_PIXEL_LENGTH = 12.5e-3 #mm

DEFAULT_VOLTAGE_CALIBRATION = np.asarray([255/(2*np.pi),0])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class Hologram:
    def __init__(self, widthPx = SLM_WIDTH_PX , heightPx = SLM_HEIGHT_PX, 
                 pixelLengthMM = SLM_PIXEL_LENGTH):
        self.width = widthPx
        self.height = heightPx
        self.x, self.y = np.meshgrid(np.arange(-np.floor(self.width/2),np.floor(self.width/2)),
            np.arange(-np.floor(self.height/2),np.floor(self.height/2)))
        self.rho, self.phi = cart2pol(self.x, self.y)
        self.rho = self.rho*pixelLengthMM
        self.holoBase = np.zeros((self.height,self.width))
        self.phaseCalibration = np.zeros((self.height,self.width))
        self.voltageCalibration = DEFAULT_VOLTAGE_CALIBRATION
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

    def _finalizeHologram(self):
        self._applyPhaseCalibration()
        self._applyVoltageCalibration()
    
    def _linearInterpolRadius(self, n, OAM_start, deltaL = 1):
        omegas = np.linspace(self.omega_min,self.omega_max,n+1)
        radia = np.interp([self.omega_min, self.omega_max],
                          omegas,
                          [self.rMin, self.rMax])
        OAM = np.flip(np.linspace(OAM_start,OAM_start+(n-1)*deltaL))
        return radia, OAM

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
            :param rMin: minimum radius of the OAM zones
            :type rMin: float            
            :param rMax: maximum radius of the OAM zones
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
    
    def customFunction(self, boundaries, OAMs):
        """
        Creates a hologram with custom shaped OAM distribution

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
    
    def setHoloBase(self, mode = 0):
        def makeCheck(size):
            #implement
            check = np.zeros((self.height,self.width))
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

    def zernikeCorrection(self):
        """
        from current setup:
        pist = 0.; x_tilt = -0.; y_tilt = -0.; dfoc = 0; ver_ast=0.03; ob_ast = 0.00; h_coma = 0.00; v_coma = 0.0; v_tref = 0.; ob_tref = 0.; spher = 0.00;

%zernike corrections
function  Holo_correc  = zernike( pist, x_tilt , y_tilt , dfoc ,ver_ast, ob_ast, h_coma , v_coma ,v_tref,ob_tref, spher, H,V, RHO,PHI)

    Holo_0 = (zeros(V,H)+1)*pist;
    Holo_1 = x_tilt * (RHO.*cos(mod(PHI,2*pi))) ;
    Holo_2 = y_tilt* (RHO.*sin(mod(PHI,2*pi)) );
    Holo_3 = dfoc * ((2* ( RHO.* RHO) -1)) ;
    Holo_4 = ver_ast * (RHO.* RHO.*cos(2 * mod(PHI,2*pi)));
    Holo_5 = ob_ast * (( RHO.* RHO).*sin(2 * mod(PHI,2*pi)));
    Holo_6 = h_coma * ((3*( RHO.* RHO)-2) .* RHO .* cos(mod(PHI,2*pi)));
    Holo_7 = v_coma * ((3*( RHO.* RHO)-2) .* RHO .* sin(mod(PHI,2*pi)));
    Holo_8 = v_tref * (( RHO.* RHO.* RHO) .* sin(3*mod(PHI,2*pi)));
    Holo_9 = ob_tref * (( RHO.* RHO.* RHO) .* cos(3*mod(PHI,2*pi)));
    Holo_10 = spher * (6*(RHO.*RHO.*RHO.*RHO) - 6*(RHO.*RHO) + 1);
    
    
    Holo_correc =  Holo_0 + Holo_1 + Holo_2+ Holo_3+ Holo_4+ Holo_5+ Holo_6+ Holo_7+ Holo_8 + Holo_9 + Holo_10 ;   
end
        """
        return




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

function [radia, OAM] = linearInterpolRadius(n, OAM_start, deltaL)
arguments
    n (1,1) 
    OAM_start (1,1) 
    deltaL (1,1) = 1
end
    omega_max = 840e-9;
    omega_min = 774.405e-9;
    rMin = 5.63;
    rMax = 6.05;
    omegas = linspace(omega_min,omega_max,n+1);
    radia = interp1([omega_min,omega_max],[rMin, rMax],omegas);
    OAM = flip(OAM_start:deltaL:OAM_start+(n-1)*deltaL);
end

function [radia, OAM] = parabolaOAM(n, OAM_start, OAM_max)
%Makes parabolic profile of the OAM with the vertex at middle radius
arguments
    n (1,1) 
    OAM_start (1,1) 
    OAM_max (1,1)
end
    if mod(n,2) == 0
        ME = MException('parabolaOAM:ValueError',"n must be an odd integer number");
        throw(ME)
    end
    omega_max = 840e-9;
    omega_min = 774.405e-9;
    rMin = 5.63;
    rMax = 6.05;
    function p = parabola(x)
        h = ceil(max(n)/2);
        p = ((OAM_max-OAM_start)./(n-h).^2).*(x-h).^2+OAM_start;
    end
    omegas = linspace(omega_min,omega_max,n+1);
    radia = interp1([omega_min,omega_max],[rMin, rMax],omegas);
    OAM = ceil(parabola(linspace(1,n,n)));
end

function [radia, OAM] = stepfunctionOAM(n, step_pos,OAM_start, OAM_gap, delta_l1, delta_l2)
arguments
    n (1,1)
    step_pos (1,1) 
    OAM_start (1,1)
    OAM_gap (1,1)
    delta_l1 (1,1) = -1
    delta_l2 (1,1) = 1
end
    [radia, OAM1] = linearInterpolRadius(n, OAM_start+OAM_gap, delta_l1);
    [~, OAM2] = linearInterpolRadius(step_pos, OAM_start, delta_l2);
    OAM1(step_pos:end) = OAM2;
    OAM = OAM1;
end

function [holoOut, dphi] = applyGratingToSections(boundaries,holoIn,g, opts)
arguments
    boundaries (:,1)
    holoIn (1024,1280)
    g (1,1)
    opts.dphi (:,1) = -1
    opts.plot = false
end
    H=1280; V=1024; %%Number of Horizontal and Vertical pixels
    PixelSize = 12.5e-3; % [mm] pixel
    x = PixelSize.*(-H/2:(H/2-1)); %[mm] x width of window
    y = PixelSize.*(-V/2:(V/2-1)); %[mm] y width of window
    [X,Y] = meshgrid(x,y);
    [PHI,RHO] = cart2pol(X,Y);
    holoOut = holoIn;
    n = length(boundaries)-1;
    if opts.dphi == -1
        opts.dphi = 2*pi/n;
        opts.dphi = opts.dphi*linspace(1,n,n);
        dphi = opts.dphi;
    else
        if length(opts.dphi) ~= n
            ME = MException('applyGratingToSections:IndexError',"length of ´dphi´ must equal ´n´!");
            throw(ME) 
        end
    end
    for i = 1:1:n
        grating = mod(2*pi*(cos(opts.dphi(i))*g*X/H+sin(opts.dphi(i))*g*Y/V)./PixelSize,2*pi);
        if opts.plot
            figure()
            imagesc(grating)
            colormap("gray")
        end
        new = holoIn((RHO>boundaries(i))&(RHO<boundaries(i+1))) + grating((RHO>boundaries(i))&(RHO<boundaries(i+1)));
        holoOut((RHO>boundaries(i))&(RHO<boundaries(i+1))) = new;
    end
    holoOut = mod(holoOut,2*pi);
end

function outHolo = calibrationHolo(rMin, rMax, opts)
arguments
    rMin 
    rMax 
    opts.thickness = 0.4 %mm
    opts.nSample = 5
    opts.OAM = 60
    opts.deltaOAM = -1
    opts.g = 30
end
    %slm constants
    H=1280; V=1024; %%Number of Horizontal and Vertical pixels
    PixelSize = 12.5e-3; % [mm] pixel
    x = PixelSize.*(-H/2:(H/2-1)); %[mm] x width of window
    y = PixelSize.*(-V/2:(V/2-1)); %[mm] y width of window
    [X,Y] = meshgrid(x,y);
    [PHI,RHO] = cart2pol(X,Y);
    thickness = opts.thickness;
    nSample = opts.nSample;
    OAM = opts.OAM;
    if opts.deltaOAM == -1
        deltaOAM = round(OAM/4,0);
    else
        deltaOAM = opts.deltaOAM;
    end
    g = opts.g;
    %
    omega_max = 840e-9;
    omega_min = 774.405e-9;
    outerBoundaries = linspace(rMax,rMax+thickness,nSample+1);
    innerBoundaries = linspace(rMin-thickness,rMin,nSample+1);
    oMask = (RHO>outerBoundaries(1))&(RHO<outerBoundaries(end));
    iMask = (RHO>innerBoundaries(1))&(RHO<innerBoundaries(end));
    holo = zeros(V,H);
    innerHolo = mod((OAM-deltaOAM)*PHI ,2*pi);
    outerHolo = mod(OAM*PHI ,2*pi);
    holo(oMask) = outerHolo(oMask);
    holo(iMask) = innerHolo(iMask);
    [outHolo, dphi] = applyGratingToSections(outerBoundaries,holo,g);
    outHolo = applyGratingToSections(innerBoundaries,outHolo,g, dphi = flip(dphi));
    fprintf("Calibration: each sample of the radius has thickness %.2f mm;\ninner radius maximum: %.2f;\nouter radius minimum: %.2f",[thickness/nSample,rMin,rMax]);
end

"""