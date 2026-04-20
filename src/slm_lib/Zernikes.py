import math
import numpy as np
from zernpy import generate_polynomials, ZernPol, fit_polynomials
from zernpy.calculations.fit_zernike_pols import crop_phases_img
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from slm_lib.Hologram import SLM_WIDTH_PX, SLM_HEIGHT_PX, SLM_PIXEL_LENGTH, create_checkerboard
from copy import deepcopy

DEFAULT_PIXELPITCH = 29.6e-6  # [m] default pixel pitch for the source WFS, used for real-world normalization if not set otherwise

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def radial_zernike_extended(n: int, m: int, r):
    """
    Radial Zernike polynomial R_n^m(r) evaluated for arbitrary r >= 0.
    This also works for r > 1, i.e. outside the classical unit disk.
    """
    r = np.asarray(r, dtype=float)
    m = abs(int(m))
    n = int(n)

    if n < 0:
        raise ValueError("n must be >= 0")
    if m > n:
        raise ValueError("abs(m) must be <= n")
    if (n - m) % 2 != 0:
        return np.zeros_like(r, dtype=float)

    R = np.zeros_like(r, dtype=float)
    upper = (n - m) // 2
    for k in range(upper + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        R += c * r ** (n - 2 * k)
    return R


def zernike_extended(m: int, n: int, r, theta):
    """
    Real-valued Zernike polynomial using the common real basis:
      m = 0: R_n^0(r)
      m > 0: R_n^m(r) cos(m theta)
      m < 0: R_n^|m|(r) sin(|m| theta)
    for arbitrary r >= 0.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    R = radial_zernike_extended(n, m, r)

    if m == 0:
        return R
    elif m > 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(abs(m) * theta)
    
def device_physical_radius(devicePixelpitch, deviceRadiusPx, deviceCropRadius):
        return devicePixelpitch *deviceRadiusPx * deviceCropRadius
    
# Convenience function to handle and analyse files from a folder of phase files

def extract_coefficients_phasix(path:Path, crop_radius=0.95, polinomial_order=4):
    """
    Convenience function to load phase files from a folder based on the phasix-WFS structure, average them, fit Zernikes, and save the coefficients to a file.
        :param path: Path to the folder containing the phase files (expects files named "PHA*.csv")
        :type path: Path
        :param crop_radius: Crop radius for the pupil in the source image, defined as the beam radius normalized to the image radius. E.g. 0.9 means the beam radius is 90% of the image radius. Default is 0.95.
        :type crop_radius: float
        :param polinomial_order: Maximum radial order of Zernike polynomials to fit. Default is 4.
        :type polinomial_order: int
    """
    phase_file = PhaseFile.average_files_phasix(path, crop_radius)
    phase_file.fit_zernikes_to_file(generate_polynomials(polinomial_order))
    save_path = path.parent.parent / f"{path.parent}_coefficients"
    phase_file.save_zernike_coefficients(save_path)
    return phase_file.coefficients

def load_zernike_coefficients_from_file(path:Path)-> tuple[list[float], list[ZernPol], float]:
    """Load Zernike coefficients and corresponding polynomials from a .zrnk file.
    returns the coefficients, the corresponding ZernPol objects, and the real-world normalization factor for the source measurement (physical radius of the fitted pupil in the source image).
    Use PhaseFile(path) to obtain a PhaseFile object with the coefficients and Zernikes loaded, and the normalization factor set as an attribute. This function is useful if you just want to load the coefficients and Zernikes without creating a PhaseFile object.
        :param path: Path to the .zrnk file containing Zernike coefficients and polynomial information
        :type path: Path
        :return: Tuple containing the list of coefficients, list of corresponding ZernPol objects, and the real-world normalization factor
        :rtype: tuple[list[float], list[ZernPol], float]
    """
    if not path.exists():
        raise ValueError(f"File {path} does not exist.")
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")
    if not path.suffix == ".zrnk":
        raise ValueError(f"File {path} does not have .zrnk extension.")
    
    def get_delimiter_from_header(header_line:str):
        pn = "Polynomial Name"
        left = header_line.find(pn)+len(pn)
        right = header_line.strip(pn).find("m")+len(pn)
        return header_line[left:right].strip()
    
    coefficients = []
    zernikes = []
    realworldNormalizationFactor = None
    with open(path, 'r') as f:
        delimiter = get_delimiter_from_header(next(f))  # read header and determine delimiter
        for line in f:
            name, m_str, n_str, coef_str, realworldNormalizationFactor_str = line.strip().split(delimiter)
            m = int(m_str)
            n = int(n_str)
            coef = float(coef_str)
            realworldNormalizationFactor = float(realworldNormalizationFactor_str)
            coefficients.append(coef)
            zernikes.append(ZernPol(m=m, n=n))
    return coefficients, zernikes, realworldNormalizationFactor

def scale_zernike_coefficients_raw(coefficients: list[float], zernikes: list[ZernPol], scaleFactor: float)->list[float]:
        """
        Static version of transform_pupil_size to scale Zernike coefficients for a different circular(!) pupil size.
            :param coefficients: List of Zernike coefficients to scale
            :type coefficients: list[float]
            :param zernikes: List of ZernPol objects corresponding to the coefficients, used to determine the radial order for scaling
            :type zernikes: list[ZernPol]
            :param scaleFactor: The factor by which to scale the pupil radius (e.g. R_dev / R_fit)
            :type scaleFactor: float
            :return: A new list of Zernike coefficients corresponding to the scaled pupil size.
            :rtype: list[float]
        """
        if len(coefficients) != len(zernikes):
            raise ValueError("Number of coefficients must match number of Zernikes.")
        
        transformed_coeffs = []
        for a, z in zip(coefficients, zernikes):
            m, n = z.get_mn_orders()
            transformed_a = a * (scaleFactor ** n)
            transformed_coeffs.append(transformed_a)
        return transformed_coeffs

def normalize_zernike_coefficients_raw(coeffs: list[float], zernikes: list[ZernPol], normalization_factor: float)->list[float]:
    """
    Normalize Zernike coefficients by scaling them according to the radial order of the corresponding Zernike polynomials and a given normalization factor. 
    This is useful for comparing coefficients obtained from different measurements or devices with different pupil sizes.
    Its intended to use the real-world normalization factor obtained from the source measurement as the normalization_factor, which corresponds to the physical radius of the fitted pupil in the source image. This way, the coefficients are normalized to a common physical scale, allowing for meaningful comparison and transformation to target devices with different pupil sizes.
        :param coeffs: List of Zernike coefficients to normalize
        :type coeffs: list[float]
        :param zernikes: List of ZernPol objects corresponding to the coefficients, used to determine the radial order for scaling
        :type zernikes: list[ZernPol]
        :param normalization_factor: The factor by which to normalize the coefficients, typically the individual real-world normalization factor for the source measurement.
        :type normalization_factor: float
    """
    return scale_zernike_coefficients_raw(coeffs, zernikes, 1/normalization_factor)

class ZernikeAnalysisUtils:
    """
    static utility functions for comparing and subtracting Zernike coefficients and polynomials from different measurements, without the need to create PhaseFile objects. These functions operate on raw lists of coefficients and ZernPol objects, along with the necessary normalization factors, to perform the desired comparisons and transformations. This allows for more flexible analysis of Zernike data without being tied to the PhaseFile class structure.
    """
    @staticmethod
    def make_polar_mesh(shape, crop_radius=1.0, phase_offset = np.pi*3/2, crop_pupil=True):
        """
        Polar mesh on the target device, normalized to the target device pupil.
        r_dev_norm = 1 at the target pupil edge.

            :param shape: Shape of the target device (height, width)
            :type shape: tuple[int]
            :param crop_radius: Normalized Crop radius for the target device pupil [0.5-1.0], defined as the beam radius normalized to the device physical radius. E.g. 0.9 means the beam radius is 90% of the device physical radius.
            :type crop_radius: float
            :param phase_offset: Phase offset for the polar mesh
            :type phase_offset: float
            :param crop_pupil: Whether to crop the pupil
            :type crop_pupil: bool

        """
        print(f"Making device polar mesh with\nshape={shape}, crop_radius={crop_radius}, phase_offset={phase_offset}")
        # Calculate center of the input image - depending on its sizes
        rows, cols = shape
        img_min_size = min(rows, cols)
        img_max_size = max(rows, cols)
        if rows % 2 == 0:
            center_row = (rows // 2) - 0.5
        else:
            center_row = (rows - 1) // 2
        if cols % 2 == 0:
            center_col = (cols // 2) - 0.5
        else:
            center_col = (cols - 1) // 2
        # Define the radius for cropping, different rules applied for avoiding removing many pixels out
        if img_min_size % 2 == 0:
            r = (img_min_size // 2) - 0.5
        else:
            r = (img_min_size - 1) // 2
        if img_max_size - img_min_size < 2:
            if img_max_size % 2 == 0:
                r = (img_max_size // 2) - 0.5
            else:
                r = (img_max_size - 1) // 2
        cy = center_row
        cx = center_col
        R0 = r * crop_radius

        y, x = np.indices((rows, cols))
        yy = y - cy 
        xx = x - cx 

        r = np.sqrt(yy**2 + xx**2) / R0
        theta = (np.arctan2(xx, yy) + phase_offset)%(2*np.pi)  # keep same axis convention as your original code if desired
        if crop_pupil:
            r[r > 1] = np.nan
            theta[r > 1] = np.nan
        return r, theta

    @staticmethod
    def evaluate_zernikes_on_mesh(coefficients: list[float], zernikes: list[ZernPol], r_mesh, theta_mesh):
        """
        Evaluate the Zernike sum on a given polar mesh.
            :param coefficients: List of Zernike coefficients
            :type coefficients: list[float]
            :param zernikes: List of Zernike polynomials corresponding to the coefficients
            :type zernikes: list[ZernPol]
            :param r_mesh: Radial coordinates of the mesh
            :type r_mesh: np.ndarray
            :param theta_mesh: Angular coordinates of the mesh
            :type theta_mesh: np.ndarray
            :return: The evaluated phase pattern on the mesh corresponding to the Zernike sum.
            :rtype: np.ndarray
        """
        if len(coefficients) != len(zernikes):
            raise ValueError("Number of coefficients must match number of Zernikes.")
        
        total_phase = np.zeros_like(r_mesh)
        for a, z in zip(coefficients, zernikes):
            m, n = z.get_mn_orders()
            total_phase += a * zernike_extended(m, n, r_mesh, theta_mesh)
        return total_phase

    @staticmethod
    def compare_zernikes_raw(measured_zernikes: list[ZernPol], reference_zernikes: list[ZernPol])->tuple[list[int], list[int], list[tuple[int, int]]]:
        """
        Compare measured and reference Zernike polynomials, returns the indices of the same polynomials in both lists and their corresponding (m, n) orders.
            :param measured_zernikes: List of measured Zernike polynomials.
            :type measured_zernikes: list[ZernPol]
            :param reference_zernikes: List of reference Zernike polynomials.
            :type reference_zernikes: list[ZernPol]
            :return: List of indices of same polynomials.
            :rtype: tuple[list[int], list[int], list[tuple[int, int]]]
        """
        if len(measured_zernikes) != len(reference_zernikes):
            print("Warning: Measured and reference Zernike lists dont have the same length.")

        ref_zernike_indices = [z.get_mn_orders() for z in reference_zernikes]
        measured_zernike_indices = [z.get_mn_orders() for z in measured_zernikes]
        ref = []
        measured = []
        index = []
        for m in measured_zernike_indices:
            if m in ref_zernike_indices:
                ref.append(ref_zernike_indices.index(m))
                measured.append(measured_zernike_indices.index(m))
                index.append(m)
        return ref, measured, index

    @staticmethod
    def subtract_coefficients_raw(measured_coeffs: list[float], measured_zernikes: list[ZernPol], reference_coeffs: list[float], reference_zernikes: list[ZernPol], measured_normalization_factor: float = 1, reference_normalization_factor: float = 1)->tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Subtract reference Zernike coefficients from measured coefficients to isolate the aberrations of interest.
            :param measured_coeffs: List of Zernike coefficients obtained from the measurement.
            :type measured_coeffs: list[float]
            :param measured_zernikes: List of Zernike polynomials corresponding to the measured coefficients.
            :type measured_zernikes: list[ZernPol]
            :param reference_coeffs: List of Zernike coefficients representing the reference or baseline aberrations to subtract.
            :type reference_coeffs: list[float]
            :param reference_zernikes: List of Zernike polynomials corresponding to the reference coefficients.
            :type reference_zernikes: list[ZernPol]
            :param measured_normalization_factor: The factor by which to normalize the measured coefficients.
            :type measured_normalization_factor: float
            :param reference_normalization_factor: The factor by which to normalize the reference coefficients.
            :type reference_normalization_factor: float
            :return: List of Zernike coefficients after subtraction, and the corresponding (m, n) orders of the Zernike polynomials that were compared.
            :rtype: tuple[np.ndarray, list[tuple[int, int]]]
        """
        if reference_normalization_factor == 1 and measured_normalization_factor == 1:
            print("Warning: No normalization applied to coefficients. Make sure the coefficients are on the same scale before subtraction.")

        ref_ind, meas_ind, zern_indices = ZernikeAnalysisUtils.compare_zernikes_raw(measured_zernikes, reference_zernikes)
        measured_coeffs = normalize_zernike_coefficients_raw(measured_coeffs, measured_zernikes, measured_normalization_factor)
        reference_coeffs = normalize_zernike_coefficients_raw(reference_coeffs, reference_zernikes, reference_normalization_factor)
        measured_coeffs = np.asarray(measured_coeffs)[np.asarray(meas_ind)]
        reference_coeffs = np.asarray(reference_coeffs)[np.asarray(ref_ind)]
        return measured_coeffs - reference_coeffs, zern_indices

    @staticmethod
    def compare_zernikes(measured: PhaseFile, reference: PhaseFile)->tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Compare measured and reference Zernike coefficients from two PhaseFile objects, returns the coefficients for the same polynomials and their corresponding (m, n) orders.
            :param measured: PhaseFile object containing the measured Zernike coefficients and polynomials.
            :type measured: PhaseFile
            :param reference: PhaseFile object containing the reference Zernike coefficients and polynomials.
            :type reference: PhaseFile
            :return: List of Zernike coefficients after subtraction, and the corresponding (m, n) orders of the Zernike polynomials that were compared.
            :rtype: tuple[np.ndarray, list[tuple[int, int]]]
        """
        if not measured._hasZernikCoefficients():
            raise ValueError("Measured PhaseFile does not have coefficients or Zernikes.")
        if not reference._hasZernikCoefficients():
            raise ValueError("Reference PhaseFile does not have coefficients or Zernikes.")
        
        new = ZernikeAnalysisUtils.compare_zernikes_raw(
            measured.zernikes,
            reference.zernikes
        )
        return PhaseFile._fromZernikes(new[0], new[1], None)  # create an uninitialized instance

    @staticmethod
    def subtract_zernikes(measured: PhaseFile, reference: PhaseFile, basis:str = "None")->PhaseFile:
        """
        Subtract measured and reference Zernike coefficients from two PhaseFile objects, returns the difference of the coefficients for the same polynomials and their corresponding (m, n) orders.
            :param measured: PhaseFile object containing the measured Zernike coefficients and polynomials.
            :type measured: PhaseFile
            :param reference: PhaseFile object containing the reference Zernike coefficients and polynomials.
            :type reference: PhaseFile
            :param basis: "None" (Default), "measured", or "reference". Determines the basis for the resulting coefficients. "None" means both are normalized and the coefficients are returned on to global scale. "measured" means the resulting coefficients are normalized to the measured coefficients' normalization factor, and "reference" means they are normalized to the reference coefficients' normalization factor. This is important for ensuring that the resulting coefficients are on a meaningful scale for comparison or further analysis.
            :return: PhaseFile object containing the difference of Zernike coefficients for the same polynomials, and the corresponding (m, n) orders of the Zernike polynomials that were compared. The real-world normalization factor is set to 1, indicating that the coefficients are normalized to a common physical scale.
            :rtype: PhaseFile
        """
        if not measured._hasZernikCoefficients():
            raise ValueError("Measured PhaseFile does not have coefficients or Zernikes.")
        if not reference._hasZernikCoefficients():
            raise ValueError("Reference PhaseFile does not have coefficients or Zernikes.")
        if not basis in ["None", "measured", "reference"]:
            raise ValueError("Basis must be 'None', 'measured', or 'reference'.")
        
        new = ZernikeAnalysisUtils.subtract_coefficients_raw(
            measured.coefficients,
            measured.zernikes,
            reference.coefficients,
            reference.zernikes,
            measured.realworldNormalizationFactor,
            reference.realworldNormalizationFactor
        )
        new[0]
        zernike_list = [ZernPol(m=m, n=n) for m, n in new[1]]
        if basis == "measured":
            return PhaseFile._fromZernikes(new[0], zernike_list, 1, f"{measured.path.stem} - {reference.path.stem}.zrnk").scale_zernike_coefficients(measured.realworldNormalizationFactor)  # create an uninitialized instance 
        elif basis == "reference":
            return PhaseFile._fromZernikes(new[0], zernike_list, 1, f"{measured.path.stem} - {reference.path.stem}.zrnk").scale_zernike_coefficients(reference.realworldNormalizationFactor)  # create an uninitialized instance
        
        return PhaseFile._fromZernikes(new[0], zernike_list, 1, f"{measured.path.stem} - {reference.path.stem}.zrnk")  # create an uninitialized instance

#   plotting
class PlottingUtils:
    @staticmethod
    def plot_phase_on_axes(ax:Axes, phase, title = "Phase", cmap="viridis", wrapped = True):
        """
        Utility function to plot a phase array on given axes with consistent color scaling and optional wrapping.
            :param ax: Matplotlib axes to plot on
            :type ax: Axes
            :param phase: 2D array of phase values to plot
            :type phase: np.ndarray
            :param title: Title for the plot, default "Phase"
            :type title: str
            :param cmap: Colormap to use, default "viridis"
            :type cmap: str
            :param wrapped: Whether to wrap the phase values to [0, 2pi] for plotting, default True
            :type wrapped: bool
        """
        if wrapped:
            phase_to_plot = np.mod(phase, 2*np.pi)
        else:
            phase_to_plot = phase
        c = ax.imshow(phase_to_plot, cmap=cmap, vmin=0 if wrapped else None, vmax=2*np.pi if wrapped else None)
        ax.set_title(title)
        plt.colorbar(c, ax=ax)

    @staticmethod
    def plot_coefficients_on_axes(ax:Axes, coefficients: list[float], zernikes: list[ZernPol], exclude_indices: list[int] = [], title: str = "Zernike Coefficients", names = True):
        """ 
        Utility function to plot Zernike coefficients as a bar chart on given axes, with labels and optional exclusion of certain indices.
            :param ax: Matplotlib axes to plot on
            :type ax: Axes
            :param coefficients: List of Zernike coefficients to plot
            :type coefficients: list[float]
            :param zernikes: List of Zernike polynomials corresponding to the coefficients, used for labeling
            :type zernikes: list[ZernPol]
            :param exclude_indices: List of indices of Zernike orders to exclude from the plot (e.g. piston, tip, tilt), default []
            :type exclude_indices: list[int]
            :param names: Whether to include polynomial names in the labels, default True
            :type names: bool
        """
        if names:
            name_labels = [f"{z.get_polynomial_name()} ({z.get_mn_orders()[0]}, {z.get_mn_orders()[1]})" 
                            for i, z in enumerate(zernikes) if i not in exclude_indices]
        else:
            name_labels = [f"({z.get_mn_orders()[0]}, {z.get_mn_orders()[1]})" 
                           for i, z in enumerate(zernikes) if i not in exclude_indices]
        coefficients = [coef for i, coef in enumerate(coefficients) if i not in exclude_indices]
        ax.bar(name_labels, coefficients)
        ax.set_title(title)
        ax.set_ylabel("Coefficient Value")
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()

class PhaseFile:
    """
    Load phase data and fit/evaluate Zernikes.
    _____________________________________________
    Workflow:
    1. Load phase data from file using PhaseFile.loadFile(path).
    2. Set the crop_radius to define the normalized pupil size in the source image.
    3. Set the pixel pitch of the source WFS to define the real-world normalization (if differs from default)
    4. Define the Zernike modes to fit using generate_polynomials.
    5. Fit Zernikes to the phase data using fitZernikesToFile(zernikes).
    6. For target device mapping, define the target device parameters (pixel pitch, radius in pixels, crop radius).
    7. Use get_scale_factor_to_device to calculate the scaling factor for the Zernike coefficients based on the source and device pupil sizes.
    8. Use transform_pupil_size to get the transformed Zernike coefficients for the target device.
    9. Evaluate the transformed Zernikes on the target device's polar mesh using evaluate_zernikes_on_mesh to get the phase pattern to apply on the device.
    """
    def __new__(cls, *args, **kwargs):
        instance = super(PhaseFile, cls).__new__(cls)
        instance.zernikes = None
        instance.coefficients = None
        instance.pixelpitch = None
        instance.fileSize = None
        instance.minSize = None
        instance.realworldNormalizationFactor = None
        instance.path = None
        instance.crop_radius = None
        
        return instance

    def __init__(self, path: Path, delimiter=",", crop_radius=0.5, pixelpitch=DEFAULT_PIXELPITCH):
        self.zernikes: list[ZernPol] | None = None
        self.coefficients: list[float] | None = None
        self.pixelpitch = None
        self.fileSize = None
        self.minSize = None
        self.realworldNormalizationFactor = None
        self.path = path
        self.crop_radius = None
        if not self._isZernikeFile():
            self.phase = np.loadtxt(path, delimiter=delimiter)
            self.fileSize = self.phase.shape
            self.minSize = min(self.fileSize)
            self.crop_radius = crop_radius
            self.set_pixelpitch(pixelpitch)
            self.realworldNormalizationFactor = self._calcNormalizationFactor()
        else:
            self._loadZernikeCoefficientsFromFile()

    def _setZernikes(self, zernikes: list[ZernPol]):
        self.zernikes = zernikes

    def _createNormPolarMesh(self):
        """
        Normalized polar mesh for the source image.
        Radius is normalized so r=1 corresponds to the fitted pupil edge.
        Outside that circle, r is set to NaN.
        """
        c_r = self.crop_radius
        if self.crop_radius is None:
            c_r = 1

        if self.fileSize is None:
            h, w = int(SLM_HEIGHT_PX), int(SLM_WIDTH_PX)

        else:
            h, w = self.fileSize
        minSize = min(h, w)/2
        cy = np.ceil(h / 2)
        cx = np.ceil(w / 2)
        R0 = minSize * c_r

        y, x = np.indices((h, w))
        yy = y - cy
        xx = x - cx

        r = np.sqrt(yy**2 + xx**2) / R0
        theta = np.arctan2(xx, yy)  # keep same axis convention as your original code if desired

        r[r > 1] = np.nan
        return r, theta
    
    def _isZernikeFile(self):
        return self.path.suffix == ".zrnk"
    
    def _loadZernikeCoefficientsFromFile(self):
        def get_delimiter_from_header(header_line:str):
            pn = "Polynomial Name"
            left = header_line.find(pn)+len(pn)
            right = header_line.strip(pn).find("m")+len(pn)
            return header_line[left:right].strip()
            
        coefficients = []
        zernikes = []
        realworldNormalizationFactor = None
        with open(self.path, 'r') as f:
            delimiter = get_delimiter_from_header(next(f))  # read header and determine delimiter
            for line in f:
                name, m_str, n_str, coef_str, realworldNormalizationFactor_str = line.strip().split(delimiter)
                m = int(m_str)
                n = int(n_str)
                coef = float(coef_str)
                realworldNormalizationFactor = float(realworldNormalizationFactor_str)
                coefficients.append(coef)
                zernikes.append(ZernPol(m=m, n=n))
        self.coefficients = coefficients
        self.zernikes = zernikes
        self.realworldNormalizationFactor = realworldNormalizationFactor
        self.crop_radius = None
        self.phase = self.phase_from_fit(wrapped = False)  # reconstruct the phase from the coefficients and Zernikes for potential further analysis or plotting

    def _calcNormalizationFactor(self):
        #in m, corresponds to the physical radius of the fitted pupil in the source image, used for normalizing coefficients to a common physical scale
        if self.pixelpitch is None or self.crop_radius is None:
            return None
        return self.minSize / 2 * self.pixelpitch * self.crop_radius
    
    def _hasZernikCoefficients(self):
        return self.coefficients is not None and self.zernikes is not None
    ##############################################################################################
    # Static methods
    @staticmethod
    def _fromZernikes(coefficients: list[float], zernikes: list[ZernPol], normalization_factor:float, filename: str = "None.zrnk") -> "PhaseFile":
        """
        Create a PhaseFile object from given Zernike coefficients and polynomials, along with the necessary parameters to define the normalization factor.
            :param coefficients: List of Zernike coefficients
            :type coefficients: list[float]
            :param zernikes: List of Zernike polynomials corresponding to the coefficients
            :type zernikes: list[ZernPol]
            :param fileSize: Size of the original phase file (height, width) used for fitting, needed to define the normalization factor
            :type fileSize: tuple[int, int]
            :param crop_radius: Crop radius used for fitting, needed to define the normalization factor
            :type crop_radius: float
            :param pixelpitch: Pixel pitch used for fitting, needed to define the normalization factor
            :type pixelpitch: float
        """
        p = PhaseFile.__new__(PhaseFile)  # create an uninitialized instance
        p.zernikes = zernikes
        p.coefficients = coefficients
        p.realworldNormalizationFactor = normalization_factor
        p.path = Path(filename).with_suffix(".zrnk")
        return p

    #   loading and averaging phase files
    @staticmethod
    def load_file(path, delimiter=",", crop_radius=0.5, pixelpitch=DEFAULT_PIXELPITCH):
        """
        Based on numpy loadtxt, but returns a PhaseFile object with the loaded phase data and parameters set.
            :param path: Path to the file containing phase data
            :type path: Path or str
            :param delimiter: Delimiter used in the file, default ","
            :type delimiter: str
            :param crop_radius: 0.5
            :type crop_radius: float
            :param pixelpitch: Pixelpitch of device, defaultDEFAULT_PIXELPITCH
            :type pixelpitch: float
        """
        return PhaseFile(path, delimiter, crop_radius, pixelpitch)
    
    @staticmethod
    def average_files_phasix(folder:Path, crop_radius=0.5):
        """
        Load multiple phasefiles from the given folder and average them.
        Returns a PhaseFile object with the averaged phase and the given crop_radius.
            :param folder: Path to the folder containing the files
            :type folder: Path
            :param crop_radius: crop radius for analysis, 0.5
            :type crop_radius: float
        """
        phase_files = sorted(folder.glob("PHA*.csv"))
        if not phase_files:
            raise ValueError(f"No files found in {folder} matching pattern 'PHA*.csv'")

        phases = []
        for f in phase_files:
            phase_data = np.loadtxt(f, delimiter=",")
            phases.append(phase_data)

        avg_phase = np.mean(np.asarray(phases), axis=0)
        avg_phase_file = PhaseFile.__new__(PhaseFile)  # create an uninitialized instance
        avg_phase_file.phase = avg_phase
        avg_phase_file.fileSize = avg_phase.shape
        avg_phase_file.minSize = min(avg_phase_file.fileSize)
        avg_phase_file.crop_radius = crop_radius
        avg_phase_file.pixelpitch = DEFAULT_PIXELPITCH
        avg_phase_file.realworldNormalizationFactor = avg_phase_file._calcNormalizationFactor()
        return avg_phase_file
        
    #   polar mesh and Zernike evaluation
    
    def set_pixelpitch(self, pixelpitch):
        self.pixelpitch = pixelpitch
        self.realworldNormalizationFactor = self._calcNormalizationFactor()

    def fit_zernikes_to_file(self, zernikes: list[ZernPol]):
        self._setZernikes(zernikes)
        fit = fit_polynomials(
            self.phase,
            self.zernikes,
            crop_radius=self.crop_radius,
            return_cropped_image=True,
        )
        self.coefficients = fit[0]
        return fit

    def source_physical_radius(self):
        """
        Physical radius of the fitted source pupil.
        """
        if self.realworldNormalizationFactor is None:
            raise ValueError("Need both pixelpitch and crop_radius to define source physical radius.")
        return self.realworldNormalizationFactor
        
    def get_scale_factor_to_device(self, devicePixelpitch, deviceRadiusPx, deviceCropRadius):
        """
        Scale factor s = R_dev / R_fit
        """
        R_fit = self.source_physical_radius()
        print(f"Source physical radius: {R_fit} m")
        R_dev = device_physical_radius(devicePixelpitch, deviceRadiusPx, deviceCropRadius)
        print(f"Device physical radius: {R_dev} m")
        return R_dev / R_fit
    
    def transform_pupil_size(self, scaleFactor)->list[float]:
        """
        Transform the fitted Zernike coefficients to correspond to a different pupil size.
        This is based on the scaling property of Zernike polynomials: R_n^m(r) ~ r^n.
        The new coefficients are a_k' = a_k * (scaleFactor^n_k) where n_k is the radial order of the k-th Zernike.

        :param scaleFactor: The factor by which to scale the pupil radius (e.g. R_dev / R_fit)
        :type scaleFactor: float
        :return: A new list of Zernike coefficients corresponding to the scaled pupil size.
        :rtype: list[float]
        """
        if not self._hasZernikCoefficients():
            raise ValueError("Fit the Zernikes first.")

        transformed_coeffs = []
        for a, z in zip(self.coefficients, self.zernikes):
            m, n = z.get_mn_orders()
            # The scaling of Zernike polynomials with radius is R_n^m(r) ~ r^n
            transformed_a = a * (scaleFactor ** n)
            transformed_coeffs.append(transformed_a)   
        return transformed_coeffs
    
    def normalize(self):
        """
        Normalize the Zernike coefficients by scaling them according to the radial order of the corresponding Zernike polynomials and the real-world normalization factor for the source measurement.
        This is useful for comparing coefficients obtained from different measurements or devices with different pupil sizes.
        Retruns a new PhaseFile object with the normalized coefficients and the normalization factor set to 1, indicating that the coefficients are now normalized to a common physical scale.
        To normalize everything to the basis r=[0,1] used for fitting, use the real-world normalization factor obtained from the source measurement as the normalization_factor, which corresponds to the physical radius of the fitted pupil in the source image. This way, the coefficients are normalized to a common physical scale, allowing for meaningful comparison and transformation to target devices with different pupil sizes.
        - Math:
            Zernikes real world: Z(R) [Z]=rad, [R]=m
            Zernikes normalized: Z(r) [Z]=rad, [r]=1
             Z = a_r * R; [a_r] = rad/m
             R = r * scaleFactor; [scaleFactor] = m
             Z = a_r * r * scaleFactor = a_n * r; [a_n] = rad
             scaleFactor = crop_radius * pixelpitch * min(fileSize)/2 = physical radius of the fitted pupil in the source image
             a_n actually corresponds to the value obtained by Zernpy.fit
             a_n = a_r * scaleFactor = a_r * physical radius of the fitted pupil in the source image
            Therefore you need to normalize the coefficients by dividing them by the physical radius of the fitted pupil in the source image to get the normalized coefficients corresponding to the basis r=[0,1] used for fitting.

        """
        if self.realworldNormalizationFactor is None:
            raise ValueError("Real-world normalization factor is not defined. Make sure pixelpitch and crop_radius are set for the source measurement.")
        P = deepcopy(self)
        P.coefficients = self.transform_pupil_size(1/self.realworldNormalizationFactor)
        P.realworldNormalizationFactor = 1
        return P
    
    def scale_zernike_coefficients(self, scale_factor):
        """
        Scale the Zernike coefficients by a given factor, without changing the corresponding Zernike polynomials. This is a simple scaling of the coefficients and does not take into account the scaling properties of the Zernike polynomials with radius. Use transform_pupil_size for that purpose.
            :param scale_factor: The factor by which to scale the coefficients (e.g. to adjust for different measurement units or to apply a global scaling)
            :type scale_factor: float
            :return: A new list of Zernike coefficients scaled by the given factor.
            :rtype: list[float]
        """
        if not self._hasZernikCoefficients():
            raise ValueError("Fit the Zernikes first.")
        p = deepcopy(self)
        p.coefficients = self.transform_pupil_size(scale_factor)
        p.realworldNormalizationFactor *= scale_factor
        return p
    
    def phase_from_fit(self, wrapped = True, r_mesh = None, theta_mesh = None):
        """
        Evaluate the Phasefront corresponding to the fitted Zernike coefficients and polynomials on the normalized polar mesh of the source image. This returns the phase pattern that corresponds to the fitted Zernikes, which can be used for visualization or for further analysis such as comparison with target device patterns or for generating compensation patterns.
        """
        if not self._hasZernikCoefficients():
            raise ValueError("Fit the Zernikes first.")
        if r_mesh is None or theta_mesh is None:
            r_mesh, theta_mesh = self._createNormPolarMesh()
        if wrapped:
            return np.mod(ZernikeAnalysisUtils.evaluate_zernikes_on_mesh(self.coefficients, self.zernikes, r_mesh, theta_mesh), 2*np.pi)
        return ZernikeAnalysisUtils.evaluate_zernikes_on_mesh(self.coefficients, self.zernikes, r_mesh, theta_mesh)
    
    def plot_phase(self, title = "Phase", cmap="viridis", wrapped = True):
        fig, ax = plt.subplots(figsize=(6,6))
        PlottingUtils.plot_phase_on_axes(ax, self.phase_from_fit(wrapped=wrapped), title, cmap, wrapped)
        return fig
    
    def plot_zernike_modes(self, wrapped = False, cmap="coolwarm"):
        if not self._hasZernikCoefficients():
            raise ValueError("No Zernikes to plot. Fit the Zernikes first.")
        
        fig, axs = plt.subplots(1, len(self.zernikes), figsize=(4*len(self.zernikes), 4))
        for z, ax in zip(self.zernikes, axs):
            m, n = z.get_mn_orders()
            mode_phase = zernike_extended(m, n, *self._createNormPolarMesh())*self.coefficients[self.zernikes.index(z)]
            PlottingUtils.plot_phase_on_axes(ax, mode_phase, title=f"{z.get_polynomial_name()}\nZernike m={m}, n={n}", cmap=cmap, wrapped=wrapped)
        fig.tight_layout()
        fig.show()
        return fig
    
    def plot_coefficients_on_axes(self, ax:Axes, exclude_indices: list[int] = [], title = None, names = True):
        """
        Plot Zernike coefficients as a bar chart on given axes, with labels and optional exclusion of certain indices.
            :param ax: Matplotlib axes to plot on
            :type ax: Axes
            :param exclude_indices: List of indices of Zernike orders to exclude from the plot (e.g. piston, tip, tilt), default []
            :type exclude_indices: list[int]
            :param title: Title for the plot, default "Zernike Coefficients"
            :type title: str
            :param names: Whether to include polynomial names in the labels, default True
            :type names: bool
        """
        if title is None:
            title = f"{self.path.stem}"
        PlottingUtils.plot_coefficients_on_axes(ax, self.coefficients, self.zernikes, exclude_indices, title, names)
    
    def plot_coefficients(self, exclude_indices = [], title = None, names = True):
        if not self._hasZernikCoefficients():
            raise ValueError("No coefficients to plot. Fit the Zernikes first.")
        
        fig, ax = plt.subplots(figsize=(8,6))
        PlottingUtils.plot_coefficients_on_axes(ax, self.coefficients, self.zernikes, exclude_indices, title, names)
        return fig
    

    def make_SLM_phase_pattern(self, deviceCropRadius, devicePixelpitch=SLM_PIXEL_LENGTH, shape = (SLM_HEIGHT_PX, SLM_WIDTH_PX), checkerboard = None, wrapped = True, invertCoefficients = False):
        """
        Convenience function to get the phase pattern to apply on the SLM/device directly from the fitted Zernikes and target device parameters.
        This combines the scaling of coefficients and evaluation on the device polar mesh.

            :param deviceCropRadius: Crop radius on the device where Zernikes are evaluated. Refferes to the Beamradius normalized to the device physical radius. E.g. 0.9 means the beam radius is 90% of the device physical radius.
            :type deviceCropRadius: float
            :param devicePixelpitch: Pixelpitch of the SLM/Screen. Default is Hologram.SLM_PIXEL_LENGTH.
            :type devicePixelpitch: float
            :param shape: Shape of the SLM/Screen. Default is (Hologram.SLM_HEIGHT_PX, Hologram.SLM_WIDTH_PX).
            :type shape: tuple[int]
            :param checkerboard: Optional checkerboard pattern on the outside of the phase pattern. If None, no checkerboard is applied. If int, defines the size of the checkerboard squares in pixels.
            :type checkerboard: int or None
            :param wrapped: Whether to wrap the phase values. Default is True.
            :type wrapped: bool
            :param invertCoefficients: Whether to invert the sign of the Zernike coefficients. This can be useful if the SLM adds a phase delay and you want to pre-compensate for it. Default is False.
            :type invertCoefficients: bool
            :return: Phase pattern to apply on the SLM/device.
        """
        if not self._hasZernikCoefficients():
            raise ValueError("Fit the Zernikes first.")
        if checkerboard is not None and (not isinstance(checkerboard, int) or checkerboard <= 0):
            raise ValueError("Checkerboard must be a positive integer defining the square size in pixels, or None to disable.")
        background_pattern = np.zeros(shape)
        if checkerboard is not None:
            background_pattern = create_checkerboard(*shape, checkerboard)
        deviceRadiusPx=np.max(shape)//2
        scaleFactor = self.get_scale_factor_to_device(devicePixelpitch, deviceRadiusPx, deviceCropRadius)
        transformed_coeffs = self.transform_pupil_size(scaleFactor)
        print(f"Scale factor to device: {scaleFactor}\nTransformed coefficients: {transformed_coeffs}")
        r_mesh, theta_mesh = self.make_polar_mesh(shape, crop_radius=deviceCropRadius)
        print(f"Device polar mesh created with shape {r_mesh.shape} and crop_radius {deviceCropRadius}: r_mesh max={np.nanmax(r_mesh)}, r_mesh min={np.nanmin(r_mesh)}")
        beam = self.evaluate_zernikes_on_mesh(transformed_coeffs, self.zernikes, r_mesh, theta_mesh)
        if invertCoefficients:
            beam = -beam
        if wrapped:
            beam = np.mod(beam, 2*np.pi)
        return np.where(np.isnan(beam), background_pattern, beam)

    def save_zernike_coefficients(self, save_path: Path, delimiter=","):
        """
        Save the fitted Zernike coefficients to a CSV file.
            :param save_path: Path to save the coefficients CSV file
            :type save_path: Path
            :param delimiter: Delimiter to use in the CSV file, default ","
            :type delimiter: str
        """
        save_path = save_path.with_suffix(".zrnk")
        if not self._hasZernikCoefficients():
            raise ValueError("Fit the Zernikes first.")
        
        with open(save_path, 'w') as f:
            f.write(f"Polynomial Name{delimiter}m{delimiter}n{delimiter}Coefficient{delimiter}Realworld Normalization Factor\n")
            for z, coef in zip(self.zernikes, self.coefficients):
                m, n = z.get_mn_orders()
                name = z.get_polynomial_name()
                f.write(f"{name}{delimiter}{m}{delimiter}{n}{delimiter}{coef}{delimiter}{self.realworldNormalizationFactor}\n")
        print(f"Zernike coefficients saved to {save_path}")
    
if __name__ == "__main__":
    print("This is the Zernikes module. It provides the PhaseFile class for loading phase data, fitting Zernike polynomials, and transforming coefficients for different device pupil sizes. It also includes utility functions for plotting and saving coefficients.")