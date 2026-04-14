import math
import numpy as np
from zernpy import generate_polynomials, ZernPol, fit_polynomials
from zernpy.calculations.fit_zernike_pols import crop_phases_img
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from slm_lib.Hologram import SLM_WIDTH_PX, SLM_HEIGHT_PX, SLM_PIXEL_LENGTH, create_checkerboard

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
            zernikes.append(ZernPol(m, n))
    return coefficients, zernikes, realworldNormalizationFactor

def compare_zernikes(measured_zernikes: list[ZernPol], reference_zernikes: list[ZernPol])->tuple[list[int], list[int], list[tuple[int, int]]]:
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
            measured.append(measured_zernike_indices.index(ref_zernike_indices.index(m)))
            index.append(m)
    return ref, measured, index

def subtract_coefficients(measured_coeffs: list[float], reference_coeffs: list[float])->tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Subtract reference Zernike coefficients from measured coefficients to isolate the aberrations of interest.
        :param measured_coeffs: List of Zernike coefficients obtained from the measurement.
        :type measured_coeffs: list[float]
        :param reference_coeffs: List of Zernike coefficients representing the reference or baseline aberrations to subtract.
        :type reference_coeffs: list[float]
        :return: List of Zernike coefficients after subtraction, and the corresponding (m, n) orders of the Zernike polynomials that were compared.
        :rtype: tuple[np.ndarray, list[tuple[int, int]]]
    """
    ref_ind, meas_ind, zern_indices = compare_zernikes(measured_coeffs, reference_coeffs)
    measured_coeffs = np.asarray(measured_coeffs)[np.asarray(meas_ind)]
    reference_coeffs = np.asarray(reference_coeffs)[np.asarray(ref_ind)]
    return measured_coeffs - reference_coeffs, zern_indices

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

    def __init__(self, path: Path, delimiter=",", crop_radius=0.5, pixelpitch=DEFAULT_PIXELPITCH):
        self.zernikes: list[ZernPol] | None = None
        self.coefficients: list[float] | None = None
        self.pixelpitch = None
        self.fileSize = None
        self.minSize = None
        self.realworldNormalizationFactor = None
        self.path = path
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
        if self.crop_radius is None:
            raise ValueError("crop_radius must be set before building the mesh.")

        h, w = self.fileSize
        cy = np.ceil(h / 2)
        cx = np.ceil(w / 2)
        R0 = np.ceil(self.minSize / 2) * self.crop_radius

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
            left = header_line.find("Polynomial Name")+len("Polynomial Name")
            right = header_line.find("m")
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
                zernikes.append(ZernPol(m, n))
        self.coefficients = coefficients
        self.zernikes = zernikes
        self.realworldNormalizationFactor = realworldNormalizationFactor

    def _calcNormalizationFactor(self):
        if self.pixelpitch is None or self.crop_radius is None:
            return None
        return self.minSize / 2 * self.pixelpitch * self.crop_radius
    ##############################################################################################
    # Static methods

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
        """
        if len(coefficients) != len(zernikes):
            raise ValueError("Number of coefficients must match number of Zernikes.")
        
        total_phase = np.zeros_like(r_mesh)
        for a, z in zip(coefficients, zernikes):
            m, n = z.get_mn_orders()
            total_phase += a * zernike_extended(m, n, r_mesh, theta_mesh)
        return total_phase
    
    #   plotting
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
    def plot_coefficients_on_axes(ax:Axes, coefficients: list[float], zernikes: list[ZernPol], exclude_indeces: list[int] = []):
        """ 
        Utility function to plot Zernike coefficients as a bar chart on given axes, with labels and optional exclusion of certain indices.
            :param ax: Matplotlib axes to plot on
            :type ax: Axes
            :param coefficients: List of Zernike coefficients to plot
            :type coefficients: list[float]
            :param zernikes: List of Zernike polynomials corresponding to the coefficients, used for labeling
            :type zernikes: list[ZernPol]
            :param exclude_indeces: List of indices of Zernike orders to exclude from the plot (e.g. piston, tip, tilt), default []
            :type exclude_indeces: list[int]
        """
        name_labels = [f"{z.get_polynomial_name()} ({z.get_mn_orders()[0]}, {z.get_mn_orders()[1]})" 
                       for i, z in enumerate(zernikes) if i not in exclude_indeces]
        coefficients = [coef for i, coef in enumerate(coefficients) if i not in exclude_indeces]
        ax.bar(name_labels, coefficients)
        ax.set_title("Zernike Coefficients")
        ax.set_ylabel("Coefficient Value")
        plt.xticks(rotation=90)
        plt.tight_layout()


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

    def device_physical_radius(self, devicePixelpitch, deviceRadiusPx, deviceCropRadius):
        return self.calcNormalizationFactorStatic(
            devicePixelpitch,
            deviceRadiusPx,
            deviceCropRadius,
        )

    def get_scale_factor_to_device(self, devicePixelpitch, deviceRadiusPx, deviceCropRadius):
        """
        Scale factor s = R_dev / R_fit
        """
        R_fit = self.source_physical_radius()
        print(f"Source physical radius: {R_fit} m")
        R_dev = self.device_physical_radius(devicePixelpitch, deviceRadiusPx, deviceCropRadius)
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
        if self.coefficients is None or self.zernikes is None:
            raise ValueError("Fit the Zernikes first.")

        transformed_coeffs = []
        for a, z in zip(self.coefficients, self.zernikes):
            m, n = z.get_mn_orders()
            # The scaling of Zernike polynomials with radius is R_n^m(r) ~ r^n
            transformed_a = a * (scaleFactor ** n)
            transformed_coeffs.append(transformed_a)   
        return transformed_coeffs
    
    def show_phase(self, title = "Phase", cmap="viridis", wrapped = True):
        fig, ax = plt.subplots(figsize=(6,6))
        self.plot_phase_on_axes(ax, self.phase, title, cmap, wrapped)
        plt.show()
        return fig
    
    def plot_zernike_modes(self, wrapped = False, cmap="coolwarm"):
        if self.zernikes is None:
            raise ValueError("No Zernikes to plot. Fit the Zernikes first.")
        
        fig, axs = plt.subplots(1, len(self.zernikes), figsize=(4*len(self.zernikes), 4))
        for z, ax in zip(self.zernikes, axs):
            m, n = z.get_mn_orders()
            mode_phase = zernike_extended(m, n, *self._createNormPolarMesh())*self.coefficients[self.zernikes.index(z)]
            self.plot_phase_on_axes(ax, mode_phase, title=f"{z.get_polynomial_name()}\nZernike m={m}, n={n}", cmap=cmap, wrapped=False)
        fig.tight_layout()
        fig.show()
        return fig
    
    def plot_coefficients(self, exclude_indeces = []):
        if self.coefficients is None or self.zernikes is None:
            raise ValueError("No coefficients to plot. Fit the Zernikes first.")
        
        fig, ax = plt.subplots(figsize=(8,4))
        self.plot_coefficients_on_axes(ax, self.coefficients, self.zernikes, exclude_indeces)
        fig.show()
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
        if self.coefficients is None or self.zernikes is None:
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
        if self.coefficients is None or self.zernikes is None:
            raise ValueError("Fit the Zernikes first.")
        
        with open(save_path, 'w') as f:
            f.write(f"Polynomial Name{delimiter}m{delimiter}n{delimiter}Coefficient{delimiter}Realworld Normalization Factor\n")
            for z, coef in zip(self.zernikes, self.coefficients):
                m, n = z.get_mn_orders()
                name = z.get_polynomial_name()
                f.write(f"{name}{delimiter}{m}{delimiter}{n}{delimiter}{coef}{delimiter}{self.realworldNormalizationFactor}\n")
        print(f"Zernike coefficients saved to {save_path}")
    
if __name__ == "__main__":
    crop_radius = 0.55

    dir = Path(r"C:\Users\smile\Downloads\260227_2\260227_2\WF_k6Def_0SphAb\Results")
    phaseF = PhaseFile.loadFile(dir / "PHA SID4 03h41m24s102ms.csv", delimiter=",", crop_radius=crop_radius, pixelpitch=29.6e-6)

    # choose modes however you like
    fitPols = generate_polynomials(3)
    fit = phaseF.fitZernikesToFile(fitPols)

    # target device example
    device_shape = (1200, 1024)  # e.g. SLM resolution
    device_pixelpitch = 12.5e-6  # e.g. SLM pixel pitch in meters
    device_radius_px = min(device_shape) / 2
    device_crop_radius = 0.9

    device_shape = phaseF.fileSize  # e.g. SLM resolution
    device_pixelpitch = DEFAULT_PIXELPITCH  # e.g. SLM pixel pitch in meters
    device_radius_px = min(device_shape) / 2
    device_crop_radius = 0.9
    print(f"device_radius = {device_radius_px} px, device_crop_radius = {device_crop_radius}, device physical radius = {phaseF.device_physical_radius(device_pixelpitch, device_radius_px, device_crop_radius)} m")

    transformed_coeffs = phaseF.transform_pupil_size(
        phaseF.get_scale_factor_to_device(device_pixelpitch, device_radius_px, device_crop_radius)
    )
    print(f"realworld normalization factor for source: {phaseF.realworldNormalizationFactor} m")
    print(f"Scale factor to device: {phaseF.get_scale_factor_to_device(device_pixelpitch, device_radius_px, device_crop_radius)}")
    phaseTransformed = phaseF.evaluate_zernikes_on_mesh(transformed_coeffs, fitPols, *phaseF.make_polar_mesh(device_shape, device_crop_radius)) 

    fig = plt.figure(figsize=(12, 6))
    ZernPol.plot_sum_zernikes_on_fig(fig,fit[0], fitPols)
    ax = fig.add_subplot(4,4,1)
    c = ax.imshow(phaseF.phase)
    plt.colorbar(c, ax=ax)
    ax.set_title("Original Phase")
    ax2 = fig.add_subplot(4,4,5)
    c2 = ax2.imshow(fit[1])
    plt.colorbar(c2, ax=ax2)
    ax2.set_title("Evaluated Phase")
    ax3 = fig.add_subplot(4,4,9)
    c3 = ax3.imshow(phaseF.make_polar_mesh(device_shape, device_crop_radius)[1], cmap="coolwarm")
    plt.colorbar(c3, ax=ax3)
    ax3.set_title("Polar Mesh phase")
    ax4 = fig.add_subplot(4,4,13)
    c4 = ax4.imshow(phaseTransformed%(2*np.pi), cmap="coolwarm")
    ax4.set_title("Phasefront on device")
    plt.colorbar(c4, ax=ax4)
    plt.show()
    