import math
import numpy as np
from zernpy import generate_polynomials, ZernPol, fit_polynomials
from zernpy.calculations.fit_zernike_pols import crop_phases_img
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

DEFAULT_PIXELPITCH = 29.6e-6  # default pixel pitch for the source WFS, used for real-world normalization if not set otherwise

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
        self.path = path
        self.phase = np.loadtxt(path, delimiter=delimiter)
        self.fileSize = self.phase.shape
        self.minSize = min(self.fileSize)
        self.crop_radius = crop_radius
        self.zernikes: list[ZernPol] | None = None
        self.coefficients: list[float] | None = None
        self.pixelpitch = None
        self.set_pixelpitch(pixelpitch)
        self.realworldNormalizationFactor = self._calcNormalizationFactor()

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

    @staticmethod
    def calcNormalizationFactorStatic(pixelpitch, deviceRadiusPx, cropRadiusNormalized):
        return deviceRadiusPx * cropRadiusNormalized * pixelpitch

    def _calcNormalizationFactor(self):
        if self.pixelpitch is None or self.crop_radius is None:
            return None
        return self.minSize / 2 * self.pixelpitch * self.crop_radius

    @staticmethod
    def loadFile(path, delimiter=",", crop_radius=0.5, pixelpitch=DEFAULT_PIXELPITCH):
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

    def set_pixelpitch(self, pixelpitch):
        self.pixelpitch = pixelpitch
        self.realworldNormalizationFactor = self._calcNormalizationFactor()

    def fitZernikesToFile(self, zernikes: list[ZernPol]):
        self._setZernikes(zernikes)
        fit = fit_polynomials(
            self.phase,
            self.zernikes,
            crop_radius=self.crop_radius,
            return_cropped_image=True,
        )
        self.coefficients = fit[0]
        return fit

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
        

    @staticmethod
    def make_polar_mesh(shape, crop_radius=1.0, phase_offset = np.pi*3/2, crop_pupil=True):
        """
        Polar mesh on the target device, normalized to the target device pupil.
        r_dev_norm = 1 at the target pupil edge.
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
        R_dev = self.device_physical_radius(devicePixelpitch, deviceRadiusPx, deviceCropRadius)
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
    
    @staticmethod
    def evaluate_zernikes_on_mesh(coefficients: list[float], zernikes: list[ZernPol], r_mesh, theta_mesh):
        """
        Evaluate the Zernike sum on a given polar mesh.
        """
        if len(coefficients) != len(zernikes):
            raise ValueError("Number of coefficients must match number of Zernikes.")
        
        total_phase = np.zeros_like(r_mesh)
        for a, z in zip(coefficients, zernikes):
            m, n = z.get_mn_orders()
            total_phase += a * zernike_extended(m, n, r_mesh, theta_mesh)
        return total_phase
    
    @staticmethod
    def plot_phase_on_axes(ax:Axes, phase, title = "Phase", cmap="viridis", wrapped = True):
        if wrapped:
            phase_to_plot = np.mod(phase, 2*np.pi)
        else:
            phase_to_plot = phase
        c = ax.imshow(phase_to_plot, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(c, ax=ax)

    def show_phase(self, title = "Phase", cmap="viridis", wrapped = True):
        fig, ax = plt.subplots(figsize=(6,6))
        self.plot_phase_on_axes(ax, self.phase, title, cmap, wrapped)
        plt.show()
        return fig
    

    
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
    