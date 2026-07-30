"""
Code to compute finite-size charge corrections for charged defects in periodic
systems.

The charge-correction methods implemented are:

1. Extended FNV (eFNV) / Kumagai correction for isotropic and anistropic
   systems. This is the recommended default as it works regardless of system
   (an)isotropy, has a more efficient implementation and tends to be more
   robust in cases of small supercells.
   Includes:

       a) anisotropic PC energy (optionally multiple PCs).
       b) potential alignment by atomic site averaging outside Wigner Seitz
          radius.

2. Freysoldt (FNV) correction for isotropic systems. Only recommended if eFNV
   correction fails for some reason. Includes:

       a) point-charge (PC) energy.
       b) potential alignment by planar averaging.

If you use the corrections implemented in this module, please cite:

    Kumagai and Oba, Phys. Rev. B. 89, 195205 (2014) for the eFNV correction
    or
    Freysoldt, Neugebauer, and Van de Walle, Phys. Rev. Lett. 2009 for FNV.

**Note**: Ideally, the "defect site" used for all charge corrections should
actually be the centre of the localised charge in the defect supercell. Usually
this coincides with the defect site, but not always (e.g. vacancy where the
charge localises as a polaron on a neighbouring atom, complex defects etc.).
For sufficiently large supercells this is usually fine, as the defect and
centre-of-charge sites are close enough that any resulting quantitative error
in the correction is negligible. However, in cases where we have large
finite-size correction values, this can be significant. If some efficient
methods for determining the centroid of charge difference between bulk/defect
LOCPOTs (for FNV) or bulk/defect site potentials (for eFNV) were developed,
they should be used here.
"""

import contextlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from monty.json import MontyDecoder
from pymatgen.analysis.defects.corrections import freysoldt
from pymatgen.analysis.defects.utils import CorrectionResult
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.util.typing import PathLike

from doped.analysis import _convert_dielectric_to_tensor
from doped.utils import vise_handling
from doped.utils.parsing import (
    _get_bulk_supercell,
    _get_core_potentials_from_outcar_obj,
    _get_defect_supercell,
    _get_defect_supercell_frac_coords,
    get_core_potentials_from_outcar,
    get_locpot,
    get_site_mappings,
    get_wigner_seitz_radius,
)
from doped.utils.plotting import doped_plot_style, format_defect_name
from doped.utils.supercells import _get_min_image_distance_from_matrix


def _monty_decode_nested_dicts(d):
    """
    Decode any MSON dicts in ``d`` (in place), which may be nested in dicts or
    lists, tolerating per-key decoding failures.

    Used to ensure ``DefectEntry.calculation_metadata`` is decoded (after
    reloading from JSON).
    """
    for key, value in d.items():
        try:
            d[key] = MontyDecoder().process_decoded(value)
        except Exception as exc:
            print(f"Failed to decode {key} with error {exc!r}")


def _check_if_None_and_raise_error_if_so(var, var_name, display_name):
    if var is None:
        raise ValueError(
            f"{var_name} must be provided as an argument or be present in the `defect_entry` "
            f"`calculation_metadata` (as `{display_name}`), neither of which is the case here!"
        )


def _get_and_check_metadata(entry, key, display_name):
    value = entry.calculation_metadata.get(key, None)
    _check_if_None_and_raise_error_if_so(value, display_name, key)
    return value


def _check_if_pathlike_and_get_locpot_or_core_pots(
    locpot_or_outcar: Locpot | Outcar | PathLike | dict,
    obj_type: str = "locpot",
    dir_type: str = "",
    total_energy: list | float | None = None,
):
    if isinstance(locpot_or_outcar, PathLike):
        if obj_type == "locpot":
            return get_locpot(locpot_or_outcar)
        return get_core_potentials_from_outcar(  # otherwise OUTCAR
            locpot_or_outcar, dir_type=dir_type, total_energy=total_energy
        )

    if isinstance(locpot_or_outcar, Outcar):
        return _get_core_potentials_from_outcar_obj(
            locpot_or_outcar, dir_type=dir_type, total_energy=total_energy
        )

    if not isinstance(locpot_or_outcar, Locpot | Outcar | dict):
        raise TypeError(
            f"`{obj_type}` input must be either a path to a {obj_type.upper()} file or a pymatgen "
            f"{obj_type.upper()[0] + obj_type[1:]} object, but got {type(locpot_or_outcar)} instead."
        )

    return locpot_or_outcar


def get_freysoldt_correction(
    defect_entry,
    dielectric: float | np.ndarray | list | None = None,
    defect_locpot: PathLike | Locpot | dict | None = None,
    bulk_locpot: PathLike | Locpot | dict | None = None,
    plot: bool = False,
    filename: PathLike | None = None,
    axis: int | None = None,
    verbose: bool = True,
    style_file: PathLike | None = None,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the `isotropic` Freysoldt (FNV) correction for the
    input defect_entry.

    This function `does not` add the correction to ``defect_entry.corrections``
    (but the ``defect_entry.get_freysoldt_correction`` method does). If this
    correction is used, please cite Freysoldt's original paper;
    10.1103/PhysRevLett.102.016402.

    The defect coordinates are taken as the relaxed defect site by default
    (``DefectEntry.defect_supercell_site``) -- which is the bulk site for vacancies,
    but this can be overridden with the ``defect_frac_coords`` keyword argument.

    As a general rule of thumb, the charge correction terms should follow
    relatively consistent trends in terms of magnitudes. A large outlier
    (easily scanned with
    :meth:`~doped.thermodynamics.DefectThermodynamics.get_formation_energies`)
    often indicates something unusual/unexpected. See the FNV/eFNV and other
    finite-size charge correction papers for further details.

    Args:
        defect_entry:
            |DefectEntry| object for which to compute the FNV finite-size
            charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constants (ionic + static contributions), in the
            same xyz Cartesian basis as the supercell calculations (likely but
            not necessarily the same as the raw output of a VASP dielectric
            calculation, if an oddly-defined primitive cell is used). If
            ``None``, then the dielectric constant is taken from the
            ``defect_entry`` ``calculation_metadata`` if available.
            See the :ref:`Dielectric Constant <GGA_workflow_tutorial:7. Dielectric constant>`
            tutorial section for information on calculating and converging the
            dielectric constant.
        defect_locpot:
            Path to the output VASP LOCPOT file from the defect supercell
            calculation, or the corresponding ``pymatgen`` ``Locpot`` object,
            or a dictionary of the planar-averaged potential in the form:
            ``{i: Locpot.get_average_along_axis(i) for i in [0,1,2]}``.
            If ``None``, will try to use ``defect_locpot_dict`` from the
            ``defect_entry`` ``calculation_metadata`` if available.
        bulk_locpot:
            Path to the output ``VASP`` ``LOCPOT`` file from the bulk supercell
            calculation, or the corresponding ``pymatgen`` ``Locpot`` object,
            or a dictionary of the planar-averaged potential in the form:
            ``{i: Locpot.get_average_along_axis(i) for i in [0,1,2]}``.
            If ``None``, will try to use ``bulk_locpot_dict`` from the
            ``defect_entry`` ``calculation_metadata`` if available.
        plot (bool):
            Whether to plot the FNV electrostatic potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (PathLike):
            Filename to save the FNV electrostatic potential plots to.
            If None, plots are not saved.
        axis (int or None):
            If int, then the FNV electrostatic potential plot along the
            specified axis (0, 1, 2 for a, b, c) will be plotted. Note that the
            output charge correction is still that for `all` axes. If ``None``,
            then all three axes are plotted.
        verbose (bool):
            Whether to print the correction energy (default = True).
        style_file (PathLike):
            Path to a ``.mplstyle`` file to use for the plot. If ``None``
            (default), uses the default ``doped`` style
            (from ``doped/utils/doped.mplstyle``).
        **kwargs:
            Additional kwargs to pass to
            :func:`~pymatgen.analysis.defects.corrections.freysoldt.get_freysoldt_correction`
            (e.g. ``energy_cutoff``, ``mad_tol``, ``q_model``, ``step``,
            ``defect_frac_coords``).

    Returns:
        ``CorrectionResults`` (summary of the corrections applied and
        metadata), and the ``matplotlib`` ``Figure`` object (or axis object if
        axis specified) if ``plot`` is ``True``.
    """
    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    if hasattr(defect_entry, "calculation_metadata"):
        _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    if dielectric is None:
        dielectric = _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")
    dielectric = _convert_dielectric_to_tensor(dielectric)

    defect_locpot = defect_locpot or _get_and_check_metadata(
        defect_entry, "defect_locpot_dict", "Defect LOCPOT"
    )
    bulk_locpot = bulk_locpot or _get_and_check_metadata(defect_entry, "bulk_locpot_dict", "Bulk LOCPOT")

    defect_locpot = _check_if_pathlike_and_get_locpot_or_core_pots(defect_locpot, obj_type="locpot")
    bulk_locpot = _check_if_pathlike_and_get_locpot_or_core_pots(bulk_locpot, obj_type="locpot")

    fnv_correction = freysoldt.get_freysoldt_correction(
        q=defect_entry.charge_state,
        dielectric=dielectric,
        defect_locpot=defect_locpot,
        bulk_locpot=bulk_locpot,
        lattice=_get_defect_supercell(defect_entry).lattice if isinstance(defect_locpot, dict) else None,
        defect_frac_coords=kwargs.pop(
            "defect_frac_coords", _get_defect_supercell_frac_coords(defect_entry)
        ),  # _relaxed_ defect coords (except for vacancies)
        **kwargs,
    )

    if verbose:
        print(f"Calculated Freysoldt (FNV) correction is {fnv_correction.correction_energy:.3f} eV")

    if not plot and filename is None:
        return fnv_correction

    axis_label_dict = {0: r"$a$-axis", 1: r"$b$-axis", 2: r"$c$-axis"}
    if axis is None:
        with doped_plot_style(style_file):
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 3.5), dpi=600)
            for direction in range(3):
                plot_FNV(
                    fnv_correction.metadata["plot_data"][direction],
                    ax=axs[direction],
                    title=axis_label_dict[direction],
                    style_file=style_file,
                )
    else:
        plot_FNV(fnv_correction.metadata["plot_data"][axis], title=axis_label_dict[axis])
        fig = plt.gcf()

    if filename:
        plt.savefig(filename, bbox_inches="tight", transparent=True)

    return fnv_correction, fig


def plot_FNV(plot_data, title=None, ax=None, style_file=None):
    """
    Plots the planar-averaged electrostatic potential against the long range
    and short range models from the FNV correction method.

    Code templated from the original PyCDT and ``pymatgen.analysis.defects``
    implementations.

    Args:
         plot_data (dict):
            Dictionary of Freysoldt correction metadata to plot
            (i.e. ``defect_entry.corrections_metadata["plot_data"][axis]``
            where ``axis`` is one of [0, 1, 2] specifying which axis to plot
            along (a, b, c)).
         title (str):
            Title for the plot. Default is no title.
         ax (matplotlib.axes.Axes):
            Axes object to plot on. If ``None``, makes new ``Figure``.
         style_file (PathLike):
            Path to a mplstyle file to use for the plot. If ``None`` (default),
            uses the default ``doped`` style (from doped/utils/doped.mplstyle).
    """
    if not plot_data["pot_plot_data"]:
        raise ValueError(
            "Input `plot_data` has no `pot_plot_data` entry. Cannot plot FNV potential "
            "alignment before running correction!"
        )

    x = plot_data["pot_plot_data"]["x"]
    v_R = plot_data["pot_plot_data"]["Vr"]
    diff = plot_data["pot_plot_data"]["dft_diff"]
    short_range = plot_data["pot_plot_data"]["short_range"]
    check = plot_data["pot_plot_data"]["check"]
    C = plot_data["pot_plot_data"]["shift"]

    with doped_plot_style(style_file):
        if ax is None:
            plt.close("all")  # close any previous figures
            _fig, ax = plt.subplots()
        (line1,) = ax.plot(x, v_R, c="black", zorder=1, label="FNV long-range model ($V_{lr}$)")
        (line2,) = ax.plot(x, diff, c="red", label=r"$\Delta$(Locpot)")
        (line3,) = ax.plot(
            x,
            short_range,
            c="green",
            label=r"$V_{sr}$ = $\Delta$(Locpot) - $V_{lr}$" + "\n(pre-alignment)",
        )
        leg1 = ax.legend(handles=[line1, line2, line3], loc=9)  # middle top legend
        ax.add_artist(leg1)  # so isn't overwritten with later legend call

        line4 = ax.axhline(C, color="k", linestyle="--", label=f"Alignment constant C = {C:.3f} V")
        tmpx = [x[i] for i in range(check[0], check[1])]
        poly_coll = ax.fill_between(tmpx, -100, 100, facecolor="red", alpha=0.15, label="Sampling region")
        ax.legend(handles=[line4, poly_coll], loc=8)  # bottom middle legend

        ax.set_xlim(round(x[0]), round(x[-1]))
        ymin = min(*v_R, *diff, *short_range)
        ymax = max(*v_R, *diff, *short_range)
        ax.set_ylim(-0.2 + ymin, 0.2 + ymax)
        ax.set_xlabel(r"Distance along axis ($\AA$)")
        ax.set_ylabel("Potential (V)")
        ax.axhline(y=0, linewidth=0.2, color="black")
        if title is not None:
            ax.set_title(str(title))
        ax.set_xlim(0, max(x))

        return ax


def get_kumagai_correction(
    defect_entry,
    dielectric: float | np.ndarray | list | None = None,
    defect_region_radius: float | None = None,
    excluded_indices: list[int] | None = None,
    point_charges: list[tuple[float, np.ndarray | list]] | None = None,
    defect_outcar: PathLike | Outcar | None = None,
    bulk_outcar: PathLike | Outcar | None = None,
    plot: bool = False,
    filename: PathLike | None = None,
    verbose: bool = True,
    style_file: PathLike | None = None,
    **kwargs,
) -> CorrectionResult:
    """
    Function to compute the Kumagai (eFNV) finite-size charge correction for
    the input defect_entry. Compatible with both isotropic/cubic and
    anisotropic systems.

    This function `does not` add the correction to ``defect_entry.corrections``
    (but the defect_entry.get_kumagai_correction method does). If this
    correction is used, please cite the Kumagai & Oba (eFNV) paper:
    10.1103/PhysRevB.89.195205 and the ``pydefect`` paper: "Insights into
    oxygen vacancies from high-throughput first-principles calculations" Yu
    Kumagai, Naoki Tsunoda, Akira Takahashi, and Fumiyasu Oba Phys. Rev.
    Materials 5, 123803 (2021) -- 10.1103/PhysRevMaterials.5.123803

    Typically for reasonably well-converged supercell sizes, the default
    ``defect_region_radius`` works perfectly well. However, for certain
    materials at small/intermediate supercell sizes, you may want to adjust
    this (and/or ``excluded_indices``) to ensure the best sampling of the
    plateau region away from the defect position -- ``doped`` should throw a
    warning in these cases (about the correction error being above the default
    tolerance (50 meV)). For example, with layered materials, the defect charge
    is often localised to one layer, so we may want to adjust
    ``defect_region_radius`` and/or ``excluded_indices`` to ensure that only
    sites in other layers are used for the sampling region (plateau) -- see
    example on doped docs Tips page.

    The defect coordinates are taken as the relaxed defect site by default
    (``DefectEntry.defect_supercell_site``) -- which is the bulk site for
    vacancies, but this can be overridden with the ``defect_coords`` keyword
    argument.

    As a general rule of thumb, the charge correction terms should follow
    relatively consistent trends in terms of magnitudes. A large outlier
    (easily scanned with
    :meth:`~doped.thermodynamics.DefectThermodynamics.get_formation_energies`)
    often indicates something unusual/unexpected. See the FNV/eFNV and other
    finite-size charge correction papers for further details.

    Args:
        defect_entry (|DefectEntry|):
            |DefectEntry| object for which to compute the Kumagai finite-size
            charge correction.
        dielectric (float or int or 3x1 matrix or 3x3 matrix):
            Total dielectric constants (ionic + static contributions), in the
            same xyz Cartesian basis as the supercell calculations (likely but
            not necessarily the same as the raw output of a VASP dielectric
            calculation, if an oddly-defined primitive cell is used). If
            ``None``, then the dielectric constant is taken from the
            ``defect_entry`` ``calculation_metadata`` if available.
            See the :ref:`Dielectric Constant <GGA_workflow_tutorial:7. Dielectric constant>`
            tutorial section for information on calculating and converging the
            dielectric constant.
        defect_region_radius (float):
            Radius of the defect region (in Å). Sites outside the defect
            region are used for sampling the electrostatic potential far
            from the defect (to obtain the potential alignment).
            If ``None`` (default), uses the Wigner-Seitz radius of the
            supercell.
        excluded_indices (list):
            List of site indices (in the defect supercell) to exclude from
            the site potential sampling in the correction calculation/plot.
            If ``None`` (default), no sites are excluded.
        point_charges (list[tuple[float, ArrayLike]]):
            List of ``(charge, frac_coords)`` pairs specifying a multicentre
            point-charge model (e.g. for constituent point defects of a
            defect complex), used in place of the default single point charge.
            The charges should sum to the defect charge state. TODO: Currently
            the sites sampled from alignment are based on the distance from the
            nearest point charge, not the overall defect site.
            Default is ``None`` (single point charge).
        defect_outcar (PathLike or |Outcar|):
            Path to the output ``VASP`` ``OUTCAR`` file from the defect
            supercell calculation, or the corresponding ``pymatgen`` |Outcar|
            object. If ``None``, will try to use the ``defect_site_potentials``
            from the ``defect_entry`` ``calculation_metadata`` if available.
        bulk_outcar (PathLike or |Outcar|):
            Path to the output ``VASP`` ``OUTCAR`` file from the bulk supercell
            calculation, or the corresponding ``pymatgen`` |Outcar| object.
            If ``None``, will try to use the ``bulk_site_potentials``
            from the ``defect_entry`` ``calculation_metadata`` if available.
        plot (bool):
            Whether to plot the Kumagai site potential plots (for
            manually checking the behaviour of the charge correction here).
        filename (PathLike):
            Filename to save the Kumagai site potential plots to.
            If None, plots are not saved.
        verbose (bool):
            Whether to print the correction energy (default = True).
        style_file (PathLike):
            Path to a mplstyle file to use for the plot. If ``None`` (default),
            uses the default ``doped`` style (from doped/utils/doped.mplstyle).
        **kwargs:
            Additional kwargs to pass to
            ``pydefect.corrections.efnv_correction.ExtendedFnvCorrection``
            (e.g. ``charge``, ``defect_region_radius``, ``defect_coords``).

    Returns:
        ``CorrectionResults`` (summary of the corrections applied and
        metadata), and the ``matplotlib`` ``Figure`` object if ``plot`` is
        ``True``.
    """
    with vise_handling():  # avoid vise issues (warning suppression, logging, Windows bug)
        from pydefect.analyzer.calc_results import CalcResults
        from pydefect.corrections.efnv_correction import ExtendedFnvCorrection, PotentialSite
        from pydefect.corrections.ewald import Ewald
        from pydefect.corrections.site_potential_plotter import SitePotentialMplPlotter
        from pydefect.defaults import defaults
        from pydefect.util.error_classes import SupercellError

    # TODO switch for sampling away from all charges vs sampling away from defect_coords
    # (probably the centroid)
    sample_away_from_all_charges = True

    def doped_make_efnv_correction(
        charge: float,
        calc_results: CalcResults,
        perfect_calc_results: CalcResults,
        dielectric_tensor: np.ndarray,
        defect_coords: np.ndarray | list,
        defect_region_radius: float | None = None,
        accuracy: float = defaults.ewald_accuracy,
        unit_conversion: float = 180.95128169876497,
        excluded_indices: list | None = None,
        point_charges: list[tuple[float, np.ndarray | list]] | None = None,
    ):
        r"""
        This is a modified version of ``pydefect``\'s ``make_efnv_correction``
        function (in ``pydefect.cli.vasp.make_efnv_correction``) to allow the
        defect region radius to be adjusted (e.g. in cases of layered
        materials, where often the defect charge is localised to one layer, so
        we likely want to adjust the defect region radius to ensure that only
        `other` layers are used for the sampling (plateau) region), and to
        allow the model charge to be split over multiple point charges (e.g.
        over the constituent point defects of a defect complex).

        If defect_region_radius is not specified, then the ``pydefect`` default
        (which is the Wigner-Seitz region of the supercell) is used.
        """
        if calc_results.structure.lattice != perfect_calc_results.structure.lattice:
            raise SupercellError("The lattice constants for defect and perfect models are different")
        lattice = calc_results.structure.lattice

        if point_charges is not None:
            pc_charges, pc_frac_coords = zip(*point_charges, strict=True)
            if not abs(sum(pc_charges) - charge) < 1e-2:
                raise ValueError(
                    f"The sum of the supplied point charges ({np.sum(pc_charges):+.3f}) does not "
                    f"match the total defect charge state ({charge:+.3f})."
                )
        else:
            pc_charges = [charge]
            pc_frac_coords = [defect_coords]
        pc_charges = np.array(pc_charges, dtype=float)
        pc_frac_coords = np.array(pc_frac_coords, dtype=float)

        sites, rel_coords = [], []

        excluded_indices = [] if excluded_indices is None else [int(i) for i in excluded_indices]

        mapping = get_site_mappings(
            calc_results.structure, perfect_calc_results.structure, threshold=np.inf
        )

        for _d_to_p_dist, d, p in mapping:
            if d not in excluded_indices and not any(i is None for i in [d, p]):
                specie = str(calc_results.structure[d].specie)
                frac_coords = calc_results.structure[d].frac_coords

                # TODO distance to the nearest point charge right now
                distance = (
                    min(
                        lattice.get_distance_and_image(pc_coords, frac_coords)[0]
                        for pc_coords in pc_frac_coords
                    )
                    if sample_away_from_all_charges
                    else lattice.get_distance_and_image(defect_coords, frac_coords)[0]
                )
                pot = calc_results.potentials[d] - perfect_calc_results.potentials[p]

                sites.append(PotentialSite(specie, distance, pot, None))

                # rel coords to all centres to reconstruct model potential
                rel_coords.append([(frac_coords - pc_coords).tolist() for pc_coords in pc_frac_coords])

        ewald = Ewald(lattice.matrix, dielectric_tensor, accuracy=accuracy)

        # Monkey-patch pydefect Ewald functions to vectorise the calculations (otherwise can be slow):
        from scipy.special import erfc

        def ewald_real(self, include_self: bool, shift: list[float]) -> float:
            R = np.asarray(list(self.r_lattice_set(include_self, shift)))  # (N, 3)
            rMr = np.einsum("ij,jk,ik->i", R, self.epsilon_inv, R)
            root = np.sqrt(rMr)
            return np.sum(erfc(self.mod_ewald_param * root) / root) / (4 * np.pi * self.root_epsilon)

        def ewald_rec(self, coord) -> float:
            cart_coord = coord @ self.lattice
            G = np.asarray(list(self.g_lattice_set()))  # (N, 3)
            gEg = np.einsum("ij,jk,ik->i", G, self.dielectric_tensor, G)
            phase = np.cos(G @ cart_coord)
            return np.sum(np.exp(-gEg / (4 * self.mod_ewald_param**2)) / gEg * phase) / self.volume

        Ewald.ewald_real = ewald_real
        Ewald.ewald_rec = ewald_rec

        # for multi centre: point charge correction is linear so
        # E_corr = 1/2 * sum over i,j [q_i*q_j*(v_periodic(r_ij)-v_isolated(r_ij))]
        # plus alignment?
        point_charge_correction = 0.0
        d_min = 0.5 * _get_min_image_distance_from_matrix(lattice.matrix)
        if np.any(pc_charges):
            # self image term correction i=j
            point_charge_correction = -ewald.lattice_energy * np.sum(pc_charges**2)

            # cross image term correction i<j
            for i, (q_i, x_i) in enumerate(zip(pc_charges, pc_frac_coords, strict=True)):
                for j, (q_j, x_j) in enumerate(zip(pc_charges, pc_frac_coords, strict=True)):
                    if j <= i or not (q_i and q_j):
                        continue
                    d_ij, image = lattice.get_distance_and_image(x_i, x_j)
                    x_ij = x_j + image - x_i

                    # catch coincident charges
                    if d_ij < 0.1:
                        raise ValueError(
                            f"Point charges {i} and {j} are (near-)coincident (separation "
                            f"{d_ij:.2f} Å), giving a divergent point-charge model -- these "
                            f"should be combined into a single point charge!"
                        )

                    # validity of multicentre
                    if d_ij > d_min:
                        warnings.warn(
                            f"Point charge separation ({d_ij:.2f} Å) exceeds half the minimum "
                            f"image distance of the supercell ({d_min:.2f} Å), so the multicentre "
                            f"point-charge model may be unreliable."
                        )

                    v_periodic = ewald.atomic_site_potential(x_ij.tolist())
                    r_ij = lattice.get_cartesian_coords(x_ij)
                    v_isolated = 1 / (
                        4 * np.pi * ewald.root_epsilon * np.sqrt(r_ij @ ewald.epsilon_inv @ r_ij)
                    )
                    point_charge_correction += -q_i * q_j * (v_periodic - v_isolated)

        # ? this is not WS radius in efnv paper (min_image_distance/2)?
        if defect_region_radius is None:
            defect_region_radius = get_wigner_seitz_radius(lattice)

        # model potential is just sum of point charge potentials
        for site, site_rel_coords in zip(sites, rel_coords, strict=False):
            if site.distance > defect_region_radius:
                if not np.any(pc_charges):
                    site.pc_potential = 0
                else:
                    site.pc_potential = (
                        sum(
                            pc_charge * ewald.atomic_site_potential(rel_coord)
                            for pc_charge, rel_coord in zip(pc_charges, site_rel_coords, strict=False)
                            if pc_charge
                        )
                        * unit_conversion
                    )

        return ExtendedFnvCorrection(
            charge=charge,
            point_charge_correction=point_charge_correction * unit_conversion,
            defect_region_radius=defect_region_radius,
            sites=sites,
            defect_coords=tuple(defect_coords),
        )

    # ensure calculation_metadata are decoded in case defect_dict was reloaded from json
    if hasattr(defect_entry, "calculation_metadata"):
        _monty_decode_nested_dicts(defect_entry.calculation_metadata)

    if dielectric is None:
        dielectric = _get_and_check_metadata(defect_entry, "dielectric", "Dielectric constant")
    dielectric = _convert_dielectric_to_tensor(dielectric)

    core_potentials_dict = {}
    for key, outcar in zip(["defect", "bulk"], [defect_outcar, bulk_outcar], strict=False):
        total_energy = []
        with contextlib.suppress(Exception):
            total_energy.append(
                defect_entry.sc_entry.energy if key == "defect" else defect_entry.bulk_entry.energy
            )
            total_energy.append(
                defect_entry.calculation_metadata["run_metadata"][f"{key}_vasprun_dict"]["output"][
                    "ionic_steps"
                ][-1]["electronic_steps"][-1]["e_0_energy"]
            )

        if outcar is not None:
            core_potentials_dict[key] = _check_if_pathlike_and_get_locpot_or_core_pots(
                outcar, obj_type="outcar", dir_type=key, total_energy=total_energy
            )
        else:
            core_potentials_dict[key] = _get_and_check_metadata(
                defect_entry,
                f"{key}_site_potentials",
                f"{key.capitalize()} OUTCAR (for atomic site potentials)",
            )

    defect_supercell = _get_defect_supercell(defect_entry).copy()
    defect_supercell.remove_oxidation_states()  # pydefect needs structure without oxidation states
    defect_calc_results_for_eFNV = CalcResults(
        structure=defect_supercell,
        energy=np.inf,
        magnetization=np.inf,
        potentials=core_potentials_dict["defect"],
    )

    bulk_supercell = _get_bulk_supercell(defect_entry).copy()
    bulk_supercell.remove_oxidation_states()  # pydefect needs structure without oxidation states
    if bulk_supercell.lattice != defect_supercell.lattice:  # pydefect will crash
        # check if the difference is tolerable (< 0.01 Å)
        if not np.allclose(bulk_supercell.lattice.matrix, defect_supercell.lattice.matrix, atol=1e-2):
            warnings.warn(
                f"Bulk and defect supercells have different lattices, and so the eFNV (Kumagai) "
                f"correction may be unreliable!"
                f"\nBulk lattice:\n{bulk_supercell.lattice}\nDefect lattice:\n{defect_supercell.lattice}"
            )
        # scale bulk lattice to match defect lattice, so pydefect doesn't crash:
        bulk_supercell.lattice = defect_supercell.lattice

    bulk_calc_results_for_eFNV = CalcResults(
        structure=bulk_supercell,
        energy=np.inf,
        magnetization=np.inf,
        potentials=core_potentials_dict["bulk"],
    )

    efnv_correction = doped_make_efnv_correction(
        charge=defect_entry.charge_state,
        calc_results=defect_calc_results_for_eFNV,
        perfect_calc_results=bulk_calc_results_for_eFNV,
        dielectric_tensor=dielectric,
        defect_coords=kwargs.pop(
            "defect_coords", _get_defect_supercell_frac_coords(defect_entry)
        ),  # _relaxed_ defect coords (except for vacancies)
        defect_region_radius=defect_region_radius,
        excluded_indices=excluded_indices,
        point_charges=point_charges,
        **kwargs,
    )
    kumagai_correction_result = CorrectionResult(
        correction_energy=efnv_correction.correction_energy,
        metadata={"pydefect_ExtendedFnvCorrection": efnv_correction},
    )

    if verbose:
        print(
            f"Calculated Kumagai (eFNV) correction is {kumagai_correction_result.correction_energy:.3f} eV"
        )

    if not plot and filename is None:
        return kumagai_correction_result

    spp = SitePotentialMplPlotter.from_efnv_corr(
        title=f"{format_defect_name(defect_entry.name, False)} -- eFNV Site Potentials",
        efnv_correction=efnv_correction,
    )
    with doped_plot_style(style_file):
        plt.close("all")  # close any previous figures
        spp.construct_plot()
        fig = spp.plt.gcf()
        ax = fig.gca()

        # reformat plot slightly:
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        # shade in sampling region:
        spp.plt.fill_between(
            [spp.defect_region_radius, x_lims[1]],
            *y_lims,
            color="purple",
            alpha=0.15,
            label="Sampling Region",
        )
        spp.plt.ylim(*y_lims)  # reset y-lims after fill_between

        # update legend:
        handles, labels = ax.get_legend_handles_labels()
        labels = [
            label.replace("point charge", "Point Charge (PC)").replace(
                "potential difference", r"$\Delta V$"
            )
            for label in labels
        ]
        labels = [
            label + r" ($V_{defect} - V_{bulk}$)" if Element.is_valid_symbol(label) else label
            for label in labels
        ]

        # add entry for dashed red line:
        handles += [Line2D([0], [0], **spp._mpl_defaults.hline)]
        labels += [
            rf"Avg. $\Delta V$ = {spp.ave_pot_diff:.3f} V",
        ]
        ax.legend_.remove()
        ax.legend(handles, labels, loc="best", borderaxespad=0, fontsize=8)
        x_label = (
            "Distance from nearest PC"
            if point_charges and sample_away_from_all_charges
            else "Distance from defect"
        )
        ax.set_xlabel(f"{x_label} ({spp._x_unit})", size=spp._mpl_defaults.label_font_size)

    if filename:
        spp.plt.savefig(filename, bbox_inches="tight", transparent=True)

    return kumagai_correction_result, fig
