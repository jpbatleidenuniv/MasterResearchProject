from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from swiftsimio import cosmo_array
from swiftsimio.visualisation.projection import project_pixel_grid


def plot_box_masses(
    data_field,
    ax: Axes,
    vminmax: tuple[float, float],
    region: cosmo_array | None = None,
    resolution: int = 512,
    title: str = "Stellar mass",
) -> AxesImage:
    """Draw the projected surface density of a particle group (e.g. data.stars) on ax."""
    if region is not None:
        region = region.ravel()
        assert region.shape == (6,), f"region should have 6 entries, got {region.shape}"
        extent = tuple(region.to_value("Mpc"))  # (xmin, xmax, ymin, ymax)
    else:
        extent = None

    img = (
        project_pixel_grid(
            data=data_field,
            resolution=resolution,
            project="masses",
            region=region,
            parallel=True,
            periodic=True,
        )
        .to_physical("Msun/kpc**2")
        .T
    )

    cmap = pyplot.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))

    im = ax.imshow(
        img.value,
        norm=LogNorm(*vminmax),
        cmap=cmap,
        origin="lower",
        extent=extent,
    )

    ax.set_aspect("equal")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$X$ [cMpc]", fontsize=14)
    ax.set_ylabel(r"$Y$ [cMpc]", fontsize=14)
    ax.set_title(title, fontsize=12)
    return im
