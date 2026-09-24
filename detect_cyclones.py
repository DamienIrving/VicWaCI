"""Detect cyclones at 500 hPa in ERA5 and save the output to netcdf.

Save monthly files of cyclone masks for the period 1980-01 to 2022-12

Notes
-----
- Based on ACCESS-NRI's NCI xp65 anaylsis3-25.11 conda environment
- Uses  M. Barnes local installation of dynlib
    - https://folk.uib.no/csp001/dynlib_doc/
    - https://git.app.uib.no/Clemens.Spensberger/dynlib/-/blob/master/
- Data:
    - ERA5 total precipitation (tp) and geopotential (z) (project rt52 on NCI)

Example
-------
```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-25.11
MIN=30
python /g/data/xv83/as3189/vicwaci/detect_cyclones.py --min ${MIN}
```
"""

import os
import sys
import importlib.util

pkg_dir = "/g/data/xv83/mb0427/dynlib-1.6.1.dev7+g2e093da-py3.11.egg"
wxsys_dir = "/g/data/gb02/mb0427/WxSysLib"

os.environ["WXSYSLIBDIR"] = wxsys_dir
os.environ["DYNLIBDIR"] = pkg_dir

sys.path.insert(0, pkg_dir)
sys.path.insert(0, wxsys_dir)

importlib.invalidate_caches()

import dynlib
from dynlib import detect, gridlib, dynfor

import argparse
from datetime import datetime, timedelta
from dask.distributed import Client
from functools import wraps
import gc
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time
from tqdm import tqdm
import xarray as xr


g = 9.80665  # m s-2


def time_it(func):
    """Decorator to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = timedelta(seconds=time.perf_counter() - start)
        print(f"'{func.__name__}' elapsed time: {elapsed}")
        return result

    return wrapper


def get_ERA5_geopotenital_height_500hPa(dt):
    """Open and format ERA5 geopotential at 500 hPa.

    Parameters
    ----------
    dt : datetime
        Target year and month the ERA5 data to be opened.

    Returns
    -------
    z : xarray.DataArray
        Geopotential height [gpm] at 500 hPa at 3-hour intervals.
    """

    # Open geopotential (z)
    z = xr.open_mfdataset(
        f"/g/data/rt52/era5/pressure-levels/reanalysis/z/{dt.year}/z_era5_oper_pl_{dt.strftime('%Y%m')}*.nc"
    )
    # Convert to geopotential height [gpm] (meters above sea level)
    z = z.z / g
    z.attrs["units"] = "gpm"
    # Select variable at 500hPa
    z = z.sel(level=500)
    # Resample to 3 hourly intervals (ERA5 is 1-hourly)
    z = z.resample(time="3h").nearest()
    return z


def get_ERA5_geopotenital_height_sfc():
    """Open and format ERA5 geopotential at surface for orography."""

    # Open surface geopotential (time invariant)
    z_sfc = xr.open_dataset("/g/data/xv83/as3189/vicwaci/data/z_era5_oper_sfc.nc")
    z_sfc = z_sfc["z"]

    # Convert to geopotential height [gpm] (meters above sea level)
    z_sfc /= g  # m2 s-2 / m s-2 = m
    z_sfc.attrs["units"] = "gpm"
    return z_sfc


@time_it
def detect_cyclones(dt, cyc_minprominence, sensitivity_analysis=False, test=False):
    """Detect cyclones at 500 hPa in ERA5 and save the output to netcdf.

    Parameters
    ----------
    dt : datetime
        Target year and month the ERA5 data to be opened.
    cyc_minprominence : float
        Minimum prominence of cyclones to be detected [gpm].
    sensitivity_analysis : bool, optional
        If True, save output for sensitivity analysis of cyc_minprominence. Default is False.
    test : bool, optional
        If True, only run for a single day for testing. Default is False
    Notes
    -----
    - Output of dynlib.detect.cyclone_by_contour() is a cyclone mask and cycmeta data about each cyclone. Only saving cycmask to netcdf.
        cycmask: (time, lat, lon)
            Cyclone mask with different integers designating different cyclones
        cycmeta: (time, nn, 5)
            cycmeta data about each cyclone: (1) latitude and (2) longitude of centre, (3) minimum SLP, (4) SLP at the outermost contour, and (5) cyclone size. Note that nn = 1000 (Maximum number of cyclones to be detected)
    """

    path_features = "/g/data/xv83/as3189/vicwaci/data/weather_features/cyclones_500hpa"
    outfile = (
        f"{path_features}/ea.ans.{dt.strftime('%Y%m')}.500.cycmask_min{cyc_minprominence}gpm.nc"
    )

    if sensitivity_analysis:
        # @todo: change filename convention to match full files
        path_features += f"/sensitivity_analysis/min{cyc_minprominence}gpm"
        outfile = f"{path_features}/ea.ans.{dt.strftime('%Y%m')}.500.cycmask.nc"

    if os.path.exists(outfile):
        # Check if file opens correctly. If not, delete and run.
        try:
            ds = xr.open_dataset(outfile)
            print(f"File {outfile} already exists and opens correctly.")
            ds.close()
            return
        except Exception as e:
            print(f"File {outfile} already exists but cannot be opened. Deleting and re-running.")
            raise e

    # Open geopotential (z) - NB: z on sfc is time invariant, but not on pressure levels
    z = get_ERA5_geopotenital_height_500hPa(dt)
    z_sfc = get_ERA5_geopotenital_height_sfc()

    if test:
        z = z.sel(time=dt.strftime("%Y-%m-%d"))

    # Create grid instance with orography
    lon2d, lat2d = np.meshgrid(z_sfc.longitude, z_sfc.latitude)
    grid = gridlib.grid_by_latlon(lat2d, lon2d)
    grid.oro = np.ascontiguousarray(z_sfc.values, dtype=np.float64)

    # Convert z to contiguous array for dynlib
    z_values = np.ascontiguousarray(z.values, dtype=np.float64)

    dynfor.config.cyc_minprominence = cyc_minprominence
    cycmask, cycmeta = detect.cyclone_by_contour(z_values, grid)

    # Should have dims (time, lat, lon) with a unique integer for each cyclone
    cycmask = xr.DataArray(cycmask, dims=z.dims, coords=z.coords, name="cycmask")
    # cycmeta = xr.DataArray(cycmeta, dims=("time", "nn", "info"), name="cycmeta")

    # Convert to 0-1 mask
    cycmask = xr.where(cycmask != 0, 1, 0)
    # Convert to int8 to save space
    cycmask = cycmask.astype(np.int8)

    # Format output dataset and save to netcdf
    ds = cycmask.to_dataset(name="cycmask")
    ds["cycmask"].attrs["long_name"] = "Cyclone detection mask"
    ds["cycmask"].attrs["units"] = "(0-1)"
    ds["cycmask"].attrs[
        "description"
    ] = f"Mask of cyclones detected at 500 hPa in ERA5 using dynlib.detect.cyclone_by_contour using a minimum prominence of {cyc_minprominence} gpm"
    ds.attrs["history"] = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Created {outfile} by dynlib-{dynlib.version}"
    )

    encoding = {"zlib": True, "complevel": 4, "shuffle": True}
    ds.to_netcdf(outfile, compute=True, encoding={"cycmask": encoding})
    print(f"Saved cyclone mask to {outfile}")
    # Clean up - close datasets and delete variables to free memory
    ds.close()
    z.close()
    z_sfc.close()
    del z, z_sfc, z_values, cycmask, cycmeta, ds
    return


if __name__ == "__main__":
    # use argparse to get command line arguments for min (cyc_minprominence)
    client = Client()
    parser = argparse.ArgumentParser(
        description="Detect cyclones at 500 hPa in ERA5 and save the output to netcdf."
    )
    parser.add_argument("-y1", type=str, default="1980")
    parser.add_argument("-y2", type=str, default="2022")
    parser.add_argument(
        "--min",
        type=int,
        default=30,
        help="Minimum prominence of cyclones to be detected [gpm]. Default is 30 gpm.",
    )
    parser.add_argument(
        "--s",
        action="store_true",
        help="Run a sensitivity analysis for different values of cyc_minprominence. Default is False.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test using a small time slice. Default is False.",
    )

    args = parser.parse_args()
    cyc_minprominence = args.min
    y1 = args.y1
    y2 = args.y2
    sensitivity_analysis = args.s

    if sensitivity_analysis:
        y1, y2 = "1981", "1981"
    date_range = xr.date_range(f"{y1}-01-01", f"{y2}-12-31", freq="1MS")

    for dt in tqdm(date_range, desc="Detecting cyclones"):
        detect_cyclones(
            dt, cyc_minprominence=cyc_minprominence, sensitivity_analysis=sensitivity_analysis
        )
