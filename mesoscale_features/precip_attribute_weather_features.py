"""Run precipitation attribution to weather features over Australia.

Save monthly files for the period 1980-01 to 2022-12

Notes
-----
- Based on ACCESS-NRI's NCI xp65 anaylsis3-25.11 conda environment
- Uses  M. Barnes local installation of dynlib
    - https://folk.uib.no/csp001/dynlib_doc/
    - https://git.app.uib.no/Clemens.Spensberger/dynlib/-/blob/master/
- Data:
    - ERA5 total precipitation (tp) (project rt52 on NCI)
- Function runtime with 6cpus and ~30GB takes around 5minutes, but it gets slower each loop iteration.
    - Suggest running in 1 year batches

Example
-------
```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-25.11
python /g/data/xv83/as3189/vicwaci/precip_attribute_weather_features.py -m1 ${m1} -y1 ${y1} -m2 ${m2} -y2 ${y2}
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
from dynlib import utils

import argparse
import calendar
from datetime import datetime, timedelta
from dask.distributed import Client
from functools import wraps
import gc
import numpy as np
import time
from tqdm import tqdm
import xarray as xr


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


def format_filename_dates(dt):
    """Format the start and end dates of a month for ERA5 filenames."""
    start = dt.replace(day=1)
    end = dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
    return f"{start:%Y%m%d}-{end:%Y%m%d}"


def mta_lines_to_mask(ds, grid):
    """Convert MTA lines to a mask on the target grid.

    Parameters
    ----------
    ds: Dataset of MTA lines (dims=index)
    grid: Dataset containing target latitude and longitude

    Returns
    -------
    mask : xarray.DataArray
        Mask of shape (time, latitude, longitude)
    """

    ds = ds.rename({"date": "time"})
    # Convert to DataFrame
    df = ds[["time", "latitude", "longitude", "viwvt"]].to_dataframe()
    df = df.reset_index(drop=True)

    # Convert positions to nearest lat/lon values
    df["latitude"] = grid.latitude.sel(latitude=xr.DataArray(df.latitude), method="nearest").values
    df["longitude"] = grid.longitude.sel(
        longitude=xr.DataArray(df.longitude), method="nearest"
    ).values

    # If several points fall into the same grid cell, retain the maximum viwvt.
    df = df.groupby(["time", "latitude", "longitude"], as_index=False)["viwvt"].max()
    # Convert back to xarray
    mask = df.set_index(["time", "latitude", "longitude"])["viwvt"].to_xarray()

    # Ensure time, lat lon grids are complete
    mask = mask.reindex(
        time=np.unique(ds.time),
        latitude=grid.latitude,
        longitude=grid.longitude,
        fill_value=np.nan,
    )
    return mask


@time_it
def run_attribution(
    dt,
    path_features,
    path_output,
    min_precip_mm=0.1,
    cyc_minprominence=30,
    chunks=None,
    test_run=False,
):
    """Detect cyclones at 500 hPa in ERA5 and save the output to netcdf.

    Parameters
    ----------
    dt : datetime
        Target year and month the ERA5 data to be opened.
    cyc_minprominence : float
        Minimum prominence of cyclones to be detected [gpm].
    test_run : bool, optional
        If True, only run for a single day for testing. Default is False

    """

    time_str = dt.strftime("%Y%m")
    outfile = f"{path_output}/ERA5_precip_attribution_{time_str}_aus_min{cyc_minprominence}gpm.nc"
    if os.path.exists(outfile):
        # Check if file opens correctly. If not, delete and run.
        try:
            ds = xr.open_dataset(outfile)
            print(f"File {outfile} already exists and opens correctly. Skipping.")
            ds.close()
            return
        except Exception as e:
            print(f"File {outfile} already exists but cannot be opened. Deleting and re-running.")
            print(f"Error: {e}")
            # os.remove(outfile)
            return

    # C: ERA5 cyclones (surface)
    file_cyc = f"{path_features}/cyclones_ETH/ea.ans.{time_str}.sfc.cycmask.nc"
    ds_cyc = xr.open_dataset(file_cyc, chunks=chunks)

    # V: ERA5 cyclones (at 500 hPa)
    file_cyc_500 = f"{path_features}/cyclones_500hpa/ea.ans.{time_str}.500.cycmask_min{cyc_minprominence}gpm.nc"
    ds_cyc_500 = xr.open_dataset(file_cyc_500, chunks=chunks)
    ds_cyc_500 = ds_cyc_500.drop_vars("level")

    # F: ERA5 fronts
    file_fronts = f"{path_features}/fronts/ea.ans.{time_str}.850.frovo_id.nc"
    ds_fronts = xr.open_dataset(file_fronts, chunks=chunks)
    ds_fronts = ds_fronts.isel(lev=0, drop=True)

    # M: ERA5 MTAs
    file_mta = f"{path_features}/mta/ea.ans.{time_str}.sfc.mta.nc"
    ds_mta_lines = xr.open_dataset(file_mta, chunks=chunks)
    ds_mta = mta_lines_to_mask(ds_mta_lines, ds_cyc)
    ds_mta = xr.where(~np.isnan(ds_mta), ds_mta, 0)

    # Open ERA5 precip (tp)
    # Open the previous day's file as well to ensure we have the full 3-hourly aggregation for the first timestep of the current day.
    dt_prev = dt - timedelta(days=1)
    file_pr_prev = f"/g/data/rt52/era5/single-levels/reanalysis/tp/{dt_prev.year}/tp_era5_oper_sfc_{format_filename_dates(dt_prev)}.nc"
    file_pr = f"/g/data/rt52/era5/single-levels/reanalysis/tp/{dt.year}/tp_era5_oper_sfc_{format_filename_dates(dt)}.nc"

    ds_pr = xr.open_mfdataset([file_pr_prev, file_pr], chunks=chunks)
    # Convert from hourly to 3hr aggregate preceding each timestep
    ds_pr = ds_pr.resample(time="3h", label="right").sum()
    ds_pr = ds_pr.sel(time=ds_cyc.time)  # Ensure time alignment with cyclones
    # Update units
    ds_pr["tp"] *= 1e3
    ds_pr["tp"].attrs["units"] = "mm 3hr-1"

    # Apply minimum precipitation threshold
    ds_pr = ds_pr.where(ds_pr > min_precip_mm)

    # Create a DataArray with the same dimensions and coordinates
    dims = list(ds_cyc.dims)
    coords = {dim: ds_cyc[dim].values for dim in dims}

    # Create dataset of variables
    da = xr.Dataset(None, coords=ds_cyc.coords)
    da["tp"] = ds_pr["tp"]
    da["C"] = ds_cyc.cycmask > 0
    da["V"] = ds_cyc_500.cycmask > 0
    da["F"] = ds_fronts.frovo_id > 0
    da["M"] = xr.DataArray(ds_mta != 0, dims=dims, coords=coords, name="mta")

    # Create mask conditions (letters correspond to the variable in the dataset)
    masks_conditions = [
        ("C", lambda x: (x.C > 0)),  # cyclone mask
        ("V", lambda x: (x.V > 0)),  # cyclone mask (500hPa)
        ("F", lambda x: (x.F > 0)),  # front mask
        ("M", lambda x: (x.M > 0)),  # MTA mask (lines)
    ]

    # Domain boundary (applied before attribution)
    lat_slice, lon_slice = slice(-5, -65), slice(90, 180)
    # Subset the data to the domain of interest (Australia)
    da = da.sel(latitude=lat_slice, longitude=lon_slice)

    # https://git.app.uib.no/Clemens.Spensberger/dynlib/-/blob/master/lib/utils.py?ref_type=heads#L997
    # Factor of -1 (indicates precip maxima rather than minima)
    ds = utils.attribute_to_features(da, masks_conditions, var="tp", factor=-1)

    for i, v in enumerate(tqdm(list(ds.data_vars))):
        if i == 0:
            total_vars = ds[v]
        else:
            total_vars = total_vars + ds[v]

    # Add total precipitation to the dataset
    ds["total"] = total_vars.rename("total")

    # Add metadata to the dataset
    ds["C"].attrs["description"] = "Precipitation attributed to surface cyclones"
    ds["V"].attrs["description"] = "Precipitation attributed to cyclones at 500 hPa"
    ds["F"].attrs["description"] = "Precipitation attributed to fronts"
    ds["M"].attrs["description"] = "Precipitation attributed to moisture transport axis (lines)"
    ds["U"].attrs["description"] = "Unattributed precipitation"
    ds["total"].attrs[
        "description"
    ] = f"Total precipitation > {min_precip_mm} mm 3hr-1 (attributed and unattributed)"
    ds.attrs["description"] = (
        f"Precipitation attribution to weather features (cyclones, fronts, moisture transport axis) for ERA5 over Australia. Minimum precipitation threshold: {min_precip_mm} mm 3hr-1. Cyclone detection at 500hPa with minimum prominence of {cyc_minprominence} gpm."
    )

    ds.attrs["history"] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by {dynlib.version}"

    var_encoding = {"zlib": True, "complevel": 4, "shuffle": True}
    encoding = {var: var_encoding for var in ds.data_vars}

    ds.to_netcdf(outfile, compute=True, encoding=encoding)
    print(f"Saved attribution to {outfile}")

    ds_cyc.close()
    ds_cyc_500.close()
    ds_fronts.close()
    ds_mta_lines.close()
    ds_mta.close()
    ds_pr.close()
    ds.close()
    da.close()
    gc.collect()
    return


if __name__ == "__main__":
    client = Client()
    parser = argparse.ArgumentParser(
        description="Attribute ERA5 weather features to precipitation and save the output to netcdf."
    )
    parser.add_argument("-y1", type=str, default="1980")
    parser.add_argument("-y2", type=str, default="01")
    parser.add_argument("-m1", type=str, default="2022")
    parser.add_argument("-m2", type=str, default="12")

    args = parser.parse_args()
    y1, y2 = args.y1, args.y2
    m1, m2 = args.m1, args.m2

    # Set paths and parameters
    path_features = "/g/data/xv83/as3189/vicwaci/data/weather_features"
    path_output = "/g/data/xv83/as3189/vicwaci/data/precip_attribution"

    min_precip_mm = 0.1  # Minimum precipitation (mm) threshold
    cyc_minprominence = 30  # used for cyclones at 500hPa

    # y1, y2 = 1980, 2022
    date_range = xr.date_range(f"{y1}-{m1}-01", f"{y2}-{m2}-01", freq="1MS")
    for dt in tqdm(date_range):
        run_attribution(
            dt,
            path_features=path_features,
            path_output=path_output,
            min_precip_mm=min_precip_mm,
            cyc_minprominence=cyc_minprominence,
            chunks=None,
        )
