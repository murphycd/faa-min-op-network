#!/usr/bin/env uv run
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import numpy

import pandas
import requests

# https://www.faa.gov/ato/navigation-programs/vor-retention-list
vor_retention_source = {
    "filename": "vor_retention.csv",
    "url": "https://www.faa.gov/sites/faa.gov/files/2022-02/VOR_Retention_List_2022-2-7.xlsx",
    "id_column": "ID",
}

# https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
nav_data_source = {
    "filename": "nav_data.csv",
    "url": "https://nfdc.faa.gov/webContent/28DaySub/extra/03_Sep_2026_NAV_CSV.zip",
    "internal_name": "NAV_BASE.csv",
    "id_column": "NAV_ID",
    "keep_columns": ["NAV_ID", "NAV_TYPE", "NAME", "LAT_DECIMAL", "LONG_DECIMAL", "ELEV", "MAG_VARN", "ALT_CODE"],
}

mon_data_constants = {
    "filename": "mon_data.csv",
}

_RAW_SERVICE_VOLUMES_VOR = [
    ("T", 1000, 12000, 25),
    ("L", 1000, 18000, 40),
    ("H", 1000, 14499, 40),
    ("H", 14500, 17999, 100),
    ("H", 18000, 45000, 130),
    ("H", 45000, 99999, 100),
    ("VL", 1000, 4999, 40),
    ("VL", 5000, 17999, 70),
    ("VH", 1000, 4999, 40),
    ("VH", 5000, 14499, 70),
    ("VH", 14500, 17999, 100),
    ("VH", 18000, 45000, 130),
    ("VH", 45000, 99999, 100),
]
service_volumes_vor = [
    {"class": cls, "altitude": {"min": min_alt, "max": max_alt}, "miles": miles}
    for cls, min_alt, max_alt, miles in _RAW_SERVICE_VOLUMES_VOR
]

# force redownload data from faa website even if files already exist
REDOWNLOAD = False

# force reparse of downloaded data into mon_data even if file already exists
REPARSE = False


def main():

    working_dir = Path(__file__).parent / "data"
    working_dir.mkdir(exist_ok=True)

    mon_data = get_mon_data(working_dir)

    print(mon_data) # debug
    
    frd_data = generate_frd_data(
        nav_set_df=mon_data,
        step_size=1,
        start_dist=0,
        end_dist=130,
        start_radial=0,
        end_radial=359,
        radial_step=1,
    )

    print(frd_data)


def generate_frd_data(
    nav_set_df,
    step_size=1,
    start_dist=0,
    end_dist=130,
    start_radial=0,
    end_radial=359,
    radial_step=1,
):
    """For each starting point in nav_set_df, generates an FRD grid from start

    to end distance and start to end radial (inclusive).
    """
    # 1. Generate 1D ranges for both dimensions
    radial_range = numpy.arange(
        start_radial, end_radial + radial_step, radial_step
    )

    if start_dist == 0:
        # Exclude origin from grid expansion to avoid radial redundancy at zero distance
        non_zero_dists = numpy.arange(
            start_dist + step_size, end_dist + step_size, step_size
        )
        dist_grid, radial_grid = numpy.meshgrid(non_zero_dists, radial_range)

        # 2. Flatten grid combinations and prepend a single origin point
        step_dists = numpy.hstack(([0], dist_grid.ravel()))[numpy.newaxis, :]
        step_radials = numpy.hstack(([start_radial], radial_grid.ravel()))[
            numpy.newaxis, :
        ]
    else:
        dist_range = numpy.arange(start_dist, end_dist + step_size, step_size)
        dist_grid, radial_grid = numpy.meshgrid(dist_range, radial_range)
        step_dists = dist_grid.ravel()[numpy.newaxis, :]
        step_radials = radial_grid.ravel()[numpy.newaxis, :]

    points_per_seed = step_dists.shape[1]
    # 4. Convert input columns to 2D column vectors: Shape (N, 1)
    lats = nav_set_df["LAT_DECIMAL"].to_numpy()[:, numpy.newaxis]
    lons = nav_set_df["LONG_DECIMAL"].to_numpy()[:, numpy.newaxis]
    mag_vars = nav_set_df["MAG_VARN"].to_numpy()[:, numpy.newaxis]

    # 5. Execute the custom calculation logic
    # Outputs will all have the shape: (N, total_combinations)
    new_lat, new_lon, new_radial, new_dist = custom_point_logic(
        lats, lons, mag_vars, step_radials, step_dists
    )

    # 6. Flatten the 2D structures into 1D vectors for the final table
    flat_lat = new_lat.ravel()
    flat_lon = new_lon.ravel()
    flat_radial = new_radial.ravel()
    flat_dist = new_dist.ravel()

    # 7. Repeat structural tracking metadata (IDs, SSVs) so rows align perfectly
    flat_ids = numpy.repeat(nav_set_df["NAV_ID"].to_numpy(), points_per_seed)
    flat_ssvs = numpy.repeat(nav_set_df["ALT_CODE"].to_numpy(), points_per_seed)

    # 8. Construct the final massive DataFrame exactly once
    output_df = pandas.DataFrame(
        {
            "parent_id": flat_ids,
            "generated_lat": flat_lat,
            "generated_lon": flat_lon,
            "radial": flat_radial,
            "distance": flat_dist,
            "ssv": flat_ssvs,
        }
    )

    return output_df


def custom_point_logic(lats, lons, mag_vars, step_radials, step_dists):
    """Your custom navigation/geospatial math goes here.

    Inputs are shaped for 2D broadcasting:
    - lats, lons, mag_vars: Shape (N, 1)
    - step_radials, step_dists:   Shape (1, execution_steps)

    Returns: Four arrays of shape (N, execution_steps)
    """
    # Adjust radial by magnetic variation to obtain true bearing
    true_bearing = step_radials + mag_vars  # Shape: (N, execution_steps)

    # Convert angular quantities to radians for spherical trigonometry
    bearing_rad = numpy.radians(true_bearing)
    lat_rad = numpy.radians(lats)
    lon_rad = numpy.radians(lons)

    # Angular distance delta (assuming 1 nautical mile = 1/60th of a degree arc)
    angular_dist = numpy.radians(step_dists / 60.0)

    # Great circle destination calculation (spherical direct geodesic problem)
    sin_lat = numpy.sin(lat_rad)
    cos_lat = numpy.cos(lat_rad)
    sin_delta = numpy.sin(angular_dist)
    cos_delta = numpy.cos(angular_dist)
    sin_bearing = numpy.sin(bearing_rad)
    cos_bearing = numpy.cos(bearing_rad)

    new_lat_rad = numpy.arcsin(
        sin_lat * cos_delta + cos_lat * sin_delta * cos_bearing
    )
    new_lon_rad = lon_rad + numpy.arctan2(
        sin_bearing * sin_delta * cos_lat,
        cos_delta - sin_lat * numpy.sin(new_lat_rad),
    )

    new_lat = numpy.degrees(new_lat_rad)
    new_lon = numpy.degrees(new_lon_rad)

    # Wrap longitudes to standard [-180, 180] degree range
    new_lon = (new_lon + 180.0) % 360.0 - 180.0

    # Track output metrics (pass the grids through so they map to rows)
    # Multiplying by numpy.ones_like(lats) ensures it expands to shape (N, execution_steps)
    new_radial = step_radials * numpy.ones_like(lats)
    new_dist = step_dists * numpy.ones_like(lats)

    return new_lat, new_lon, new_radial, new_dist


def get_mon_data(working_dir):
    mon_data_file = working_dir / mon_data_constants["filename"]
    if REPARSE or not mon_data_file.is_file():
        print(f"MON data file {mon_data_file} not found. Constructing MON data...")
        mon_data = construct_mon_data(working_dir)
    else:
        print(f"Loading MON data from {mon_data_file}...")
        mon_data = pandas.read_csv(mon_data_file)
        
    return mon_data


def construct_mon_data(working_dir):
    # load or fetch RET data
    ret_file = working_dir / vor_retention_source["filename"]
    if REDOWNLOAD or not ret_file.is_file():
        print(f"Fetching RET data from {vor_retention_source['url']}...")
        response = requests.get(vor_retention_source["url"])
        response.raise_for_status()
        ret_data = pandas.read_excel(io.BytesIO(response.content))
        ret_data = strip_metadata_rows(ret_data)
        ret_data.to_csv(ret_file, index=False)
    else:
        print(f"Loading RET data from {ret_file}...")
        ret_data = pandas.read_csv(ret_file)

    # load or fetch NAV data
    nav_data_file = working_dir / nav_data_source["filename"]
    if REDOWNLOAD or not nav_data_file.is_file():
        print(f"Fetching NAV data from {nav_data_source['url']}...")
        response = requests.get(nav_data_source["url"])
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open(nav_data_source["internal_name"]) as file:
                nav_data = pandas.read_csv(file)
        nav_data.to_csv(nav_data_file, index=False)
    else:
        print(f"Loading NAV data from {nav_data_file}...")
        nav_data = pandas.read_csv(nav_data_file)

    # Remove NAV rows not found in the RET ID column
    valid_ids = set(ret_data[vor_retention_source["id_column"]])
    invalid_row_mask = ~nav_data[nav_data_source["id_column"]].isin(valid_ids)
    nav_data.drop(index=nav_data[invalid_row_mask].index, inplace=True)

    # Keep VOR-type facilities only
    valid_nav_types = ["VOR", "VOR/DME", "VORTAC"]
    non_vor_mask = ~nav_data["NAV_TYPE"].isin(valid_nav_types)
    nav_data.drop(index=nav_data[non_vor_mask].index, inplace=True)

    # Remove unlisted columns
    unwanted_columns = nav_data.columns.difference(nav_data_source["keep_columns"])
    nav_data.drop(columns=unwanted_columns, inplace=True)

    # Reset index after row/column operations
    nav_data.reset_index(drop=True, inplace=True)

    mon_data_file = working_dir / mon_data_constants["filename"]
    nav_data.to_csv(mon_data_file, index=False)
    print(f"Saved MON data to {mon_data_file}...")

    return nav_data


def strip_metadata_rows(df: pandas.DataFrame) -> pandas.DataFrame:
    # The header row in poorly formatted tabular exports typically represents the first
    # fully populated row. idxmax() returns the first occurrence of the maximum non-null count,
    # reliably isolating the header from sparse metadata rows above it.
    valid_counts = df.notna().sum(axis=1)
    header_idx = valid_counts.idxmax()

    # Casting to string ensures uniform column name types, mitigating issues
    # with mixed-type inference from the original row values.
    new_columns = (
        df.iloc[header_idx]  # pyright: ignore[reportArgumentType, reportCallIssue]
        .fillna("")
        .astype(str)
        .tolist()
    )
    df_cleaned = df.iloc[
        header_idx + 1 :  # pyright: ignore[reportOperatorIssue]
    ].copy()
    df_cleaned.columns = new_columns

    return df_cleaned.reset_index(drop=True)


if __name__ == "__main__":
    main()
