#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "geographiclib",
#   "numpy",
#   "openpyxl",
#   "pandas",
#   "pyarrow",
#   "requests",
# ]
# ///

import io
import logging
import zipfile
from collections.abc import Hashable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geographiclib.geodesic import Geodesic
from pandas._typing import DtypeArg

# Force redownload of FAA source data even if cached files already exist.
REDOWNLOAD_VOR_RET = False

# Force reconstruction of MON data even if the cached MON file exists.
REPARSE_MON = False

# Force reconstruction of FRD data even if the cached FRD file exists.
REPARSE_FRD = False

logger = logging.getLogger(__name__)

# https://www.faa.gov/ato/navigation-programs/vor-retention-list
VOR_RETENTION_SOURCE = {
    "filename": "vor_retention.csv",
    "url": "https://www.faa.gov/sites/faa.gov/files/2022-02/VOR_Retention_List_2022-2-7.xlsx",
    "id_column": "ID",
}

# https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
NAV_DATA_SOURCE = {
    "filename": "nav_data.csv",
    "url": "https://nfdc.faa.gov/webContent/28DaySub/extra/03_Sep_2026_NAV_CSV.zip",
    "internal_name": "NAV_BASE.csv",
    "id_column": "NAV_ID",
    "keep_columns": [
        "NAV_ID",
        "NAV_TYPE",
        "NAME",
        "LAT_DECIMAL",
        "LONG_DECIMAL",
        "ELEV",
        "MAG_VARN",
        "ALT_CODE",
    ],
}

MON_DATA_FILENAME = "mon_data.csv"

FRD_DATA_FILENAME = "frd_data.parquet"

VALID_NAV_TYPES = frozenset({"VOR", "VOR/DME", "VORTAC"})

# Strictly type alphanumeric identifiers to prevent pandas from coercing them (e.g. VOR ID "00A")
CSV_DTYPE_MAP = Mapping[Hashable, DtypeArg]

NAV_SCHEMA: CSV_DTYPE_MAP = {
    "NAV_ID": "str",
    "NAV_TYPE": "str",
    "ALT_CODE": "str",
}

MON_SCHEMA: CSV_DTYPE_MAP = {
    "NAV_ID": "str",
    "NAV_TYPE": "str",
    "NAME": "str",
    "LAT_DECIMAL": "float",
    "LONG_DECIMAL": "float",
    "ELEV": "float",
    "MAG_VARN": "float",
    "ALT_CODE": "str",
}

FRD_COLUMNS = [
    "parent_id",
    "generated_lat",
    "generated_lon",
    "radial",
    "distance",
    "min_alt",
    "max_alt",
]

FRD_DTYPES = {
    "generated_lat": np.float32,
    "generated_lon": np.float32,
    "radial": np.uint16,
    "distance": np.uint8,
    "min_alt": np.uint16,
    "max_alt": np.uint16,
}
FRD_LATLON_PRECISION = 4  # decimal places for lat/lon rounding

FRD_PARQUET_COMPRESSION = "brotli"
FRD_PARQUET_COMPRESSION_LEVEL = 5

# Standard service-volume rules:
# (class, minimum altitude AGL, maximum altitude AGL, maximum distance NM)
RAW_SERVICE_VOLUMES_VOR = (
    ("T", 1000, 12000, 25),
    ("L", 1000, 18000, 40),
    ("H", 1000, 14499, 40),
    ("H", 14500, 17999, 100),
    ("H", 18000, 45000, 130),
    ("H", 45000, 60000, 100),
    ("VL", 1000, 4999, 40),
    ("VL", 5000, 17999, 70),
    ("VH", 1000, 4999, 40),
    ("VH", 5000, 14499, 70),
    ("VH", 14500, 17999, 100),
    ("VH", 18000, 45000, 130),
    ("VH", 45001, 60000, 100),
)

# WGS84 ellipsoid used for the geodesic destination calculation.
WGS84 = Geodesic.WGS84


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    working_dir = Path(__file__).resolve().parent / "data"
    working_dir.mkdir(exist_ok=True)

    mon_data = get_mon_data(working_dir)
    frd_data = get_frd_data(working_dir, mon_data)

    logger.info("\n%s", frd_data)


def load_cached_csv(
    filepath: Path,
    dtypes: CSV_DTYPE_MAP | None = None,
) -> pd.DataFrame | None:
    """Safely load a CSV cache file. Returns None if invalid, empty, or corrupted."""
    if not filepath.is_file() or filepath.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(filepath, dtype=dtypes, low_memory=False)
    except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.warning(
            "Invalidating CSV cache at %s due to read error: %s", filepath, e
        )
        return None


def validate_nav_set(nav_set_df: pd.DataFrame) -> None:
    """Validate columns and numeric geographic inputs required by FRD generation."""
    required_columns = {
        "NAV_ID",
        "LAT_DECIMAL",
        "LONG_DECIMAL",
        "ELEV",
        "MAG_VARN",
        "ALT_CODE",
    }
    missing_columns = required_columns - set(nav_set_df.columns)
    if missing_columns:
        raise ValueError(
            f"MON data is missing required columns: {sorted(missing_columns)}"
        )

    if nav_set_df.empty:
        raise ValueError("MON data contains no navigation facilities")

    numeric_columns = [
        "LAT_DECIMAL",
        "LONG_DECIMAL",
        "MAG_VARN",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(
            nav_set_df[column],
            errors="coerce",
        )
        if values.isna().any():
            raise ValueError(
                f"MON data contains non-numeric or missing values in {column!r}"
            )

    latitudes = pd.to_numeric(
        nav_set_df["LAT_DECIMAL"],
        errors="coerce",
    )
    longitudes = pd.to_numeric(
        nav_set_df["LONG_DECIMAL"],
        errors="coerce",
    )

    if not latitudes.between(-90.0, 90.0).all():
        raise ValueError("MON data contains latitude outside [-90, 90]")

    if not longitudes.between(-180.0, 180.0).all():
        raise ValueError("MON data contains longitude outside [-180, 180]")

    unknown_ssvs = set(nav_set_df["ALT_CODE"].dropna()) - {
        rule[0] for rule in RAW_SERVICE_VOLUMES_VOR
    }
    if unknown_ssvs:
        raise ValueError(
            f"MON data contains unsupported ALT_CODE values: {sorted(unknown_ssvs)}"
        )


def validate_frd_data(frd_data: pd.DataFrame) -> None:
    """Validate generated FRD cache shape before reusing it."""
    missing_columns = set(FRD_COLUMNS) - set(frd_data.columns)
    if missing_columns:
        raise ValueError(
            f"FRD data is missing required columns: {sorted(missing_columns)}"
        )

    if frd_data.empty:
        raise ValueError("FRD data contains no generated points")


def compact_frd_data(frd_data: pd.DataFrame) -> pd.DataFrame:
    """Return FRD data using compact storage dtypes."""
    validate_frd_data(frd_data)

    compacted = frd_data.copy()
    # Round to fixed decimal places, then cast to float32
    compacted["generated_lat"] = (
        compacted["generated_lat"]
        .round(FRD_LATLON_PRECISION)
        .astype(FRD_DTYPES["generated_lat"])
    )

    compacted["generated_lon"] = (
        compacted["generated_lon"]
        .round(FRD_LATLON_PRECISION)
        .astype(FRD_DTYPES["generated_lon"])
    )

    for column, dtype in (
        ("radial", FRD_DTYPES["radial"]),
        ("distance", FRD_DTYPES["distance"]),
        ("min_alt", FRD_DTYPES["min_alt"]),
        ("max_alt", FRD_DTYPES["max_alt"]),
    ):
        values = pd.to_numeric(compacted[column], errors="raise")
        float_values = values.to_numpy(dtype=float)
        if np.isnan(float_values).any():
            raise ValueError(f"FRD column {column!r} contains missing values")

        rounded = np.rint(float_values)
        if column in {"radial", "distance"} and not np.array_equal(
            float_values,
            rounded,
        ):
            raise ValueError(
                f"FRD column {column!r} cannot be safely rounded to {dtype}"
            )

        if column in {"min_alt", "max_alt"}:
            # Caps values at 65535, keeping 0 as the lower floor
            rounded = np.clip(rounded, a_min=0, a_max=65535)

        integer_values = pd.Series(rounded, index=compacted.index)
        if integer_values.lt(0).any() or integer_values.gt(np.iinfo(dtype).max).any():
            raise ValueError(f"FRD column {column!r} contains values outside {dtype}")

        compacted[column] = integer_values.to_numpy(dtype=np.dtype(dtype))

    return compacted


def get_frd_data(working_dir: Path, mon_data: pd.DataFrame) -> pd.DataFrame:
    frd_data_file = working_dir / FRD_DATA_FILENAME
    frd_data = None

    if not REPARSE_FRD and frd_data_file.is_file() and frd_data_file.stat().st_size > 0:
        try:
            logger.info("Loading FRD data from %s...", frd_data_file)
            frd_data = pd.read_parquet(frd_data_file)
            validate_frd_data(frd_data)
        except Exception as e:
            logger.warning("Invalidating FRD cache due to read error: %s", e)
            frd_data = None

    if frd_data is None:
        logger.info("Constructing FRD data at %s...", frd_data_file)
        frd_data = generate_frd_data(
            nav_set_df=mon_data,
            step_dist=1.0,
            start_dist=0.0,
            end_dist=130.0,
            start_radial=0.0,
            end_radial=359.0,
            radial_step=1.0,
        )
        frd_data = compact_frd_data(frd_data)
        validate_frd_data(frd_data)
        frd_data.to_parquet(
            frd_data_file,
            compression=FRD_PARQUET_COMPRESSION,
            compression_level=FRD_PARQUET_COMPRESSION_LEVEL,
            index=False,
        )
        logger.info("Saved FRD data to %s...", frd_data_file)

    return frd_data


def get_mon_data(working_dir: Path) -> pd.DataFrame:
    mon_data_file = working_dir / MON_DATA_FILENAME
    mon_data = None

    if not REPARSE_MON:
        logger.info("Attempting to load MON data from %s...", mon_data_file)
        mon_data = load_cached_csv(mon_data_file, dtypes=MON_SCHEMA)

    if mon_data is None:
        logger.info("Constructing MON data at %s...", mon_data_file)
        mon_data = construct_mon_data(working_dir)
        mon_data.to_csv(mon_data_file, index=False)
        logger.info("Saved MON data to %s...", mon_data_file)

    validate_nav_set(mon_data)
    return mon_data


def construct_mon_data(working_dir: Path) -> pd.DataFrame:
    # Load or fetch RET data.
    ret_file = working_dir / VOR_RETENTION_SOURCE["filename"]
    ret_data = (
        None
        if REDOWNLOAD_VOR_RET
        else load_cached_csv(
            ret_file, dtypes={VOR_RETENTION_SOURCE["id_column"]: "str"}
        )
    )

    if ret_data is None:
        url = VOR_RETENTION_SOURCE["url"]
        logger.info("Fetching RET data from %s...", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        ret_data = pd.read_excel(io.BytesIO(response.content))
        ret_data = strip_metadata_rows(ret_data)
        ret_data.to_csv(ret_file, index=False)

    # Load or fetch NAV data.
    nav_data_file = working_dir / NAV_DATA_SOURCE["filename"]
    nav_data = (
        None
        if REDOWNLOAD_VOR_RET
        else load_cached_csv(nav_data_file, dtypes=NAV_SCHEMA)
    )

    if nav_data is None:
        url = NAV_DATA_SOURCE["url"]
        logger.info("Fetching NAV data from %s...", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            internal_name = NAV_DATA_SOURCE["internal_name"]
            if internal_name not in zip_file.namelist():
                raise ValueError(f"{internal_name!r} was not found in FAA NAV archive")

            with zip_file.open(internal_name) as file:
                nav_data = pd.read_csv(file, dtype=NAV_SCHEMA, low_memory=False)

        nav_data.to_csv(nav_data_file, index=False)

    required_nav_columns = set(NAV_DATA_SOURCE["keep_columns"])
    missing_nav_columns = required_nav_columns - set(nav_data.columns)
    if missing_nav_columns:
        raise ValueError(
            f"NAV data is missing required columns: {sorted(missing_nav_columns)}"
        )

    required_ret_columns = {VOR_RETENTION_SOURCE["id_column"]}
    missing_ret_columns = required_ret_columns - set(ret_data.columns)
    if missing_ret_columns:
        raise ValueError(
            f"RET data is missing required columns: {sorted(missing_ret_columns)}"
        )

    # Keep NAV facilities whose IDs occur in the VOR retention list.
    valid_ids = set(ret_data[VOR_RETENTION_SOURCE["id_column"]])
    nav_data = nav_data[nav_data[NAV_DATA_SOURCE["id_column"]].isin(valid_ids)]

    # Keep VOR-type facilities only.
    nav_data = nav_data[nav_data["NAV_TYPE"].isin(VALID_NAV_TYPES)].copy()

    # Keep only the columns required downstream.
    nav_data = nav_data.loc[
        :,
        NAV_DATA_SOURCE["keep_columns"],
    ].reset_index(drop=True)

    # Normalize critical numeric fields before caching MON data.
    numeric_columns = [
        "LAT_DECIMAL",
        "LONG_DECIMAL",
        "ELEV",
        "MAG_VARN",
    ]
    nav_data[numeric_columns] = nav_data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return nav_data


def strip_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Locate the RET header using known column names and remove preceding rows."""
    expected_columns = {"ID"}
    header_idx = None

    for idx in range(len(df)):
        values = set(df.iloc[idx].dropna().astype(str).str.strip())
        if expected_columns.issubset(values):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("Could not locate RET header row containing 'ID'")

    new_columns = df.iloc[header_idx].fillna("").astype(str).str.strip().tolist()

    df_cleaned = df.iloc[header_idx + 1 :].copy()
    df_cleaned.columns = new_columns

    return df_cleaned.reset_index(drop=True)


def generate_frd_data(
    nav_set_df: pd.DataFrame,
    step_dist: float,  # nautical miles
    start_dist: float,  # nautical miles
    end_dist: float,
    start_radial: float,  # magnetic degrees
    end_radial: float,  # inclusive
    radial_step: float,
) -> pd.DataFrame:
    """Generate an FRD grid around each VOR in ``nav_set_df``.

    Distances are in nautical miles and radials are magnetic degrees.
    The distance and radial endpoints are inclusive and must fall exactly
    on their respective step sizes.

    At distance zero, only one point is generated and its radial is
    conventionally set to zero.
    """
    validate_grid_parameters(
        step_size=step_dist,
        start_dist=start_dist,
        end_dist=end_dist,
        start_radial=start_radial,
        end_radial=end_radial,
        radial_step=radial_step,
    )
    validate_nav_set(nav_set_df)

    radial_range = make_inclusive_range(
        start=start_radial,
        end=end_radial,
        step=radial_step,
    )

    if start_dist == 0:
        non_zero_dists = make_inclusive_range(
            start=start_dist + step_dist,
            end=end_dist,
            step=step_dist,
        )
        if non_zero_dists.size:
            dist_grid, radial_grid = np.meshgrid(
                non_zero_dists,
                radial_range,
                indexing="xy",
            )
            step_dists = np.hstack(([0.0], dist_grid.ravel()))
            step_radials = np.hstack(([0.0], radial_grid.ravel()))
        else:
            step_dists = np.array([0.0])
            step_radials = np.array([0.0])
    else:
        dist_range = make_inclusive_range(
            start=start_dist,
            end=end_dist,
            step=step_dist,
        )
        dist_grid, radial_grid = np.meshgrid(
            dist_range,
            radial_range,
            indexing="xy",
        )
        step_dists = dist_grid.ravel()
        step_radials = radial_grid.ravel()

    points_per_seed = step_dists.size

    lats = nav_set_df["LAT_DECIMAL"].to_numpy(dtype=float)[:, np.newaxis]
    lons = nav_set_df["LONG_DECIMAL"].to_numpy(dtype=float)[:, np.newaxis]
    mag_vars = nav_set_df["MAG_VARN"].to_numpy(dtype=float)[:, np.newaxis]

    logger.info(
        "Computing %d geodesic points for %d navaids...",
        len(nav_set_df) * points_per_seed,
        len(nav_set_df),
    )
    new_lat, new_lon = calc_frd_to_lat_lon(
        lats=lats,
        lons=lons,
        mag_vars=mag_vars,
        step_radials=step_radials[np.newaxis, :],
        step_dists=step_dists[np.newaxis, :],
    )

    flat_lat = new_lat.ravel()
    flat_lon = new_lon.ravel()
    flat_radial = np.broadcast_to(
        step_radials[np.newaxis, :],
        new_lat.shape,
    ).ravel()
    flat_dist = np.broadcast_to(
        step_dists[np.newaxis, :],
        new_lat.shape,
    ).ravel()

    flat_ids = np.repeat(
        nav_set_df["NAV_ID"].to_numpy(),
        points_per_seed,
    )
    flat_ssvs = np.repeat(
        nav_set_df["ALT_CODE"].to_numpy(),
        points_per_seed,
    )

    elevs = pd.to_numeric(
        nav_set_df["ELEV"],
        errors="coerce",
    ).to_numpy(dtype=float)
    flat_elev = np.repeat(elevs, points_per_seed)

    flat_min_alt, flat_max_alt = calculate_service_volume_altitudes(
        ssvs=flat_ssvs,
        distances=flat_dist,
        elevations=flat_elev,
    )

    frd_df = pd.DataFrame(
        {
            "parent_id": flat_ids,
            "generated_lat": flat_lat,
            "generated_lon": flat_lon,
            "radial": flat_radial,
            "distance": flat_dist,
            "min_alt": flat_min_alt,
            "max_alt": flat_max_alt,
        }
    )

    # Exclude points falling outside the operational service volume limits of each navaid class.
    return frd_df.dropna(subset=["min_alt", "max_alt"]).reset_index(drop=True)


def calc_frd_to_lat_lon(
    lats: np.ndarray,
    lons: np.ndarray,
    mag_vars: np.ndarray,
    step_radials: np.ndarray,
    step_dists: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate WGS84 destination points from VOR positions.

    ``step_radials`` are magnetic bearings. FAA ``MAG_VARN`` is expected to
    use the convention that easterly variation is negative and westerly
    variation is positive, so true bearing is magnetic bearing plus
    magnetic variation.

    Distances are in nautical miles. GeographicLib performs the direct
    geodesic calculation on the WGS84 ellipsoid.
    """
    true_bearing = step_radials + mag_vars

    # GeographicLib accepts degrees and meters. Convert nautical miles to
    # meters using the international nautical mile definition.
    distance_m = step_dists * 1852.0

    # GeographicLib's Python API is scalar-oriented, so calculate each
    # station/grid combination in a vectorized-friendly flattened loop.
    # The resulting arrays are reshaped to the original broadcast shape.
    output_shape = np.broadcast_shapes(
        lats.shape,
        lons.shape,
        true_bearing.shape,
        distance_m.shape,
    )

    lat_values = np.broadcast_to(lats, output_shape).ravel()
    lon_values = np.broadcast_to(lons, output_shape).ravel()
    bearing_values = np.broadcast_to(true_bearing, output_shape).ravel()
    distance_values = np.broadcast_to(distance_m, output_shape).ravel()

    new_lat = np.empty(lat_values.size, dtype=float)
    new_lon = np.empty(lon_values.size, dtype=float)

    total_points = lat_values.size
    log_interval = max(total_points // 100, 1)

    for i, (lat, lon, bearing, distance) in enumerate(
        zip(
            lat_values,
            lon_values,
            bearing_values,
            distance_values,
            strict=True,
        )
    ):
        result = WGS84.Direct(
            lat,
            lon,
            bearing,
            distance,
        )
        new_lat[i] = result["lat2"]
        new_lon[i] = result["lon2"]

        if (i + 1) % log_interval == 0 or i + 1 == total_points:
            logger.info(
                "Computed %d/%d geodesic points (%.1f%%)",
                i + 1,
                total_points,
                100.0 * (i + 1) / total_points,
            )

    return new_lat.reshape(output_shape), new_lon.reshape(output_shape)


def calculate_service_volume_altitudes(
    ssvs: np.ndarray,
    distances: np.ndarray,
    elevations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate continuous SSV altitude limits for each generated point.

    A point at distance D is in every service-volume band for its SSV class
    whose radius is at least D. Therefore, overlapping bands are combined
    using the lowest applicable minimum altitude and highest applicable
    maximum altitude.
    """
    min_alt = np.full(distances.shape, np.nan, dtype=float)
    max_alt = np.full(distances.shape, np.nan, dtype=float)

    for ssv_class in pd.unique(ssvs):
        if pd.isna(ssv_class):
            continue

        class_mask = ssvs == ssv_class
        class_distances = distances[class_mask]

        class_min = np.full(class_distances.shape, np.inf, dtype=float)
        class_max = np.full(class_distances.shape, -np.inf, dtype=float)

        rules = [rule for rule in RAW_SERVICE_VOLUMES_VOR if rule[0] == ssv_class]

        if not rules:
            raise ValueError(
                f"Unsupported service-volume class in ALT_CODE: {ssv_class!r}"
            )

        for _, min_ath, max_ath, max_distance in rules:
            applicable = class_distances <= max_distance
            class_min[applicable] = np.minimum(
                class_min[applicable],
                min_ath,
            )
            class_max[applicable] = np.maximum(
                class_max[applicable],
                max_ath,
            )

        out_of_bounds = np.isinf(class_min)
        class_min[out_of_bounds] = np.nan
        class_max[out_of_bounds] = np.nan

        min_alt[class_mask] = class_min + elevations[class_mask]
        max_alt[class_mask] = class_max + elevations[class_mask]

    return min_alt, max_alt


def validate_grid_parameters(
    *,
    step_size: float,
    start_dist: float,
    end_dist: float,
    start_radial: float,
    end_radial: float,
    radial_step: float,
) -> None:
    """Validate grid parameters and enforce inclusive, step-aligned endpoints."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    if radial_step <= 0:
        raise ValueError("radial_step must be positive")

    if start_dist < 0:
        raise ValueError("start_dist must not be negative")

    if start_dist > end_dist:
        raise ValueError("start_dist must not exceed end_dist")

    if start_radial > end_radial:
        raise ValueError("start_radial must not exceed end_radial")

    if not is_step_aligned(start_dist, end_dist, step_size):
        raise ValueError(
            "end_dist must be reachable from start_dist using step_size; "
            "the endpoint is inclusive and cannot be silently overshot"
        )

    if not is_step_aligned(start_radial, end_radial, radial_step):
        raise ValueError(
            "end_radial must be reachable from start_radial using radial_step; "
            "the endpoint is inclusive and cannot be silently overshot"
        )


def is_step_aligned(start: float, end: float, step: float) -> bool:
    """Return whether ``end`` is an integer number of steps from ``start``."""
    steps = (end - start) / step
    return bool(np.isclose(steps, round(steps), rtol=0.0, atol=1e-10))


def make_inclusive_range(
    *,
    start: float,
    end: float,
    step: float,
) -> np.ndarray:
    """Return a fixed-step range whose endpoint is guaranteed inclusive."""
    count = int(round((end - start) / step))
    return start + np.arange(count + 1, dtype=float) * step


if __name__ == "__main__":
    main()
