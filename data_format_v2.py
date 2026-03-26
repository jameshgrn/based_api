"""
BASED v2 training data pipeline.

Extracts bankfull hydraulic geometry from IFMHA (USGS wading measurements)
using LP3-derived Qbf (2-year return period from USGS annual peak flows)
and merges with existing clean dataset without overlap.

IFMHA units (imperial) → converted to SI:
  discharge: cfs  → m3/s  (× 0.0283168)
  width:     ft   → m     (× 0.3048)
  area:      ft2  → m2    (× 0.0929030)
  depth:     ft   → m     (computed as area/width, then × 0.3048)
  slope:     m/m  (NHD, already dimensionless)
"""

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
CFS_TO_M3S = 0.0283168
FT_TO_M = 0.3048
FT2_TO_M2 = 0.3048 ** 2

QBF_WINDOW = 0.10       # ±10% of Qbf for geometry extraction
MIN_BANKFULL_MEAS = 5   # minimum near-bankfull measurements for geometry


# ── loaders ──────────────────────────────────────────────────────────────────

def load_ifmha(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # keep only rows with discharge
    df = df[df["discharge_va"] > 0].copy()

    # unit conversion for discharge (always present)
    df["Q_m3s"] = df["discharge_va"] * CFS_TO_M3S

    # geometry columns (may be null for many rows)
    geom_mask = (
        df["chan_area"].notna() &
        df["chan_width"].notna() &
        (df["chan_area"] > 0) &
        (df["chan_width"] > 0)
    )
    df.loc[geom_mask, "width_m"] = df.loc[geom_mask, "chan_width"] * FT_TO_M
    df.loc[geom_mask, "area_m2"] = df.loc[geom_mask, "chan_area"] * FT2_TO_M2
    df.loc[geom_mask, "depth_m"] = (
        df.loc[geom_mask, "area_m2"] / df.loc[geom_mask, "width_m"]
    )

    # slope already m/m from NHD
    df["slope"] = df["SLOPE"]

    return df


def extract_bankfull(df: pd.DataFrame, qbf_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Per station:
      1. Look up LP3-derived Qbf (2-yr return period from USGS peak flows).
      2. Select IFMHA measurements within QBF_WINDOW of Qbf that have geometry.
      3. Take median W, H; use Qbf; keep median slope.
    """
    # normalise site_no to zero-padded string for join
    qbf_map = qbf_lookup.set_index("site_no")["qbf_m3s"].to_dict()

    results = []
    for site_no, grp in df.groupby("site_no"):
        site_str = str(int(site_no)).zfill(8)

        qbf = qbf_map.get(site_str)
        if qbf is None or qbf <= 0:
            continue

        slope = grp["slope"].dropna()
        if slope.empty or slope.median() < 1e-5:
            continue

        lo, hi = qbf * (1 - QBF_WINDOW), qbf * (1 + QBF_WINDOW)
        near_bf = grp[
            (grp["Q_m3s"] >= lo) &
            (grp["Q_m3s"] <= hi) &
            grp["width_m"].notna() &
            grp["depth_m"].notna()
        ]

        if len(near_bf) < MIN_BANKFULL_MEAS:
            continue

        wbf = near_bf["width_m"].median()
        hbf = near_bf["depth_m"].median()
        da = grp["DASqKm"].dropna().median()

        if wbf <= 0 or hbf <= 0:
            continue

        results.append({
            "site_id": site_str,
            "source": "IFMHA",
            "discharge": qbf,
            "width": wbf,
            "depth": hbf,
            "slope": slope.median(),
            "drainage_area_km2": da if pd.notna(da) else np.nan,
        })

    return pd.DataFrame(results)


def apply_physical_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Same physical plausibility bounds used in data_format.py v1."""
    numeric = ["discharge", "width", "depth", "slope"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    valid = (
        df[numeric].notna().all(axis=1) &
        (df[numeric] > 0).all(axis=1) &
        (df["slope"] >= 1e-5) &
        (df["width"] <= 4000) &
        (df["width"] / df["depth"] >= 3) &
        (df["width"] / df["depth"] <= 7000) &
        (df["depth"] <= 30) &   # IFMHA-specific: wading measurements cap
        (df["slope"] <= 0.20)   # NHD slope artifacts above this are implausible
    )
    return df[valid].copy()


# ── main ─────────────────────────────────────────────────────────────────────

def generate_data_v2(
    ifmha_path: str = "data/IFMHA_dataset.csv",
    qbf_path: str = "data/qbf_lp3.parquet",
    existing_path: str = "data/based_input_data_clean.csv",
    output_path: str = "data/based_input_data_v2.csv",
) -> pd.DataFrame:
    from fetch_peak_flows import main as fetch_qbf

    print("Loading IFMHA...")
    ifmha_raw = load_ifmha(ifmha_path)
    print(f"  {len(ifmha_raw):,} rows with Q>0 across {ifmha_raw['site_no'].nunique():,} stations")

    # fetch/load LP3 Qbf estimates
    try:
        qbf_lookup = pd.read_parquet(qbf_path)
        print(f"  Loaded LP3 Qbf for {len(qbf_lookup):,} stations from cache")
    except FileNotFoundError:
        print("  No Qbf cache found — fetching USGS peak flows...")
        qbf_lookup = fetch_qbf(ifmha_path=ifmha_path, qbf_out=qbf_path)

    print("Extracting bankfull geometry...")
    bankfull = extract_bankfull(ifmha_raw, qbf_lookup)
    print(f"  {len(bankfull):,} stations with bankfull geometry before filtering")

    print("Applying physical filters...")
    bankfull = apply_physical_filters(bankfull)
    print(f"  {len(bankfull):,} stations after filtering")

    print("Loading existing clean dataset...")
    existing = pd.read_csv(existing_path)
    keep = ["discharge", "width", "depth", "slope", "site_id", "source"]
    existing = existing[keep]
    print(f"  {len(existing):,} rows from existing dataset ({existing['source'].value_counts().to_dict()})")

    # stack — no site_id overlap possible (literature vs USGS gauges)
    combined = pd.concat([existing, bankfull[keep]], axis=0, ignore_index=True)

    # true duplicate rows only: all numeric values identical across all four hydraulic columns
    before = len(combined)
    combined = combined.drop_duplicates(subset=["discharge", "width", "depth", "slope"])
    dropped = before - len(combined)
    if dropped:
        print(f"  Dropped {dropped} exact duplicate rows")

    print(f"\nCombined dataset: {len(combined):,} rows")
    print(combined["source"].value_counts().to_string())

    print("\nVariable ranges:")
    for col in ["discharge", "width", "depth", "slope"]:
        s = combined[col]
        print(f"  {col}: [{s.min():.4g}, {s.max():.4g}]  median={s.median():.4g}")

    combined.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    return combined


if __name__ == "__main__":
    generate_data_v2()
