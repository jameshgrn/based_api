"""
Fetch USGS annual peak flow records for IFMHA stations via NWIS,
fit Log-Pearson III (Bulletin 17C), and return bankfull discharge
at a specified return period.

Output: data/peak_flows_raw.parquet   (cached raw peaks)
        data/qbf_lp3.parquet          (Qbf per station)

Units: peak_va in cfs → converted to m3/s
"""

import time
import io
import numpy as np
import pandas as pd
import requests
from scipy import stats

# ── config ───────────────────────────────────────────────────────────────────
RETURN_PERIOD = 2.0        # years (1.5 or 2.0 are both defensible)
MIN_PEAK_YEARS = 10        # minimum annual peaks for reliable LP3
BATCH_SIZE = 50            # sites per NWIS request
REQUEST_DELAY = 0.5        # seconds between batches (be a good citizen)
CFS_TO_M3S = 0.0283168

NWIS_PEAK_URL = "https://nwis.waterdata.usgs.gov/nwis/peak"

RAW_CACHE = "data/peak_flows_raw.parquet"
QBF_OUT = "data/qbf_lp3.parquet"


# ── NWIS fetch ────────────────────────────────────────────────────────────────

def fetch_peaks_batch(site_nos: list[str]) -> pd.DataFrame:
    """Fetch annual peak flows for a batch of sites. Returns tidy DataFrame."""
    params = {
        "site_no": ",".join(site_nos),
        "format": "rdb",
        "agency_cd": "USGS",
    }
    try:
        r = requests.get(NWIS_PEAK_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()

    # skip comment/header lines (start with #)
    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()

    # RDB: first non-comment line is header, second is format widths, rest is data
    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            sep="\t",
            skiprows=[1],  # skip the format-width line
            dtype=str,
        )
    except Exception:
        return pd.DataFrame()

    if "site_no" not in df.columns or "peak_va" not in df.columns:
        return pd.DataFrame()

    df = df[["site_no", "peak_dt", "peak_va"]].copy()
    df["peak_va"] = pd.to_numeric(df["peak_va"], errors="coerce")
    df = df.dropna(subset=["peak_va"])
    df = df[df["peak_va"] > 0]
    df["peak_m3s"] = df["peak_va"].astype(float) * CFS_TO_M3S
    return df[["site_no", "peak_dt", "peak_m3s"]]


def fetch_all_peaks(site_nos: list[str], cache_path: str = RAW_CACHE) -> pd.DataFrame:
    """Fetch peaks for all stations, using cache if available."""
    try:
        cached = pd.read_parquet(cache_path)
        cached_sites = set(cached["site_no"].unique())
        missing = [s for s in site_nos if s not in cached_sites]
        if not missing:
            print(f"  Loaded {len(cached):,} peak records from cache ({len(cached_sites):,} stations)")
            return cached
        print(f"  Cache has {len(cached_sites):,} stations; fetching {len(missing):,} missing...")
        site_nos = missing
        existing = cached
    except FileNotFoundError:
        existing = pd.DataFrame()

    batches = [site_nos[i:i + BATCH_SIZE] for i in range(0, len(site_nos), BATCH_SIZE)]
    results = []
    for i, batch in enumerate(batches):
        df = fetch_peaks_batch(batch)
        if not df.empty:
            results.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(batches)} batches fetched...")
        time.sleep(REQUEST_DELAY)

    new_data = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    all_data = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data

    if not all_data.empty:
        all_data.to_parquet(cache_path, index=False)
    print(f"  Fetched {len(new_data):,} new peak records; total {len(all_data):,} saved to cache")
    return all_data


# ── Log-Pearson III ──────────────────────────────────────────────────────────

def lp3_quantile(peak_flows_m3s: np.ndarray, return_period: float) -> float:
    """
    Fit Log-Pearson III (Bulletin 17C) to annual peaks and return the
    discharge at the given return period (years).

    Non-exceedance probability = 1 - 1/T
    Uses method of moments: mean, std, skew of log10(Q).
    """
    log_q = np.log10(peak_flows_m3s)
    mean_l = np.mean(log_q)
    std_l = np.std(log_q, ddof=1)
    skew_l = stats.skew(log_q)

    prob = 1.0 - 1.0 / return_period
    log_q_t = stats.pearson3.ppf(prob, skew=skew_l, loc=mean_l, scale=std_l)
    return float(10 ** log_q_t)


def compute_qbf(peaks: pd.DataFrame, return_period: float = RETURN_PERIOD,
                min_years: int = MIN_PEAK_YEARS) -> pd.DataFrame:
    """Fit LP3 per station and return Qbf lookup table."""
    records = []
    for site_no, grp in peaks.groupby("site_no"):
        q = grp["peak_m3s"].dropna().values
        if len(q) < min_years:
            continue
        try:
            qbf = lp3_quantile(q, return_period)
        except Exception:
            continue
        if qbf <= 0 or not np.isfinite(qbf):
            continue
        records.append({
            "site_no": site_no,
            "qbf_m3s": qbf,
            "n_peaks": len(q),
            "return_period_yr": return_period,
        })
    return pd.DataFrame(records)


# ── main ─────────────────────────────────────────────────────────────────────

def main(
    ifmha_path: str = "data/IFMHA_dataset.csv",
    qbf_out: str = QBF_OUT,
):
    print("Reading IFMHA station list...")
    sites_raw = pd.read_csv(ifmha_path, usecols=["site_no"], low_memory=False)[
        "site_no"
    ].unique()
    site_nos = [str(int(s)).zfill(8) for s in sites_raw]
    print(f"  {len(site_nos):,} unique stations")

    print("\nFetching USGS annual peak flows...")
    peaks = fetch_all_peaks(site_nos)

    if peaks.empty:
        raise RuntimeError("No peak flow data fetched — check network/NWIS availability")

    print(f"\nFitting Log-Pearson III (T = {RETURN_PERIOD} yr, min {MIN_PEAK_YEARS} peaks)...")
    qbf = compute_qbf(peaks, return_period=RETURN_PERIOD, min_years=MIN_PEAK_YEARS)
    print(f"  {len(qbf):,} stations with valid Qbf estimates")
    print(f"  Qbf range: [{qbf['qbf_m3s'].min():.3f}, {qbf['qbf_m3s'].max():.1f}] m3/s")
    print(f"  Qbf median: {qbf['qbf_m3s'].median():.2f} m3/s")

    qbf.to_parquet(qbf_out, index=False)
    print(f"\nSaved → {qbf_out}")
    return qbf


if __name__ == "__main__":
    main()
