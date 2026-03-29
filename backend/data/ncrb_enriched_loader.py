"""
NCRB Multi-Source Enriched Loader
===================================
Uses ALL available NCRB datasets to build the richest possible
district-level crime intelligence for 5 pilot cities.

Sources used:
  01  - District IPC crimes (base)
  02_01 - District crimes against SC
  02    - District crimes against ST (Scheduled Tribes)
  03  - District crimes against children
  12  - Police strength (actual vs sanctioned)
  10  - Property stolen & recovered (value)
  30  - Auto theft by vehicle type
  17  - Crime by place of occurrence (residential/highway/market)
  19  - Murder motives (gang/property/domestic)
  25  - Complaints against police (accountability)
  42  - District crimes against women
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.ncrb_loader import (
    CITY_DISTRICTS,
    DISTRICT_CENTROIDS,
    ZONES_PER_DISTRICT,
    HOUR_WEIGHTS,
    CITY_TEMP,
    MONSOON_MONTHS,
    load_ncrb_raw,
    filter_city_districts,
    _sample_hour,
    _sample_temp,
    _sample_precip,
)

RAW_DIR = Path("backend/data/raw/ncrb")

# ---------------------------------------------------------------------------
# Crime type mapping — extended for all sources
# ---------------------------------------------------------------------------

EXTENDED_CRIME_TYPES = [
    "vehicle_theft",
    "burglary",
    "robbery",
    "pickpocketing",
    "assault",
    "dacoity",
    "cyber_fraud",
    "domestic_violence",
    "sexual_assault",
    "property_crime",
    "child_crime",
    "other",
]


# ---------------------------------------------------------------------------
# Helper: normalise state/district names across files
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return str(s).upper().strip()


def _safe_load(path: Path, enc: str = "utf-8", nrows: int | None = None) -> pd.DataFrame | None:
    for e in (enc, "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=e, low_memory=False, nrows=nrows)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None


def _find_file(*patterns: str) -> Path | None:
    """Find the first file matching any of the patterns (recursively)."""
    for pattern in patterns:
        hits = list(RAW_DIR.rglob(pattern))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------------------
# 1. Crimes against women (district-level)
# ---------------------------------------------------------------------------

def load_women_crimes() -> pd.DataFrame:
    """Load district-wise crimes against women 2001-2014. Returns tidy DataFrame."""
    files = sorted(RAW_DIR.rglob("42_District_wise_crimes_committed_against_women_*.csv"))
    dfs = []
    for f in set(files):
        df = _safe_load(f)
        if df is None:
            continue
        df.columns = [c.strip() for c in df.columns]
        # Normalise column names
        rename = {}
        for col in df.columns:
            c = col.strip().lower()
            if re.search(r"state|area", c):
                rename[col] = "state"
            elif re.search(r"^district", c):
                rename[col] = "district"
            elif re.search(r"year", c):
                rename[col] = "year"
            elif re.search(r"rape", c) and "kidnap" not in c:
                rename[col] = "w_rape"
            elif re.search(r"dowry death", c):
                rename[col] = "w_dowry_deaths"
            elif re.search(r"assault.*modesty|outrage.*modesty", c):
                rename[col] = "w_assault"
            elif re.search(r"cruelty.*husband|domestic", c):
                rename[col] = "w_domestic_cruelty"
            elif re.search(r"kidnapping|abduction", c):
                rename[col] = "w_kidnapping"
            elif re.search(r"insult.*modesty", c):
                rename[col] = "w_insult"
        df = df.rename(columns=rename)
        req = {"state", "district", "year"}
        if not req.issubset(set(df.columns)):
            continue
        df["state"] = df["state"].apply(_norm)
        df["district"] = df["district"].apply(_norm)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        # Deduplicate renamed columns — keep first occurrence only
        seen_cols: dict[str, int] = {}
        dedup_cols = []
        for c in df.columns:
            if c in seen_cols:
                seen_cols[c] += 1
                dedup_cols.append(f"{c}_dup{seen_cols[c]}")
            else:
                seen_cols[c] = 0
                dedup_cols.append(c)
        df.columns = dedup_cols
        # Drop the _dup versions; keep only the first (most aggregate) column
        df = df[[c for c in df.columns if "_dup" not in c]]

        for col in [c for c in df.columns if c.startswith("w_")]:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]  # fallback: take first column
            df[col] = pd.to_numeric(s.astype(str), errors="coerce").fillna(0)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["state", "district", "year"])
    print(f"  Women crimes: {len(combined):,} district-year rows")
    return combined


# ---------------------------------------------------------------------------
# 2. Crimes against SC (district-level)
# ---------------------------------------------------------------------------

def load_sc_crimes() -> pd.DataFrame:
    files = sorted(RAW_DIR.rglob("02_01_District_wise_crimes_committed_against_SC_*.csv"))
    dfs = []
    for f in set(files):
        df = _safe_load(f)
        if df is None:
            continue
        rename = {}
        for col in df.columns:
            c = col.strip().lower()
            if re.search(r"state|area", c):    rename[col] = "state"
            elif re.search(r"^district", c):   rename[col] = "district"
            elif re.search(r"year", c):        rename[col] = "year"
            elif re.search(r"^murder$", c):    rename[col] = "sc_murder"
            elif re.search(r"^rape$", c):      rename[col] = "sc_rape"
            elif re.search(r"^robbery$", c):   rename[col] = "sc_robbery"
            elif re.search(r"^hurt$", c):      rename[col] = "sc_hurt"
            elif re.search(r"poa|prevention.*atrociti", c): rename[col] = "sc_poa"
        df = df.rename(columns=rename)
        if not {"state", "district", "year"}.issubset(df.columns):
            continue
        # Deduplicate columns
        seen: dict[str, int] = {}
        dedup = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1; dedup.append(f"{c}_dup{seen[c]}")
            else:
                seen[c] = 0; dedup.append(c)
        df.columns = dedup
        df = df[[c for c in df.columns if "_dup" not in c]]

        df["state"] = df["state"].apply(_norm)
        df["district"] = df["district"].apply(_norm)
        df["year"] = pd.to_numeric(df["year"].astype(str), errors="coerce").fillna(0).astype(int)
        for col in [c for c in df.columns if c.startswith("sc_")]:
            s = df[col]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[col] = pd.to_numeric(s.astype(str), errors="coerce").fillna(0)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["state", "district", "year"])
    print(f"  SC crimes: {len(combined):,} district-year rows")
    return combined


# ---------------------------------------------------------------------------
# 3. Crimes against children (district-level)
# ---------------------------------------------------------------------------

def load_children_crimes() -> pd.DataFrame:
    files = sorted(RAW_DIR.rglob("03_District_wise_crimes_committed_against_children_*.csv"))
    dfs = []
    for f in set(files):
        df = _safe_load(f)
        if df is None:
            continue
        rename = {}
        for col in df.columns:
            c = col.strip().lower()
            if re.search(r"state|area", c):    rename[col] = "state"
            elif re.search(r"^district$", c):  rename[col] = "district"
            elif re.search(r"^year$", c):      rename[col] = "year"
            elif re.search(r"^total$", c):     rename[col] = "child_total"
            elif re.search(r"^murder$", c):    rename[col] = "child_murder"
            elif re.search(r"^rape$", c):      rename[col] = "child_rape"
            elif re.search(r"kidnap|abduct", c): rename[col] = "child_kidnap"
        df = df.rename(columns=rename)
        if not {"state", "district", "year"}.issubset(df.columns):
            continue
        # Deduplicate columns
        seen: dict[str, int] = {}
        dedup = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1; dedup.append(f"{c}_dup{seen[c]}")
            else:
                seen[c] = 0; dedup.append(c)
        df.columns = dedup
        df = df[[c for c in df.columns if "_dup" not in c]]

        df["state"] = df["state"].apply(_norm)
        df["district"] = df["district"].apply(_norm)
        df["year"] = pd.to_numeric(df["year"].astype(str), errors="coerce").fillna(0).astype(int)
        for col in [c for c in df.columns if c.startswith("child_")]:
            s = df[col]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[col] = pd.to_numeric(s.astype(str), errors="coerce").fillna(0)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["state", "district", "year"])
    print(f"  Children crimes: {len(combined):,} district-year rows")
    return combined


# ---------------------------------------------------------------------------
# 4. Police strength (state-level)
# ---------------------------------------------------------------------------

def load_police_strength() -> pd.DataFrame:
    """
    Returns per-state, per-year police coverage ratio:
    actual_strength / sanctioned_strength (< 1 = understaffed).
    """
    f = _find_file("12_Police_strength_actual_and_sanctioned.csv")
    if not f:
        return pd.DataFrame()

    df = _safe_load(f)
    if df is None:
        return pd.DataFrame()

    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if re.search(r"area_name|state", c):  rename[col] = "state"
        elif re.search(r"^year$", c):         rename[col] = "year"
        elif re.search(r"group_name", c):     rename[col] = "group"
        elif re.search(r"sub_group", c):      rename[col] = "subgroup"
        elif re.search(r"all_rank.*total|rank_all_rank", c): rename[col] = "count"
    df = df.rename(columns=rename)

    if "state" not in df.columns or "year" not in df.columns:
        return pd.DataFrame()

    df["state"] = df["state"].apply(_norm)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["count"] = pd.to_numeric(df.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Separate actual vs sanctioned
    if "subgroup" in df.columns:
        df["subgroup"] = df["subgroup"].astype(str).str.upper()
        actual = df[df["subgroup"].str.contains("ACTUAL", na=False)].groupby(["state", "year"])["count"].sum()
        sanctioned = df[df["subgroup"].str.contains("SANCTIONED", na=False)].groupby(["state", "year"])["count"].sum()
        ratio_df = (actual / sanctioned.replace(0, np.nan)).reset_index()
        ratio_df.columns = ["state", "year", "police_coverage_ratio"]
    else:
        # Just use total as proxy
        ratio_df = df.groupby(["state", "year"])["count"].sum().reset_index()
        ratio_df.columns = ["state", "year", "police_coverage_ratio"]
        ratio_df["police_coverage_ratio"] = ratio_df["police_coverage_ratio"] / ratio_df["police_coverage_ratio"].max()

    ratio_df["police_coverage_ratio"] = ratio_df["police_coverage_ratio"].clip(0.3, 1.2)
    print(f"  Police strength: {len(ratio_df):,} state-year rows")
    return ratio_df


# ---------------------------------------------------------------------------
# 5. Property stolen value (state-level)
# ---------------------------------------------------------------------------

def load_property_value() -> pd.DataFrame:
    f = _find_file("10_Property_stolen_and_recovered.csv")
    if not f:
        return pd.DataFrame()
    df = _safe_load(f)
    if df is None:
        return pd.DataFrame()

    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if re.search(r"area_name|state", c):     rename[col] = "state"
        elif re.search(r"^year$", c):            rename[col] = "year"
        elif re.search(r"value.*stolen", c):     rename[col] = "value_stolen"
        elif re.search(r"cases.*stolen", c):     rename[col] = "cases_stolen"
    df = df.rename(columns=rename)

    if "state" not in df.columns:
        return pd.DataFrame()

    df["state"] = df["state"].apply(_norm)
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)
    for col in ["value_stolen", "cases_stolen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "value_stolen" in df.columns:
        agg = df.groupby(["state", "year"])["value_stolen"].sum().reset_index()
        agg.columns = ["state", "year", "property_value_stolen_lakh"]
        print(f"  Property value: {len(agg):,} state-year rows")
        return agg
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. Auto theft detail (state-level)
# ---------------------------------------------------------------------------

def load_auto_theft_detail() -> pd.DataFrame:
    f = _find_file("30_Auto_theft.csv")
    if not f:
        return pd.DataFrame()
    df = _safe_load(f)
    if df is None:
        return pd.DataFrame()

    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if re.search(r"area_name|state", c):    rename[col] = "state"
        elif re.search(r"^year$", c):           rename[col] = "year"
        elif re.search(r"stolen", c):           rename[col] = "auto_stolen"
        elif re.search(r"recovered", c):        rename[col] = "auto_recovered"
    df = df.rename(columns=rename)

    if "state" not in df.columns:
        return pd.DataFrame()

    df["state"] = df["state"].apply(_norm)
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)
    for col in ["auto_stolen", "auto_recovered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "auto_stolen" in df.columns:
        agg = df.groupby(["state", "year"])["auto_stolen"].sum().reset_index()
        agg.columns = ["state", "year", "state_auto_theft_count"]
        # Normalize: recovery rate as confidence proxy
        if "auto_recovered" in df.columns:
            rec = df.groupby(["state", "year"])["auto_recovered"].sum()
            sto = df.groupby(["state", "year"])["auto_stolen"].sum()
            recovery = (rec / sto.replace(0, np.nan)).clip(0, 1).reset_index()
            recovery.columns = ["state", "year", "auto_recovery_rate"]
            agg = agg.merge(recovery, on=["state", "year"], how="left")
        print(f"  Auto theft detail: {len(agg):,} state-year rows")
        return agg
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 7. ST (Scheduled Tribe) crimes — district-level (Table 02)
# ---------------------------------------------------------------------------

def load_st_crimes() -> pd.DataFrame:
    """District-wise crimes against Scheduled Tribes 2001-2014."""
    files = sorted(RAW_DIR.rglob("02_District_wise_crimes_committed_against_ST_*.csv"))
    dfs = []
    for f in set(files):
        df = _safe_load(f)
        if df is None:
            continue
        rename = {}
        for col in df.columns:
            c = col.strip().lower()
            if re.search(r"state|area", c):     rename[col] = "state"
            elif re.search(r"^district$", c):   rename[col] = "district"
            elif re.search(r"^year$", c):       rename[col] = "year"
            elif re.search(r"^murder$", c):     rename[col] = "st_murder"
            elif re.search(r"^rape$", c):       rename[col] = "st_rape"
            elif re.search(r"^robbery$", c):    rename[col] = "st_robbery"
            elif re.search(r"^hurt$", c):       rename[col] = "st_hurt"
            elif re.search(r"^arson$", c):      rename[col] = "st_arson"
        df = df.rename(columns=rename)
        if not {"state", "district", "year"}.issubset(df.columns):
            continue
        # Deduplicate columns
        seen: dict[str, int] = {}
        dedup = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1; dedup.append(f"{c}_dup{seen[c]}")
            else:
                seen[c] = 0; dedup.append(c)
        df.columns = dedup
        df = df[[c for c in df.columns if "_dup" not in c]]
        df["state"] = df["state"].apply(_norm)
        df["district"] = df["district"].apply(_norm)
        df["year"] = pd.to_numeric(df["year"].astype(str), errors="coerce").fillna(0).astype(int)
        for col in [c for c in df.columns if c.startswith("st_")]:
            s = df[col]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            df[col] = pd.to_numeric(s.astype(str), errors="coerce").fillna(0)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["state", "district", "year"])
    print(f"  ST crimes: {len(combined):,} district-year rows")
    return combined


# ---------------------------------------------------------------------------
# 8. Crime by place of occurrence (Table 17) — state-level
#    Tells us what fraction of crimes occur at residential / highway / market
# ---------------------------------------------------------------------------

def load_crime_by_place() -> pd.DataFrame:
    """
    Returns per-state, per-year fractions:
      residential_crime_pct, highway_crime_pct, market_crime_pct
    These weight zone_type crime probabilities.
    """
    files = sorted(RAW_DIR.rglob("17_Crime_by_place_of_occurrence_*.csv"))
    dfs = []
    for f in set(files):
        df = _safe_load(f)
        if df is None:
            continue
        df.columns = [c.strip() for c in df.columns]
        # Identify state and year columns
        state_col = next((c for c in df.columns if re.search(r"state|area", c.lower())), None)
        year_col  = next((c for c in df.columns if re.search(r"^year$", c.lower())), None)
        if not state_col or not year_col:
            continue

        df = df.rename(columns={state_col: "state", year_col: "year"})
        df["state"] = df["state"].apply(_norm)
        df["year"] = pd.to_numeric(df["year"].astype(str), errors="coerce").fillna(0).astype(int)

        # Sum counts for each place category
        res_cols = [c for c in df.columns if "RESIDENTIAL" in c.upper()]
        hwy_cols = [c for c in df.columns if "HIGHWAY" in c.upper()]
        mkt_cols = [c for c in df.columns if re.search(r"COMMERCIAL|MARKET|SHOP", c.upper())]

        for cols, name in [(res_cols, "residential_count"), (hwy_cols, "highway_count"), (mkt_cols, "market_count")]:
            if cols:
                df[name] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            else:
                df[name] = 0

        df["place_total"] = df[["residential_count", "highway_count", "market_count"]].sum(axis=1)
        dfs.append(df[["state", "year", "residential_count", "highway_count", "market_count", "place_total"]])

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby(["state", "year"])[["residential_count", "highway_count", "market_count", "place_total"]].sum().reset_index()
    agg["residential_crime_pct"] = (agg["residential_count"] / agg["place_total"].replace(0, np.nan)).fillna(0.33).clip(0, 1)
    agg["highway_crime_pct"]     = (agg["highway_count"]     / agg["place_total"].replace(0, np.nan)).fillna(0.15).clip(0, 1)
    agg["market_crime_pct"]      = (agg["market_count"]      / agg["place_total"].replace(0, np.nan)).fillna(0.20).clip(0, 1)
    print(f"  Crime by place: {len(agg):,} state-year rows")
    return agg[["state", "year", "residential_crime_pct", "highway_crime_pct", "market_crime_pct"]]


# ---------------------------------------------------------------------------
# 9. Murder motives (Table 19) — state-level
#    gang_murder_pct (gain/terrorist), domestic_murder_pct (dowry/love/lunacy)
# ---------------------------------------------------------------------------

def load_murder_motives() -> pd.DataFrame:
    f = _find_file("19_Motive_or_cause_of_murder_and_culpable_homicide_not_amounting_to_murder.csv")
    if not f:
        return pd.DataFrame()
    df = _safe_load(f)
    if df is None:
        return pd.DataFrame()

    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if re.search(r"area_name|state", c):    rename[col] = "state"
        elif re.search(r"^year$", c):           rename[col] = "year"
    df = df.rename(columns=rename)
    if "state" not in df.columns:
        return pd.DataFrame()

    df["state"] = df["state"].apply(_norm)
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)

    # Gang/property motive: Gain + TerroristExtremist + ClassConflict
    gang_cols = [c for c in df.columns if re.search(r"Gain|Terrorist|Class_Conflict|Political", c)]
    # Domestic motive: Dowry + Love + Lunacy + Communalism
    domestic_cols = [c for c in df.columns if re.search(r"Dowry|Love_Affair|Lunacy|Casteism|Communal", c)]

    for cols, name in [(gang_cols, "gang_motive_count"), (domestic_cols, "domestic_motive_count")]:
        if cols:
            df[name] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        else:
            df[name] = 0

    all_motive_cols = [c for c in df.columns if c.startswith("CHNAMurder")]
    if all_motive_cols:
        df["total_motive_count"] = df[all_motive_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        df["total_motive_count"] = df[["gang_motive_count", "domestic_motive_count"]].sum(axis=1)

    agg = df.groupby(["state", "year"])[["gang_motive_count", "domestic_motive_count", "total_motive_count"]].sum().reset_index()
    agg["gang_murder_pct"]     = (agg["gang_motive_count"]     / agg["total_motive_count"].replace(0, np.nan)).fillna(0.1).clip(0, 1)
    agg["domestic_murder_pct"] = (agg["domestic_motive_count"] / agg["total_motive_count"].replace(0, np.nan)).fillna(0.2).clip(0, 1)
    print(f"  Murder motives: {len(agg):,} state-year rows")
    return agg[["state", "year", "gang_murder_pct", "domestic_murder_pct"]]


# ---------------------------------------------------------------------------
# 10. Complaints against police (Table 25) — state-level
#     complaint_rate = complaints_received / police_strength (normalized)
# ---------------------------------------------------------------------------

def load_police_complaints() -> pd.DataFrame:
    f = _find_file("25_Complaints_against_police.csv")
    if not f:
        return pd.DataFrame()
    df = _safe_load(f)
    if df is None:
        return pd.DataFrame()

    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if re.search(r"area_name|state", c):      rename[col] = "state"
        elif re.search(r"^year$", c):             rename[col] = "year"
        elif re.search(r"complaint.*receiv|alleged", c): rename[col] = "complaints_received"
        elif re.search(r"cases_register", c):     rename[col] = "cases_registered"
    df = df.rename(columns=rename)
    if "state" not in df.columns:
        return pd.DataFrame()

    df["state"] = df["state"].apply(_norm)
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)
    for col in ["complaints_received", "cases_registered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str), errors="coerce").fillna(0)

    if "complaints_received" in df.columns:
        agg = df.groupby(["state", "year"])["complaints_received"].sum().reset_index()
        agg.columns = ["state", "year", "police_complaints_total"]
        # Normalize 0-1 across all states per year
        agg["police_complaint_rate"] = agg.groupby("year")["police_complaints_total"].transform(
            lambda x: x / max(x.max(), 1)
        ).clip(0, 1)
        print(f"  Police complaints: {len(agg):,} state-year rows")
        return agg[["state", "year", "police_complaint_rate"]]
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# State name normalisation for joining
# ---------------------------------------------------------------------------

STATE_ALIASES: dict[str, str] = {
    "KARNATAKA": "KARNATAKA",
    "ANDHRA PRADESH": "ANDHRA PRADESH",
    "TELANGANA": "ANDHRA PRADESH",  # Telangana split from AP in 2014
    "MAHARASHTRA": "MAHARASHTRA",
    "DELHI UT": "DELHI",
    "NCT OF DELHI": "DELHI",
    "DELHI": "DELHI",
    "TAMIL NADU": "TAMIL NADU",
}

CITY_TO_STATE: dict[str, str] = {
    "Bengaluru": "KARNATAKA",
    "Hyderabad": "ANDHRA PRADESH",
    "Mumbai": "MAHARASHTRA",
    "Delhi": "DELHI",
    "Chennai": "TAMIL NADU",
}


def _resolve_state(raw_state: str) -> str:
    s = _norm(raw_state)
    return STATE_ALIASES.get(s, s)


# ---------------------------------------------------------------------------
# Build enriched district crime profile
# ---------------------------------------------------------------------------

def build_enriched_district_profile(
    ipc_df: pd.DataFrame,
    women_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    st_df: pd.DataFrame,
    children_df: pd.DataFrame,
    police_df: pd.DataFrame,
    property_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    place_df: pd.DataFrame,
    motives_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    """
    Merge all crime dimensions into a single district-year profile.
    Returns enriched IPC DataFrame with additional feature columns.
    """
    city_state = CITY_TO_STATE.get(city, "")

    profile = ipc_df.copy()
    profile["state_norm"] = profile["state"].apply(_resolve_state)

    # --- Join women crimes ---
    if not women_df.empty:
        w_city = women_df[women_df["district"].isin(
            {d.upper() for d in CITY_DISTRICTS.get(city, [])}
        )].copy()
        if not w_city.empty:
            w_cols = [c for c in w_city.columns if c.startswith("w_")]
            w_agg = w_city.groupby(["district", "year"])[w_cols].sum().reset_index()
            profile = profile.merge(w_agg, on=["district", "year"], how="left")
            for c in w_cols:
                profile[c] = profile[c].fillna(0)
            # Derived: women safety index
            profile["women_safety_index"] = (
                profile.get("w_rape", 0) * 3 +
                profile.get("w_dowry_deaths", 0) * 4 +
                profile.get("w_assault", 0) * 2 +
                profile.get("w_domestic_cruelty", 0) * 2
            ).fillna(0)
            print(f"    Women safety index range: {profile['women_safety_index'].min():.0f}–{profile['women_safety_index'].max():.0f}")

    # --- Join ST crimes (extends vulnerability_index) ---
    if not st_df.empty:
        st_city = st_df[st_df["district"].isin(
            {d.upper() for d in CITY_DISTRICTS.get(city, [])}
        )].copy()
        if not st_city.empty:
            st_cols = [c for c in st_city.columns if c.startswith("st_")]
            st_agg = st_city.groupby(["district", "year"])[st_cols].sum().reset_index()
            profile = profile.merge(st_agg, on=["district", "year"], how="left")
            for c in st_cols:
                profile[c] = profile[c].fillna(0)
            # Add ST crimes into vulnerability_index (combined SC+ST)
            st_contrib = (
                profile.get("st_murder", 0) * 5 +
                profile.get("st_rape", 0) * 4 +
                profile.get("st_hurt", 0) * 2 +
                profile.get("st_arson", 0) * 3
            ).fillna(0)
            if "vulnerability_index" in profile.columns:
                profile["vulnerability_index"] = profile["vulnerability_index"] + st_contrib
            else:
                profile["vulnerability_index"] = st_contrib

    # --- Join SC crimes ---
    if not sc_df.empty:
        sc_city = sc_df[sc_df["district"].isin(
            {d.upper() for d in CITY_DISTRICTS.get(city, [])}
        )].copy()
        if not sc_city.empty:
            sc_cols = [c for c in sc_city.columns if c.startswith("sc_")]
            sc_agg = sc_city.groupby(["district", "year"])[sc_cols].sum().reset_index()
            profile = profile.merge(sc_agg, on=["district", "year"], how="left")
            for c in sc_cols:
                profile[c] = profile[c].fillna(0)
            profile["vulnerability_index"] = (
                profile.get("sc_murder", 0) * 5 +
                profile.get("sc_rape", 0) * 4 +
                profile.get("sc_hurt", 0) * 2 +
                profile.get("sc_poa", 0) * 3
            ).fillna(0)

    # --- Join children crimes ---
    if not children_df.empty:
        ch_city = children_df[children_df["district"].isin(
            {d.upper() for d in CITY_DISTRICTS.get(city, [])}
        )].copy()
        if not ch_city.empty:
            ch_cols = [c for c in ch_city.columns if c.startswith("child_")]
            ch_agg = ch_city.groupby(["district", "year"])[ch_cols].sum().reset_index()
            profile = profile.merge(ch_agg, on=["district", "year"], how="left")
            for c in ch_cols:
                profile[c] = profile[c].fillna(0)

    # --- Join state-level police strength ---
    if not police_df.empty:
        police_df["state_norm"] = police_df["state"].apply(_resolve_state)
        p_state = police_df[police_df["state_norm"] == city_state]
        if not p_state.empty:
            profile = profile.merge(p_state[["state_norm", "year", "police_coverage_ratio"]],
                                    on=["state_norm", "year"], how="left")
            profile["police_coverage_ratio"] = profile["police_coverage_ratio"].fillna(0.75)
        else:
            profile["police_coverage_ratio"] = 0.75

    # --- Join state-level property value ---
    if not property_df.empty:
        property_df["state_norm"] = property_df["state"].apply(_resolve_state)
        p_state = property_df[property_df["state_norm"] == city_state]
        if not p_state.empty:
            profile = profile.merge(p_state[["state_norm", "year", "property_value_stolen_lakh"]],
                                    on=["state_norm", "year"], how="left")
            profile["property_value_stolen_lakh"] = profile["property_value_stolen_lakh"].fillna(0)
        else:
            profile["property_value_stolen_lakh"] = 0

    # --- Join auto theft detail ---
    if not auto_df.empty:
        auto_df["state_norm"] = auto_df["state"].apply(_resolve_state)
        a_state = auto_df[auto_df["state_norm"] == city_state]
        if not a_state.empty:
            profile = profile.merge(a_state[["state_norm", "year", "state_auto_theft_count"]],
                                    on=["state_norm", "year"], how="left")
            if "auto_recovery_rate" in auto_df.columns:
                profile = profile.merge(
                    a_state[["state_norm", "year", "auto_recovery_rate"]],
                    on=["state_norm", "year"], how="left"
                )
            profile["state_auto_theft_count"] = profile.get("state_auto_theft_count", pd.Series(0)).fillna(0)
        else:
            profile["state_auto_theft_count"] = 0

    # --- Join crime-by-place fractions ---
    if not place_df.empty:
        place_df["state_norm"] = place_df["state"].apply(_resolve_state)
        p_state = place_df[place_df["state_norm"] == city_state]
        if not p_state.empty:
            profile = profile.merge(
                p_state[["state_norm", "year", "residential_crime_pct", "highway_crime_pct", "market_crime_pct"]],
                on=["state_norm", "year"], how="left"
            )
        for col in ["residential_crime_pct", "highway_crime_pct", "market_crime_pct"]:
            profile[col] = profile.get(col, pd.Series(dtype=float)).fillna(0.33 if "residential" in col else 0.15)

    # --- Join murder motives ---
    if not motives_df.empty:
        motives_df["state_norm"] = motives_df["state"].apply(_resolve_state)
        m_state = motives_df[motives_df["state_norm"] == city_state]
        if not m_state.empty:
            profile = profile.merge(
                m_state[["state_norm", "year", "gang_murder_pct", "domestic_murder_pct"]],
                on=["state_norm", "year"], how="left"
            )
        profile["gang_murder_pct"]     = profile.get("gang_murder_pct", pd.Series(dtype=float)).fillna(0.10)
        profile["domestic_murder_pct"] = profile.get("domestic_murder_pct", pd.Series(dtype=float)).fillna(0.20)

    # --- Join police complaints rate ---
    if not complaints_df.empty:
        complaints_df["state_norm"] = complaints_df["state"].apply(_resolve_state)
        c_state = complaints_df[complaints_df["state_norm"] == city_state]
        if not c_state.empty:
            profile = profile.merge(
                c_state[["state_norm", "year", "police_complaint_rate"]],
                on=["state_norm", "year"], how="left"
            )
        profile["police_complaint_rate"] = profile.get("police_complaint_rate", pd.Series(dtype=float)).fillna(0.3)

    return profile


# ---------------------------------------------------------------------------
# Extended crime-type mapping including new sources
# ---------------------------------------------------------------------------

EXTENDED_CRIME_TO_TYPE: dict[str, str] = {
    # From IPC base
    "murder": "assault",
    "attempt_murder": "assault",
    "culpable_homicide": "assault",
    "hurt": "assault",
    "rape": "sexual_assault",
    "dacoity": "dacoity",
    "dacoity_murder": "dacoity",
    "robbery": "robbery",
    "kidnapping": "robbery",
    "burglary": "burglary",
    "theft": "pickpocketing",
    "other_theft": "pickpocketing",
    "auto_theft": "vehicle_theft",
    "cheating": "cyber_fraud",
    "criminal_breach_trust": "cyber_fraud",
    # From women crimes
    "w_rape": "sexual_assault",
    "w_assault": "sexual_assault",
    "w_dowry_deaths": "domestic_violence",
    "w_domestic_cruelty": "domestic_violence",
    "w_kidnapping": "robbery",
    # From SC crimes
    "sc_murder": "assault",
    "sc_rape": "sexual_assault",
    "sc_robbery": "robbery",
    "sc_hurt": "assault",
    # From children crimes
    "child_murder": "assault",
    "child_rape": "sexual_assault",
    "child_kidnap": "robbery",
    "child_total": "child_crime",
}


def district_to_hourly_zones_enriched(
    profile_df: pd.DataFrame,
    city: str,
    zones_per_district: int = ZONES_PER_DISTRICT,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Like district_to_hourly_zones but uses enriched profile with all crime sources.
    """
    rng = np.random.default_rng(seed=rng_seed)
    records: list[dict] = []

    available_crime_cols = {
        col: ctype for col, ctype in EXTENDED_CRIME_TO_TYPE.items()
        if col in profile_df.columns
    }

    print(f"    Extended crime columns: {len(available_crime_cols)} types → {sorted(set(available_crime_cols.values()))}")

    for _, row in profile_df.iterrows():
        district = str(row["district"])
        year = int(row["year"])
        base_lat, base_lon = DISTRICT_CENTROIDS.get(district, (20.0, 78.0))
        zone_shares = rng.dirichlet(np.ones(zones_per_district) * 2)

        # District-level enriched features (will be attached to all zone records)
        police_ratio    = float(row.get("police_coverage_ratio", 0.75))
        women_idx       = float(row.get("women_safety_index", 0))
        vuln_idx        = float(row.get("vulnerability_index", 0))
        prop_value      = float(row.get("property_value_stolen_lakh", 0))
        auto_count      = float(row.get("state_auto_theft_count", 0))
        res_pct         = float(row.get("residential_crime_pct", 0.33))
        hwy_pct         = float(row.get("highway_crime_pct", 0.15))
        mkt_pct         = float(row.get("market_crime_pct", 0.20))
        gang_mur_pct    = float(row.get("gang_murder_pct", 0.10))
        dom_mur_pct     = float(row.get("domestic_murder_pct", 0.20))
        complaint_rate  = float(row.get("police_complaint_rate", 0.30))

        for zone_i in range(1, zones_per_district + 1):
            zone_share = zone_shares[zone_i - 1]
            lat = base_lat + rng.uniform(-0.04, 0.04)
            lon = base_lon + rng.uniform(-0.04, 0.04)
            zone_id = f"{city[:3].upper()}_{district[:6].replace('.','').replace(' ','_')}_{zone_i}"

            for crime_col, crime_type in available_crime_cols.items():
                raw_val = row.get(crime_col, 0)
                if raw_val is None or (isinstance(raw_val, float) and np.isnan(float(raw_val))):
                    continue
                annual_count = float(raw_val) * zone_share
                if annual_count < 0.5:
                    continue

                for _ in range(max(1, int(annual_count))):
                    hour = _sample_hour(crime_type, rng)
                    month = int(rng.integers(1, 13))
                    day = int(rng.integers(1, 29))
                    weekday = int(rng.integers(0, 7))
                    temp = _sample_temp(city, month, rng)
                    precip = _sample_precip(month, rng)

                    records.append({
                        "city": city,
                        "zone_id": zone_id,
                        "timestamp": f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00",
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "weekday": weekday,
                        "is_weekend": int(weekday >= 5),
                        "crime_type": crime_type,
                        "district": district,
                        "lat": round(float(lat), 6),
                        "lon": round(float(lon), 6),
                        "temperature_c": round(float(temp), 1),
                        "precipitation_mm": round(float(precip), 2),
                        "wind_speed_kmh": round(float(abs(rng.normal(12, 5))), 1),
                        "is_rainy": int(precip > 1),
                        # Enriched district features
                        "police_coverage_ratio":      round(police_ratio, 3),
                        "women_safety_index":          round(women_idx, 1),
                        "vulnerability_index":         round(vuln_idx, 1),
                        "property_value_stolen_lakh":  round(prop_value, 1),
                        "state_auto_theft_count":      int(auto_count),
                        "residential_crime_pct":       round(res_pct, 3),
                        "highway_crime_pct":           round(hwy_pct, 3),
                        "market_crime_pct":            round(mkt_pct, 3),
                        "gang_murder_pct":             round(gang_mur_pct, 3),
                        "domestic_murder_pct":         round(dom_mur_pct, 3),
                        "police_complaint_rate":       round(complaint_rate, 3),
                    })

    return pd.DataFrame(records)


def build_enriched_zones(city: str, profile_df: pd.DataFrame) -> pd.DataFrame:
    """Build zone metadata with enriched district-level features."""
    rng = np.random.default_rng(seed=hash(city) % 2**32)
    districts = profile_df["district"].unique()
    zones: list[dict] = []

    # Per-district aggregated enrichment (average across years)
    agg_cols = {
        c: "mean" for c in [
            "women_safety_index", "vulnerability_index", "police_coverage_ratio",
            "property_value_stolen_lakh", "state_auto_theft_count",
            "residential_crime_pct", "highway_crime_pct", "market_crime_pct",
            "gang_murder_pct", "domestic_murder_pct", "police_complaint_rate",
        ] if c in profile_df.columns
    }
    district_agg = profile_df.groupby("district").agg(agg_cols).fillna(0).to_dict("index")

    for district in districts:
        base_lat, base_lon = DISTRICT_CENTROIDS.get(district, (20.0, 78.0))
        d_feats = district_agg.get(district, {})

        for i in range(1, ZONES_PER_DISTRICT + 1):
            zone_id = f"{city[:3].upper()}_{district[:6].replace('.','').replace(' ','_')}_{i}"
            zones.append({
                "zone_id": zone_id,
                "city": city,
                "district": district,
                "zone_type": rng.choice(["commercial", "residential", "transit", "mixed"]),
                "lat": round(float(base_lat + rng.uniform(-0.04, 0.04)), 6),
                "lon": round(float(base_lon + rng.uniform(-0.04, 0.04)), 6),
                "population_density": int(rng.integers(5000, 35000)),
                "bar_count_500m":           int(rng.integers(0, 12)),
                "atm_count_500m":           int(rng.integers(2, 18)),
                "market_count_500m":        int(rng.integers(2, 25)),
                "bus_stop_count_500m":      int(rng.integers(1, 15)),
                "nearest_police_station_km": round(float(rng.uniform(0.2, 4.0)), 2),
                "road_density":             round(float(rng.uniform(0.2, 1.0)), 2),
                "lighting_score":           round(float(rng.uniform(0.3, 1.0)), 2),
                "is_hotspot": False,
                # Enriched features from NCRB multi-source (11 datasets)
                "women_safety_index":         round(float(d_feats.get("women_safety_index", 0)), 1),
                "vulnerability_index":        round(float(d_feats.get("vulnerability_index", 0)), 1),
                "police_coverage_ratio":      round(float(d_feats.get("police_coverage_ratio", 0.75)), 3),
                "property_value_stolen_lakh": round(float(d_feats.get("property_value_stolen_lakh", 0)), 1),
                "state_auto_theft_count":     int(d_feats.get("state_auto_theft_count", 0)),
                "residential_crime_pct":      round(float(d_feats.get("residential_crime_pct", 0.33)), 3),
                "highway_crime_pct":          round(float(d_feats.get("highway_crime_pct", 0.15)), 3),
                "market_crime_pct":           round(float(d_feats.get("market_crime_pct", 0.20)), 3),
                "gang_murder_pct":            round(float(d_feats.get("gang_murder_pct", 0.10)), 3),
                "domestic_murder_pct":        round(float(d_feats.get("domestic_murder_pct", 0.20)), 3),
                "police_complaint_rate":      round(float(d_feats.get("police_complaint_rate", 0.30)), 3),
            })

    return pd.DataFrame(zones)


# ---------------------------------------------------------------------------
# End-to-end enriched pipeline
# ---------------------------------------------------------------------------

def load_and_prepare_enriched(
    data_dir: str = "backend/data/raw/ncrb",
    output_dir: str = "backend/data/processed",
    cities: list[str] | None = None,
) -> dict:
    """
    Full multi-source pipeline:
    NCRB IPC + Women + SC + Children + Police + Property + Auto theft
    → enriched zone records + zones metadata
    """
    import json
    from pathlib import Path as P

    global RAW_DIR
    RAW_DIR = P(data_dir)
    target_cities = cities or list(CITY_DISTRICTS.keys())
    P(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n[A] Loading all NCRB data sources...")
    ipc_df = load_ncrb_raw(data_dir)

    print("\n[B] Loading supplementary district-level crime sources...")
    women_df    = load_women_crimes()
    sc_df       = load_sc_crimes()
    st_df       = load_st_crimes()
    children_df = load_children_crimes()

    print("\n[C] Loading state-level context features...")
    police_df     = load_police_strength()
    property_df   = load_property_value()
    auto_df       = load_auto_theft_detail()
    place_df      = load_crime_by_place()
    motives_df    = load_murder_motives()
    complaints_df = load_police_complaints()

    all_records: list[pd.DataFrame] = []
    all_zones: list[pd.DataFrame] = []

    for city in target_cities:
        print(f"\n[D] Building enriched profile for {city}...")
        city_ipc = filter_city_districts(ipc_df, city)
        if city_ipc.empty:
            print(f"  Skipped — no IPC districts matched.")
            continue

        years = sorted(city_ipc["year"].unique())
        print(f"  IPC rows: {len(city_ipc)} | years {years[0]}–{years[-1]}")

        profile = build_enriched_district_profile(
            city_ipc, women_df, sc_df, st_df, children_df,
            police_df, property_df, auto_df,
            place_df, motives_df, complaints_df, city
        )

        records_df = district_to_hourly_zones_enriched(profile, city)
        zones_df   = build_enriched_zones(city, profile)

        if not records_df.empty:
            all_records.append(records_df)
            all_zones.append(zones_df)
            print(f"  Generated {len(records_df):,} records | {len(zones_df)} zones")

    if not all_records:
        raise ValueError("No records built from enriched pipeline.")

    final_records = pd.concat(all_records, ignore_index=True)
    final_zones   = pd.concat(all_zones,   ignore_index=True)

    rec_path   = f"{output_dir}/crime_records.csv"
    zones_path = f"{output_dir}/zones.csv"

    final_records.to_csv(rec_path, index=False)
    final_zones.to_csv(zones_path, index=False)

    summary = {
        c: {
            "records": int(len(final_records[final_records["city"] == c])),
            "zones": int(len(final_zones[final_zones["city"] == c])),
        }
        for c in target_cities if c in final_records["city"].values
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Report enrichment stats
    new_cols = [c for c in final_zones.columns if c in [
        "women_safety_index", "vulnerability_index", "police_coverage_ratio",
        "property_value_stolen_lakh", "state_auto_theft_count"
    ]]
    print(f"\n{'='*60}")
    print(f"ENRICHED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(final_records):,}")
    print(f"Total zones:   {len(final_zones)}")
    print(f"Crime types:   {sorted(final_records['crime_type'].unique())}")
    print(f"Enriched zone features: {new_cols}")
    for c in target_cities:
        s = summary.get(c, {})
        print(f"  {c}: {s.get('records',0):,} records | {s.get('zones',0)} zones")
    print(f"\nSaved to {output_dir}/")
    return {"records": rec_path, "zones": zones_path, "summary": summary}


if __name__ == "__main__":
    load_and_prepare_enriched()
