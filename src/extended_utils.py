from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import FEATURE_COLUMNS, regression_metrics, split_train_test, cv_scores_to_summary, fold_metrics_frame, ensure_jsonable

EXTRA_FEATURE_COLUMNS = [
    'carbonization_time_h',
    'heating_rate_C_min',
    'pore_volume_cm3_g',
    'lc_nm',
    'la_nm',
]

ALL_MODELING_COLUMNS = FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS

COLUMN_MAP = {
    '序号': 'record_id',
    'DOI': 'doi',
    '材料名称': 'material_name',
    '碳化温度': 'carbonization_temperature_C',
    '碳化时间': 'carbonization_time_h',
    '加热速率': 'heating_rate_C_min',
    'd002': 'd002_nm',
    'La': 'la_nm',
    'Lc': 'lc_nm',
    'ID/IG': 'id_ig',
    'SSA': 'ssa_m2_g',
    'Volume': 'pore_volume_cm3_g',
    '类别': 'category',
    '电解液': 'electrolyte_text',
    '电流密度': 'current_density_mA_g',
    'ICE': 'ice',
    '平台容量': 'plateau_capacity_mAh_g',
    '可逆容量': 'reversible_capacity_mAh_g',
    '倍率': 'rate_capability',
}

ETHER_KEYWORDS = [
    'diglyme', 'dme', 'degdme', 'tetraglyme', 'triglyme', 'glyme',
    'dimethoxyethane', 'glycol dimethyl ether', 'methoxy ethyl', 'monoglyme', 'diethylene glycol dimethyl',
]
CARBONATE_KEYWORDS = [
    'ethylene carbonate', 'ec', 'dec', 'dmc', 'emc', 'pc', 'fec', 'vc',
    'propylene carbonate', 'carbonate', 'ethyl methyl carbonate', 'diethyl carbonate',
    'dimethyl carbonate', 'fluoroethylene carbonate', 'vinylene carbonate',
]

FEATURE_LABELS = {
    'carbonization_temperature_C': 'Carbonization temperature',
    'd002_nm': 'd002',
    'id_ig': 'ID/IG',
    'ssa_m2_g': 'SSA',
    'electrolyte_type': 'Electrolyte type',
    'current_density_mA_g': 'Current density',
    'carbonization_time_h': 'Carbonization time',
    'heating_rate_C_min': 'Heating rate',
    'pore_volume_cm3_g': 'Pore volume',
    'lc_nm': 'Lc',
    'la_nm': 'La',
    'ice': 'ICE',
    'lce': 'LCE',
    'plateau_capacity_mAh_g': 'Plateau capacity',
    'reversible_capacity_mAh_g': 'Reversible capacity',
}

TASK_TARGETS = {
    'ice': 'lce',
    'plateau': 'plateau_capacity_mAh_g',
    'capacity': 'reversible_capacity_mAh_g',
}

ORIGINAL_MASTER_COLUMNS = FEATURE_COLUMNS + ['ice', 'lce', 'plateau_capacity_mAh_g'] + EXTRA_FEATURE_COLUMNS
PRIMARY_MATCH_COLUMNS = [
    'carbonization_temperature_C', 'd002_nm', 'id_ig', 'ssa_m2_g', 'current_density_mA_g'
]
FALLBACK_MATCH_COLUMNS = [
    'carbonization_temperature_C', 'd002_nm', 'ssa_m2_g'
]
MATCH_METADATA_COLUMNS = ['record_id', 'doi', 'material_name', 'category', 'electrolyte_text', 'reversible_capacity_mAh_g']


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    text = str(value).strip().lower()
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text


def map_electrolyte_type(text: object) -> float:
    s = _normalize_text(text)
    if not s:
        return np.nan
    ether = any(k in s for k in ETHER_KEYWORDS)
    carbonate = any(k in s for k in CARBONATE_KEYWORDS)
    if ether and not carbonate:
        return 1.0
    if carbonate and not ether:
        return 0.0
    if ether:
        return 1.0
    if carbonate:
        return 0.0
    return np.nan


def load_extended_excel(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=COLUMN_MAP).copy()
    if 'record_id' not in df.columns:
        df.insert(0, 'record_id', np.arange(1, len(df) + 1))
    numeric_cols = [
        'carbonization_temperature_C', 'carbonization_time_h', 'heating_rate_C_min', 'd002_nm',
        'la_nm', 'lc_nm', 'id_ig', 'ssa_m2_g', 'pore_volume_cm3_g', 'current_density_mA_g',
        'ice', 'plateau_capacity_mAh_g', 'reversible_capacity_mAh_g', 'rate_capability'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['electrolyte_type'] = df['electrolyte_text'].map(map_electrolyte_type)
    df['lce'] = np.log(df['ice'] / (1 - df['ice']))
    return df


def save_preprocessed_csv(excel_path: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = load_extended_excel(excel_path)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_preprocessed_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_xy_for_target(df: pd.DataFrame, features: list[str], target: str):
    cols = features + [target]
    subset = df[cols].dropna().copy()
    X = subset[features].copy()
    y = subset[target].copy()
    return X, y, subset


def complete_case_count(df: pd.DataFrame, cols: Iterable[str]) -> int:
    return int(df[list(cols)].dropna().shape[0])


def _match_key_frame(df: pd.DataFrame, cols: list[str], decimals: int = 6) -> pd.Series:
    parts = []
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce').round(decimals)
        parts.append(s.map(lambda v: '' if pd.isna(v) else f'{v:.{decimals}f}'))
    out = parts[0].copy()
    for s in parts[1:]:
        out = out.str.cat(s, sep='|')
    return out


def build_original_master_dataset(
    ice_csv: str | Path,
    plateau_csv: str | Path,
    extended_csv: str | Path,
    out_csv: str | Path,
    mapping_report_csv: str | Path,
) -> pd.DataFrame:
    ice_df = pd.read_csv(ice_csv).copy()
    plateau_df = pd.read_csv(plateau_csv).copy()
    ext_df = pd.read_csv(extended_csv).copy()

    master = ice_df.merge(
        plateau_df[FEATURE_COLUMNS + ['plateau_capacity_mAh_g']],
        on=FEATURE_COLUMNS,
        how='inner',
        validate='one_to_one',
    )

    ext_keep = PRIMARY_MATCH_COLUMNS + FALLBACK_MATCH_COLUMNS + EXTRA_FEATURE_COLUMNS + MATCH_METADATA_COLUMNS
    ext_keep = list(dict.fromkeys(ext_keep))
    ext_small = ext_df[ext_keep].copy().reset_index(drop=True)
    ext_small['ext_row_id'] = np.arange(len(ext_small))
    ext_small['primary_key'] = _match_key_frame(ext_small, PRIMARY_MATCH_COLUMNS)
    ext_small['fallback_key'] = _match_key_frame(ext_small, FALLBACK_MATCH_COLUMNS)

    master['primary_key'] = _match_key_frame(master, PRIMARY_MATCH_COLUMNS)
    master['fallback_key'] = _match_key_frame(master, FALLBACK_MATCH_COLUMNS)

    primary_map = ext_small.drop_duplicates(subset=['primary_key']).set_index('primary_key')['ext_row_id'].to_dict()

    fallback_counts = ext_small.groupby('fallback_key')['ext_row_id'].nunique().to_dict()
    fallback_unique = ext_small[ext_small['fallback_key'].map(fallback_counts) == 1].set_index('fallback_key')['ext_row_id'].to_dict()

    used_ext_ids: set[int] = set()
    match_rows = []
    match_methods = []
    ext_lookup = ext_small.set_index('ext_row_id')
    for _, row in master.iterrows():
        ext_id = primary_map.get(row['primary_key'])
        method = 'primary_exact'
        if ext_id is None or ext_id in used_ext_ids:
            ext_id = fallback_unique.get(row['fallback_key'])
            method = 'fallback_exact_temp_d002_ssa'
        if ext_id is None or ext_id in used_ext_ids:
            match_rows.append({c: np.nan for c in EXTRA_FEATURE_COLUMNS + MATCH_METADATA_COLUMNS})
            match_methods.append('unmatched')
            continue
        used_ext_ids.add(int(ext_id))
        ext_row = ext_lookup.loc[int(ext_id), EXTRA_FEATURE_COLUMNS + MATCH_METADATA_COLUMNS].to_dict()
        match_rows.append(ext_row)
        match_methods.append(method)

    match_df = pd.DataFrame(match_rows)
    master = pd.concat([master.reset_index(drop=True), match_df], axis=1)
    master['mapping_method'] = match_methods
    master['mapping_found'] = master['mapping_method'] != 'unmatched'

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    master[ORIGINAL_MASTER_COLUMNS + MATCH_METADATA_COLUMNS + ['mapping_method', 'mapping_found']].to_csv(out_csv, index=False)

    report = pd.DataFrame([
        {
            'n_rows_original_master': int(len(master)),
            'n_rows_matched': int(master['mapping_found'].sum()),
            'n_rows_unmatched': int((~master['mapping_found']).sum()),
            'matched_fraction': float(master['mapping_found'].mean()),
            'n_primary_exact': int((master['mapping_method'] == 'primary_exact').sum()),
            'n_fallback_exact_temp_d002_ssa': int((master['mapping_method'] == 'fallback_exact_temp_d002_ssa').sum()),
            'n_unmatched': int((master['mapping_method'] == 'unmatched').sum()),
        }
    ])
    report.to_csv(mapping_report_csv, index=False)
    return master
