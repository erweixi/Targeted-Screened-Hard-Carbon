from __future__ import annotations

import argparse
from pathlib import Path

from .extended_utils import build_original_master_dataset


def parse_args():
    p = argparse.ArgumentParser(description='Build the original 565-row master dataset and map extra missing-value features from the uploaded Excel-derived table.')
    p.add_argument('--ice-csv', default='data/hc_dataset_ice.csv')
    p.add_argument('--plateau-csv', default='data/hc_dataset_plateau.csv')
    p.add_argument('--extended-csv', default='data/hc_dataset_extended_preprocessed.csv')
    p.add_argument('--out-csv', default='data/hc_dataset_original_master_with_missing_features.csv')
    p.add_argument('--mapping-report-csv', default='data/hc_dataset_original_master_mapping_report.csv')
    return p.parse_args()


def main():
    args = parse_args()
    df = build_original_master_dataset(
        ice_csv=args.ice_csv,
        plateau_csv=args.plateau_csv,
        extended_csv=args.extended_csv,
        out_csv=args.out_csv,
        mapping_report_csv=args.mapping_report_csv,
    )
    print(f'Saved: {Path(args.out_csv)}')
    print(f'Rows: {len(df)}')
    print(f'Mapped rows: {int(df["mapping_found"].sum())}')
    print(f'Unmatched rows: {int((~df["mapping_found"]).sum())}')


if __name__ == '__main__':
    main()
