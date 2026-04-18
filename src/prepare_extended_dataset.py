from __future__ import annotations

import argparse
from pathlib import Path

from .extended_utils import save_preprocessed_csv


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess the extended hard-carbon Excel database into a modeling CSV.')
    p.add_argument('--excel', required=True, help='Path to the uploaded Excel database')
    p.add_argument('--out-csv', default='data/hc_dataset_extended_preprocessed.csv')
    return p.parse_args()


def main():
    args = parse_args()
    df = save_preprocessed_csv(args.excel, args.out_csv)
    print(f'Saved: {Path(args.out_csv)}')
    print(f'Rows: {len(df)}')
    print(f'Electrolyte type missing: {int(df["electrolyte_type"].isna().sum())}')


if __name__ == '__main__':
    main()
