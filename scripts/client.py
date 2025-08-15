#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/predict")
    ap.add_argument("--csv", default="data/future_unseen_examples.csv")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))
    records = df.head(args.n).to_dict(orient="records")
    payload = {"records": records}

    resp = requests.post(args.api, json=payload, timeout=30)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
