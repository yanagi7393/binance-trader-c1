import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from common_utils_dev import (
    make_dirs,
    load_text,
    get_filename_by_path,
    to_parquet,
    get_filename_by_path,
    to_abs_path,
)


CONFIG = {
    "raw_spot_rawdata_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/raw/spot/"
    ),
    "raw_future_rawdata_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/raw/future/"
    ),
    "cleaned_rawdata_store_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/cleaned/"
    ),
    "candidate_assets_path": to_abs_path(__file__, "./candidate_assets.txt"),
    "query_min_start_dt": "2018-01-01",
    "boundary_dt_must_have_data": "2020-01-01",
    "use_only_spot": True,
}
OHLCV = ["open", "high", "low", "close", "volume"]


def _ffill_by_last_close(df):
    df_ = pd.concat(
        [df[["open", "high", "low"]], df["close"].resample("1min").ffill()], axis=1
    )[["open", "high", "low", "close"]].bfill(axis=1)

    return pd.concat([df_, df["volume"]], axis=1).fillna(0)


def build_rawdata(
    raw_spot_rawdata_dir=CONFIG["raw_spot_rawdata_dir"],
    raw_future_rawdata_dir=CONFIG["raw_future_rawdata_dir"],
    cleaned_rawdata_store_dir=CONFIG["cleaned_rawdata_store_dir"],
    candidate_assets_path=CONFIG["candidate_assets_path"],
    query_min_start_dt=CONFIG["query_min_start_dt"],
    boundary_dt_must_have_data=CONFIG["boundary_dt_must_have_data"],
    use_only_spot=CONFIG["use_only_spot"],
):
    assert use_only_spot in (True, False)
    make_dirs([cleaned_rawdata_store_dir])
    candidate_assets = load_text(path=candidate_assets_path)

    dfs = {}
    last_index = None
    for candidate_asset in tqdm(candidate_assets):
        spot_file_path = os.path.join(
            raw_spot_rawdata_dir, f"{candidate_asset}.parquet"
        )
        future_file_path = os.path.join(
            raw_future_rawdata_dir, f"{candidate_asset}.parquet"
        )

        spot_df = pd.read_parquet(spot_file_path)[OHLCV].sort_index()

        if os.path.exists(future_file_path) and (use_only_spot is False):
            future_df = pd.read_parquet(future_file_path)[OHLCV].sort_index()
            df = pd.concat([spot_df[spot_df.index < future_df.index[0]], future_df])

        else:
            print(f"[!] Use no future data: {candidate_asset}")
            df = spot_df.copy()

        # Cleaning
        df = df.sort_index()
        df = _ffill_by_last_close(df=df)

        # Loc & check
        df = df[query_min_start_dt:]
        if df.index[0] > pd.Timestamp(boundary_dt_must_have_data):
            print(f"[!] Skiped: {candidate_asset}")
            continue

        assert not df.isnull().any().any()
        assert len(df.index.unique()) == len(df.index)

        df.index = df.index.tz_localize("utc")
        if last_index is None:
            last_index = df.index[-1]
        else:
            last_index = min(last_index, df.index[-1])

        dfs[candidate_asset] = df

    # Cut data until last index
    for key, value in dfs.items():
        to_parquet(
            df=value[:last_index],
            path=os.path.join(cleaned_rawdata_store_dir, key + ".parquet.zstd"),
        )

    print(f"[!] Data until: {last_index}")
    print(f"[+] Built rawdata: {len(dfs)}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_rawdata)
