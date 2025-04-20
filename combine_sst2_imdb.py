import argparse
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split



# ------------- 데이터 로더 ------------- #

def load_sst2(split: str) -> pd.DataFrame:
    """Glue 'sst2' split 을 DataFrame(text, label)로 반환"""
    ds = load_dataset("glue", "sst2", split=split)
    return pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})

def load_imdb(split: str) -> pd.DataFrame:
    """HuggingFace 'imdb' split 을 DataFrame(text, label)로 반환"""
    ds = load_dataset("imdb", split=split)
    return pd.DataFrame({"text": ds["text"], "label": ds["label"]})


def main(args):
    # (1) 데이터 읽기
    sst_train = load_sst2("train")
    sst_test   = load_sst2("validation")

    imdb_train = load_imdb("train")
    imdb_test  = load_imdb("test")
    imdb_full  = pd.concat([imdb_train, imdb_test], ignore_index=True)

    if args.imdb_frac < 1.0:
        imdb_full = imdb_full.sample(frac=args.imdb_frac, random_state=42)

    # (2) 결합 및 셔플
    combined_df = pd.concat([sst_train, sst_test, imdb_full], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    r_test = 0.1 # test 10%
    r_dev = r_test / (1 - r_test) # dev 10% of train -> dev 도 10% of 전체
    # (3) Train / Dev / Test 분할
    train_pool, test_df = train_test_split(
        combined_df,
        test_size=r_test,
        random_state=42,
        stratify=combined_df["label"],
    )

    train_df, dev_df = train_test_split(
        train_pool,
        test_size=r_dev,
        random_state=42,
        stratify=train_pool["label"],   # ✅ 수정
    )
    # (4) 저장
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "combined_train.csv", index=False)
    test_df.to_csv(out_dir / "combined_test.csv",   index=False)
    dev_df.to_csv(out_dir   / "combined_dev.csv",   index=False)

    print("✅ Saved:")
    print("  -", out_dir / "combined_train.csv", f"({len(train_df)} rows)")
    print("  -", out_dir / "combined_dev.csv",   f"({len(dev_df)} rows)")    
    print("  -", out_dir / "combined_test.csv",   f"({len(test_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./combined_data", help="CSV 저장 경로")
    parser.add_argument(
        "--imdb_frac", type=float, default=1.0,
        help="IMDB 데이터 활용 비율 (디버깅 시 0.1 등으로 축소 가능)",
    )
    args = parser.parse_args()
    main(args)
