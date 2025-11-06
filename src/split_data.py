import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_RAW, TRAIN_PATH, TEST_PATH

def main():
    df = pd.read_csv(str(DATA_RAW))
    assert {"review", "sentiment"} <= set(df.columns), "需要列: review, sentiment"
    df.dropna(subset=["review", "sentiment"]).copy()
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    assert df["label"].isin([0,1]).all(), "sentiment 只支持 positive/negative"

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_df[["label", "review"]].to_csv(TRAIN_PATH, sep="\t", index=False)
    test_df[["label", "review"]].to_csv(TEST_PATH, sep="\t", index=False)
    print(f"✅ train: {len(train_df)}  test: {len(test_df)}")

if __name__ == "__main__":
    main()
