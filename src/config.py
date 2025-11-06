from pathlib import Path

# 项目根目录：.../Sentimental_Analysis
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data_raw" / "reviews.csv"
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.tsv"
TEST_PATH  = DATA_DIR / "test.tsv"

MODEL_DIR  = ROOT / "checkpoint"
VOCAB_PATH = MODEL_DIR / "vocab.npy"
MODEL_PATH = MODEL_DIR / "nb_model.npz"

VOCAB_SIZE = 20000
MIN_FREQ   = 2
USE_BIGRAM = True
ALPHA      = 1.0

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 如果后续用到需要字符串的库（如 pandas），在用的时候 str() 一下即可：
# pd.read_csv(str(DATA_RAW))
