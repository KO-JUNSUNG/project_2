# BERT Sentiment Classification on **SST‑2 + IMDB**

> **Original repo:** <https://github.com/YJiangcm/SST-2-sentiment-analysis/tree/master>  
> **This fork:** adds data‑merging, and updated dataloader for sentence‑level sentiment over mixed domains.

---

## 1. Quick Start 🚀
```bash
conda create -n bert-sent python=3.9
conda activate bert-sent
pip install -r requirements.txt
# one‑shot data prep
python combine_sst2_imdb.py --output_dir combined_data --dev_ratio 0.1
# training
python run_Bert_model.py \
  --train_path combined_data/combined_train.csv \
  --dev_path   combined_data/combined_dev.csv \
  --test_path  combined_data/combined_test.csv \
  --max_seq_len 256 --batch_size 16 --epochs 3
# evaluation (re‑use best checkpoint, no retrain)
python run_Bert_model.py \
  --mode test \
  --test_path combined_data/combined_test.csv \
  --checkpoint output/Bert_combined/best.pth.tar
