# sentiment_minimal

## Quickstart
1) 安装依赖
   pip install -r requirements.txt

2) 准备原始数据
   data_raw/reviews.csv  (两列: review, sentiment[positive|negative])

3) 切分数据
   python -m src.split_data

4) 训练
   python -m src.train

5) 评测
   python -m src.evaluate

6) 预测
   python -m src.predict --text "This movie is awesome!"
