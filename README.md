# cnn-ner

## Files
```
.
├── data
├── lib
│   ├── cnn.py
│   ├── data.py
│   ├── flags.py
│   ├── __init__.py
│   ├── word2vec.py
├── model
├── train.py
└── word2vec
    ├── cut.py
    ├── data
    ├── demo.py
    ├── log.py
    ├── pre_train.py
    ├── resume.sh
    ├── t2s.py
    ├── train.py
    ├── utils.py
    ├── wiki.py
    └── xmltools.py
```

## Dependency

- python modules
  - jieba
  - numpy
  - tensorflow
  - gensim
  - sklearn

## Start

- 训练词向量

```
cd word2vec
./train_w2c.sh
cd ..
```
- 训练数据/测试数据 放在data目录下
- 训练模型 `./train.py`
- 预测 `python prefict`
