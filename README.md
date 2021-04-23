# PointerSQL_pytorch
```
.
├── data
│   ├── glove.6B.100d.txt # 347 MB
│   ├── glove.6B.50d-relativized.txt # smaller file, for testing
│   ├── jmt_char_n_gram.txt # 969.1 MB
│   ├── wikisql_dev.dat
│   ├── wikisql_test.dat
│   └── wikisql_train.dat
├── .gitignore
├── data.py
├── main.py
├── models.py
├── README.md
└── utils.py
```

- main branch: follow setup in the paper [POINTING OUT SQL QUERIES FROM TEXT](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/nl2prog.pdf)
  - put pretrained word vectors in `/data`, uploaded to (Google Drive)[https://drive.google.com/drive/u/1/folders/1Ozk_KIa7NXIBWo2y5dYdmUQEdFAJWAlD]
  - Problem: the paper said the the pretrained vectors are set untrainable to avoid over-fitting, but many special tokens in SQL will be labeld `UNK`

- no_pretrained branch: only data changed, the rest are the same as project 2
  - no need to put pretrained word vectors
