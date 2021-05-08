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

PointerSQL GitHub: https://github.com/microsoft/PointerSQL

- main branch: follow setup in the paper [POINTING OUT SQL QUERIES FROM TEXT](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/nl2prog.pdf)
  - put pretrained word vectors in `/data`, uploaded to (Google Drive)[https://drive.google.com/drive/u/1/folders/1Ozk_KIa7NXIBWo2y5dYdmUQEdFAJWAlD]
  - Problem: the paper said the the pretrained vectors are set untrainable to avoid over-fitting, but many special tokens in SQL will be labeld `UNK`

- no_pretrained branch: only data changed, the rest are the same as project 2
  - no need to put pretrained word vectors


Namespace(batch_size=1, decoder_len_limit=22, dev_path='data/wikisql_dev_small.dat', emb_dim=100, embedding_dropout=0.2, epochs=10, hidden_dim=100, lr=0.001, n_gram_path='data/jmt_char_n_gram.txt', n_layers=3, test_path='data/wikisql_test.dat', train_path='data/wikisql_train_small.dat', use_pretrained=False, word_vecs_path='data/glove.6B.50d-relativized.txt')
Loaded 500 examples from file data/wikisql_train_small.dat
Loaded 125 examples from file data/wikisql_dev_small.dat
Loaded 15878 examples from file data/wikisql_test.dat
Train length: 45
Train output length: 22
Train matrix: [[   2    3    4 ...    0    0    0]
 [  14   15   16 ...    0    0    0]
 [  25   26   27 ...    0    0    0]
 ...
 [2720  118 1488 ...    0    0    0]
 [2727    3    4 ...    0    0    0]
 [2729 2233  539 ...    0    0    0]]; shape = (500, 45)
Total loss on epoch 1: 7742.546387
Time elapsed:  32.14895415306091
Total loss on epoch 2: 5996.418945
Time elapsed:  33.16028428077698
Total loss on epoch 3: 5551.704102
Time elapsed:  31.2605140209198
Total loss on epoch 4: 5282.408203
Time elapsed:  33.595783948898315
Total loss on epoch 5: 5049.258301
Time elapsed:  32.91062664985657
Total loss on epoch 6: 4871.126953
Time elapsed:  31.38126492500305
Total loss on epoch 7: 4715.552734
Time elapsed:  31.733427047729492
Total loss on epoch 8: 4586.744141
Time elapsed:  32.085618019104004
Total loss on epoch 9: 4469.372070
Time elapsed:  33.692546129226685
Total loss on epoch 10: 4366.692383
Time elapsed:  30.9312961101532
=======TRAIN SET=======
Example 49
  x      = "2-14218387-1 date venue score result competition name the venue for 23^january^2009"
  y_tok  = "['select', '<GO>', 'venue', 'from', '2-14218387-1', 'where', 'date', '=', '23^january^2009']"
  y_pred = "['select', 'venue', 'from', '2-14218387-1', 'where', '23^january^2009', '=', '23^january^2009']"
Example 99
  x      = "1-2731431-1 no peak location elevation^(m) prominence^(m) col^height^(m) col^location parent what is the col^location for the location of france^/^italy ?"
  y_tok  = "['select', '<GO>', 'col^location', 'from', '1-2731431-1', 'where', 'location', '=', 'france^/^italy']"
  y_pred = "['select', 'col^height^(m)', 'from', '1-2731431-1', 'where', 'france^/^italy', '=', 'france^/^italy']"
Example 149
  x      = "2-18377709-2 res. record opponent method event round time location which opponent has a record of 2–0 ?"
  y_tok  = "['select', '<GO>', 'opponent', 'from', '2-18377709-2', 'where', 'record', '=', '2–0']"
  y_pred = "['select', 'record', 'from', '2-18377709-2', 'where', 'record', '=', '2–0']"
Example 199
  x      = "1-10707142-2 rnd race^name circuit city/location date pole^position winning^driver winning^team report how many reports of races took place on date october^16 ?"
  y_tok  = "['select', 'count', 'report', 'from', '1-10707142-2', 'where', 'date', '=', 'october^16']"
  y_pred = "['select', 'race^name', 'from', '1-10707142-2', 'where', 'date', '=', 'date']"
Example 249
  x      = "2-17124374-5 date visitor score home decision series which decision has a series of 3^–^3 ?"
  y_tok  = "['select', '<GO>', 'decision', 'from', '2-17124374-5', 'where', 'series', '=', '3^–^3']"
  y_pred = "['select', 'visitor', 'from', '2-17124374-5', 'where', 'series', '=', '3^–^3']"
Example 299
  x      = "2-11970261-2 country preliminaries interview swimsuit evening^gown average what is country pennsylvania 's average where the swimsuit is smaller than 9.109 and the evening^gown is smaller than 9.163 ?"
  y_tok  = "['select', 'avg', 'average', 'from', '2-11970261-2', 'where', 'swimsuit', '<', '9.109', 'and', 'country', '=', 'pennsylvania', 'and', 'evening^gown', '<', '9.163']"
  y_pred = "['select', 'interview', 'from', '2-11970261-2', 'where', 'swimsuit', '=', '9.109']"
Example 349
  x      = "2-18304058-2 team^name schools sports host nickname(s) colors enrollment^(2013/14) what is the name of the team from schools goreville^vienna school ?"
  y_tok  = "['select', '<GO>', 'team^name', 'from', '2-18304058-2', 'where', 'schools', '=', 'goreville^vienna']"
  y_pred = "['select', 'colors', 'from', '2-18304058-2', 'where', 'schools', '=', 'goreville^vienna']"
Example 399
  x      = "1-2062148-3 result date race venue group distance weight^(kg) jockey winner/2nd what group was the race hollindale^stakes in ?"
  y_tok  = "['select', '<GO>', 'group', 'from', '1-2062148-3', 'where', 'race', '=', 'hollindale^stakes']"
  y_pred = "['select', 'date', 'from', '1-2062148-3', 'where', 'hollindale^stakes', '=', 'hollindale^stakes']"
Example 449
  x      = "2-1143966-1 season games lost tied points pct^% goals^for goals^against standing when less than 37 points are scored , what 's the lowest pct^% found ?"
  y_tok  = "['select', 'min', 'pct^%', 'from', '2-1143966-1', 'where', 'points', '<', '37']"
  y_pred = "['select', 'standing', 'from', '2-1143966-1', 'where', '37', '=', '37']"
Example 499
  x      = "1-24222929-4 episode^number^production^number title original^airdate time^of^airing total^viewers^on^hallmark total^viewers^on^hallmark+1 total^viewers rank^on^channel name the episode number for tasers^and^mind^erasers"
  y_tok  = "['select', '<GO>', 'episode^number^production^number', 'from', '1-24222929-4', 'where', 'title', '=', 'tasers^and^mind^erasers']"
  y_pred = "['select', 'original^airdate', 'from', '1-24222929-4', 'where', 'rank^on^channel', '=', 'tasers^and^mind^erasers']"
Exact logical form matches: 0 / 500 = 0.000
Token-level accuracy: 500 / 5160 = 0.097
=======DEV SET=======
Example 49
  x      = "1-15001681-1 year mens^singles womens^singles mens^doubles womens^doubles mixed^doubles in the year 2006 , the womens^singles had no^competition and the mens^doubles were roland^hilti^kilian^pfister , what were the womens^doubles"
  y_tok  = "['select', '<GO>', 'womens^doubles', 'from', '1-15001681-1', 'where', 'womens^singles', '=', 'no^competition', 'and', 'mens^doubles', '=', 'roland^hilti^kilian^pfister', 'and', 'year', '=', '2006']"
  y_pred = "['select', 'mens^doubles', 'from', '1-15001681-1', 'where', 'year', '=', 'no^competition']"
Example 99
  x      = "2-15400315-1 rank nation gold silver bronze total what is the nation when there is a total less than 27 , gold is less than 1 , and bronze is more than 1 ?"
  y_tok  = "['select', '<GO>', 'nation', 'from', '2-15400315-1', 'where', 'total', '<', '27', 'and', 'gold', '<', '1', 'and', 'bronze', '>', '1']"
  y_pred = "['select', 'total', 'from', '2-15400315-1', 'where', 'total', '=', '27']"
Exact logical form matches: 0 / 125 = 0.000
Token-level accuracy: 125 / 1329 = 0.094