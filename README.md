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

CS388 NLP final project. Implement PointSQL in pytorch.

PointerSQL GitHub (in TF): https://github.com/microsoft/PointerSQL

- main branch: follow setup in the paper [POINTING OUT SQL QUERIES FROM TEXT](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/nl2prog.pdf)
  - put pretrained word vectors in `/data`, uploaded to (Google Drive)[https://drive.google.com/drive/u/1/folders/1Ozk_KIa7NXIBWo2y5dYdmUQEdFAJWAlD]
  - Problem: the paper said the the pretrained vectors are set untrainable to avoid over-fitting, but many special tokens in SQL will be labeld `UNK`

- no_pretrained branch: only data changed, the rest are the same as project 2
  - no need to put pretrained word vectors


!python3 main.py --epochs=15 --lr=0.01

Namespace(batch_size=1, decoder_len_limit=22, dev_path='data/wikisql_dev_small.dat', emb_dim=100, embedding_dropout=0.2, epochs=15, hidden_dim=100, lr=0.01, n_gram_path='data/jmt_char_n_gram.txt', n_layers=3, test_path='data/wikisql_test.dat', train_path='data/wikisql_train_small.dat', use_pretrained=False, word_vecs_path='data/glove.6B.50d-relativized.txt')
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
Total loss on epoch 1: 5251.159180
Time elapsed:  30.40560221672058
Total loss on epoch 2: 3376.982178
Time elapsed:  30.44365620613098
Total loss on epoch 3: 2486.269043
Time elapsed:  30.013756036758423
Total loss on epoch 4: 1879.708740
Time elapsed:  29.963207960128784
Total loss on epoch 5: 1419.447876
Time elapsed:  30.029988288879395
Total loss on epoch 6: 1150.272705
Time elapsed:  30.50545573234558
Total loss on epoch 7: 943.556152
Time elapsed:  30.29408597946167
Total loss on epoch 8: 745.822693
Time elapsed:  30.167768001556396
Total loss on epoch 9: 656.699402
Time elapsed:  29.78527855873108
Total loss on epoch 10: 611.786316
Time elapsed:  30.167571544647217
Total loss on epoch 11: 532.098938
Time elapsed:  30.246660709381104
Total loss on epoch 12: 498.133820
Time elapsed:  30.215530157089233
Total loss on epoch 13: 402.455231
Time elapsed:  30.12370204925537
Total loss on epoch 14: 345.104797
Time elapsed:  29.75166416168213
Total loss on epoch 15: 300.230530
Time elapsed:  30.37826156616211
=======TRAIN SET=======
Example 19
  x      = "2-17964087-1 romanised^name chinese^name age^at^appointment foreign^nationality portfolio^attachment govt^salary what is the portfolio^attachment of the undersecretary appointed at age age^at^appointment 48 with a chinese^name of 梁鳳儀 ?"
  y_tok  = "['select', 'portfolio^attachment', 'from', '2-17964087-1', 'where', 'age^at^appointment', '=', '48', 'and', 'chinese^name', '=', '梁鳳儀']"
  y_pred = "['select', 'portfolio^attachment', 'from', '2-17964087-1', 'where', 'age^at^appointment', '=', '48', 'and', 'chinese^name', '=', '梁鳳儀']"
Example 39
  x      = "1-24908692-5 player minutes field^goals rebounds assists steals blocks points what is the maximum number of minutes associated with exactly 70 field^goals ?"
  y_tok  = "['select', 'max', 'minutes', 'from', '1-24908692-5', 'where', 'field^goals', '=', '70']"
  y_pred = "['select', 'min', 'minutes', 'from', '1-24908692-5', 'where', 'field^goals', '=', '70']"
Example 59
  x      = "2-17751859-1 tie^no home^team score away^team date what is away^team , when home^team is " boston^united " ?"
  y_tok  = "['select', 'away^team', 'from', '2-17751859-1', 'where', 'home^team', '=', 'boston^united']"
  y_pred = "['select', 'away^team', 'from', '2-17751859-1', 'where', 'home^team', '=', 'boston^united']"
Example 79
  x      = "2-1640715-1 year starts wins top^5 top^10 poles avg.^start avg.^finish winnings position in year 1990 , how many wins had an avg finish of 35 and a start less than 1 ?"
  y_tok  = "['select', 'max', 'wins', 'from', '2-1640715-1', 'where', 'avg.^finish', '=', '35', 'and', 'year', '=', '1990', 'and', 'starts', '<', '1']"
  y_pred = "['select', 'sum', 'wins', 'from', '2-1640715-1', 'where', 'avg.^finish', '=', '35', 'and', 'year', '=', '1990']"
Example 99
  x      = "1-2731431-1 no peak location elevation^(m) prominence^(m) col^height^(m) col^location parent what is the col^location for the location of france^/^italy ?"
  y_tok  = "['select', 'col^location', 'from', '1-2731431-1', 'where', 'location', '=', 'france^/^italy']"
  y_pred = "['select', 'col^location', 'from', '1-2731431-1', 'where', 'location', '=', 'france^/^italy']"
Example 119
  x      = "1-14240688-1 year division league regular^season playoffs open^cup avg.^attendance name the playoffs for 2nd^round open^cup"
  y_tok  = "['select', 'playoffs', 'from', '1-14240688-1', 'where', 'open^cup', '=', '2nd^round']"
  y_pred = "['select', 'playoffs', 'from', '1-14240688-1', 'where', 'open^cup', '=', '2nd^round']"
Example 139
  x      = "1-1000181-1 state/territory text/background^colour format current^slogan current^series notes tell me what the notes are for south^australia"
  y_tok  = "['select', 'notes', 'from', '1-1000181-1', 'where', 'current^slogan', '=', 'south^australia']"
  y_pred = "['select', 'notes', 'from', '1-1000181-1', 'where', 'current^slogan', '=', 'south^australia']"
Example 159
  x      = "2-17323283-13 round player position nationality college/junior/club^team^(league) which round has a college/junior/club^team^(league) of hamilton^red^wings^(oha) , and a position of rw ?"
  y_tok  = "['select', 'avg', 'round', 'from', '2-17323283-13', 'where', 'college/junior/club^team^(league)', '=', 'hamilton^red^wings^(oha)', 'and', 'position', '=', 'rw']"
  y_pred = "['select', 'avg', 'round', 'from', '2-17323283-13', 'where', 'college/junior/club^team^(league)', '=', 'hamilton^red^wings^(oha)', 'and', 'position', '=', 'rw']"
Example 179
  x      = "1-20704243-5 series^# season^# title directed^by written^by original^air^date u.s.^viewers^(in^millions) name who directed season 1"
  y_tok  = "['select', 'directed^by', 'from', '1-20704243-5', 'where', 'season^#', '=', '1']"
  y_pred = "['select', 'directed^by', 'from', '1-20704243-5', 'where', 'season^#', '=', '1']"
Example 199
  x      = "1-10707142-2 rnd race^name circuit city/location date pole^position winning^driver winning^team report how many reports of races took place on date october^16 ?"
  y_tok  = "['select', 'count', 'report', 'from', '1-10707142-2', 'where', 'date', '=', 'october^16']"
  y_pred = "['select', 'count', 'report', 'from', '1-10707142-2', 'where', 'date', '=', 'october^16']"
Example 219
  x      = "2-15932367-10 date visitor score home leading^scorer record who was the leading^scorer against the visiting team bulls ?"
  y_tok  = "['select', 'leading^scorer', 'from', '2-15932367-10', 'where', 'visitor', '=', 'bulls']"
  y_pred = "['select', 'leading^scorer', 'from', '2-15932367-10', 'where', 'visitor', '=', 'bulls']"
Example 239
  x      = "2-13060397-2 city country iata icao airport name the airport with iata of cxb"
  y_tok  = "['select', 'airport', 'from', '2-13060397-2', 'where', 'iata', '=', 'cxb']"
  y_pred = "['select', 'airport', 'from', '2-13060397-2', 'where', 'iata', '=', 'cxb']"
Example 259
  x      = "2-11647327-2 marginal^ordinary^income^tax^rate single married^filing^jointly^or^qualified^widow(er) married^filing^separately head^of^household name the married^filing^separately for single of $0–$8,350"
  y_tok  = "['select', 'married^filing^separately', 'from', '2-11647327-2', 'where', 'single', '=', '$0–$8,350']"
  y_pred = "['select', 'married^filing^separately', 'from', '2-11647327-2', 'where', 'single', '=', '$0–$8,350']"
Example 279
  x      = "2-14908743-2 name birth marriage became^duchess ceased^to^be^duchess death spouse which marriage has a became^duchess of 12^december^1666 ?"
  y_tok  = "['select', 'marriage', 'from', '2-14908743-2', 'where', 'became^duchess', '=', '12^december^1666']"
  y_pred = "['select', 'marriage', 'from', '2-14908743-2', 'where', 'became^duchess', '=', '12^december^1666']"
Example 299
  x      = "2-11970261-2 country preliminaries interview swimsuit evening^gown average what is country pennsylvania 's average where the swimsuit is smaller than 9.109 and the evening^gown is smaller than 9.163 ?"
  y_tok  = "['select', 'avg', 'average', 'from', '2-11970261-2', 'where', 'swimsuit', '<', '9.109', 'and', 'country', '=', 'pennsylvania', 'and', 'evening^gown', '<', '9.163']"
  y_pred = "['select', 'avg', 'average', 'from', '2-11970261-2', 'where', 'swimsuit', '>', '9.109', 'and', 'pennsylvania', 'and', 'country', '=', 'pennsylvania']"
Example 319
  x      = "1-13456202-1 school location founded affiliation mascot division what year was north^college^hill^high^school founded ?"
  y_tok  = "['select', 'max', 'founded', 'from', '1-13456202-1', 'where', 'school', '=', 'north^college^hill^high^school']"
  y_pred = "['select', 'max', 'founded', 'from', '1-13456202-1', 'where', 'school', '=', 'north^college^hill^high^school']"
Example 339
  x      = "2-1137700-3 round grand^prix pole^position fastest^lap winning^driver winning^constructor report what is the pole^position of the belgian^grand^prix ?"
  y_tok  = "['select', 'pole^position', 'from', '2-1137700-3', 'where', 'grand^prix', '=', 'belgian^grand^prix']"
  y_pred = "['select', 'pole^position', 'from', '2-1137700-3', 'where', 'grand^prix', '=', 'belgian^grand^prix']"
Example 359
  x      = "2-14946438-46 name height weight class position hometown what is the class of the position qb , name josh^riddell ?"
  y_tok  = "['select', 'class', 'from', '2-14946438-46', 'where', 'position', '=', 'qb', 'and', 'name', '=', 'josh^riddell']"
  y_pred = "['select', 'class', 'from', '2-14946438-46', 'where', 'position', '=', 'qb', 'and', 'josh^riddell', '=', 'name']"
Example 379
  x      = "2-1178059-2 d^49^√ d^48^√ d^47^√ d^46^√ d^45^√ d^44^√ d^43^√ d^42^√ d^41^√ what is the d^48^√ with a d^46^√ with r^33^o ?"
  y_tok  = "['select', 'd^48^√', 'from', '2-1178059-2', 'where', 'd^46^√', '=', 'r^33^o']"
  y_pred = "['select', 'd^48^√', 'from', '2-1178059-2', 'where', 'd^46^√', '=', 'r^33^o']"
Example 399
  x      = "1-2062148-3 result date race venue group distance weight^(kg) jockey winner/2nd what group was the race hollindale^stakes in ?"
  y_tok  = "['select', 'group', 'from', '1-2062148-3', 'where', 'race', '=', 'hollindale^stakes']"
  y_pred = "['select', 'group', 'from', '1-2062148-3', 'where', 'race', '=', 'hollindale^stakes']"
Example 419
  x      = "2-11847478-2 date venue score result competition what is the score of the competition 2002^tiger^cup^third/fourth^place match ?"
  y_tok  = "['select', 'score', 'from', '2-11847478-2', 'where', 'competition', '=', '2002^tiger^cup^third/fourth^place']"
  y_pred = "['select', 'score', 'from', '2-11847478-2', 'where', 'competition', '=', '2002^tiger^cup^third/fourth^place']"
Example 439
  x      = "2-18369370-2 player country year(s)^won total to^par what years did the player with a total larger than 157 have wins ?"
  y_tok  = "['select', 'year(s)^won', 'from', '2-18369370-2', 'where', 'total', '>', '157']"
  y_pred = "['select', 'year(s)^won', 'from', '2-18369370-2', 'where', 'total', '>', '157']"
Example 459
  x      = "1-2624098-1 modern^english^day^name old^english^day^name english^day^name^meaning glossed^from^latin^day^name latin^day^name^meaning what is the old english name of modern^english^day^name saturday ?"
  y_tok  = "['select', 'old^english^day^name', 'from', '1-2624098-1', 'where', 'modern^english^day^name', '=', 'saturday']"
  y_pred = "['select', 'old^english^day^name', 'from', '1-2624098-1', 'where', 'modern^english^day^name', '=', 'saturday']"
Example 479
  x      = "1-15187735-6 series^ep. episode netflix segment^a segment^b segment^c segment^d what is the netflix episode when the segment^b is s^highlighter ?"
  y_tok  = "['select', 'netflix', 'from', '1-15187735-6', 'where', 'segment^b', '=', 's^highlighter']"
  y_pred = "['select', 'netflix', 'from', '1-15187735-6', 'where', 'segment^b', '=', 's^highlighter']"
Example 499
  x      = "1-24222929-4 episode^number^production^number title original^airdate time^of^airing total^viewers^on^hallmark total^viewers^on^hallmark+1 total^viewers rank^on^channel name the episode number for tasers^and^mind^erasers"
  y_tok  = "['select', 'episode^number^production^number', 'from', '1-24222929-4', 'where', 'title', '=', 'tasers^and^mind^erasers']"
  y_pred = "['select', 'episode^number^production^number', 'from', '1-24222929-4', 'where', 'title', '=', 'tasers^and^mind^erasers']"
Exact logical form matches: 350 / 500 = 0.700
Token-level accuracy: 4318 / 4805 = 0.899
=======DEV SET=======
Example 19
  x      = "2-11628153-8 season competition matches draw lost points what is the total number of matches with a loss less than 5 in the 2008/2009 season and has a draw larger than 9 ?"
  y_tok  = "['select', 'count', 'matches', 'from', '2-11628153-8', 'where', 'lost', '<', '5', 'and', 'season', '=', '2008/2009', 'and', 'draw', '>', '9']"
  y_pred = "['select', 'count', 'matches', 'from', '2-11628153-8', 'where', 'loss', '>', '5', 'and', '5', 'and', 'draw', '<', 'draw']"
Example 39
  x      = "2-12666456-2 nat. name since goals transfer^fee what the since year of the player with a transfer^fee of £^75k ?"
  y_tok  = "['select', 'since', 'from', '2-12666456-2', 'where', 'transfer^fee', '=', '£^75k']"
  y_pred = "['select', 'since', 'from', '2-12666456-2', 'where', 'transfer^fee', '=', '£^75k']"
Example 59
  x      = "2-11902503-8 game march opponent score decision record which score 's game was less than 69 when the march was bigger than 2 and the opponents were the new^york^islanders ?"
  y_tok  = "['select', 'score', 'from', '2-11902503-8', 'where', 'game', '<', '69', 'and', 'march', '>', '2', 'and', 'opponent', '=', 'new^york^islanders']"
  y_pred = "['select', 'opponents', 'from', '2-11902503-8', 'where', 'game', '=', '2', 'and', 'opponents', '<', 'march']"
Example 79
  x      = "2-11866-4 subframe^# page^# name word^# bits what is the total word count with a subframe count greater than 3 ?"
  y_tok  = "['select', 'sum', 'word^#', 'from', '2-11866-4', 'where', 'subframe^#', '>', '3']"
  y_pred = "['select', 'total', 'from', '2-11866-4', 'where', 'subframe', '<', '3']"
Example 99
  x      = "2-15400315-1 rank nation gold silver bronze total what is the nation when there is a total less than 27 , gold is less than 1 , and bronze is more than 1 ?"
  y_tok  = "['select', 'nation', 'from', '2-15400315-1', 'where', 'total', '<', '27', 'and', 'gold', '<', '1', 'and', 'bronze', '>', '1']"
  y_pred = "['select', 'avg', 'nation', 'from', '2-15400315-1', 'where', 'total', '<', '27', '>', '1', 'and', 'bronze', 'and', 'bronze', '<', 'bronze']"
Example 119
  x      = "1-15621965-8 player no. nationality position years^in^orlando school/club^team what jersey number did player al^harrington wear"
  y_tok  = "['select', 'max', 'no.', 'from', '1-15621965-8', 'where', 'player', '=', 'al^harrington']"
  y_pred = "['select', 'school/club^team', 'from', '1-15621965-8', 'where', 'player', '=', 'wear']"
Exact logical form matches: 33 / 125 = 0.264
Token-level accuracy: 800 / 1237 = 0.647
