# PointerSQL_pytorch

- main branch: follow setup in the paper [POINTING OUT SQL QUERIES FROM TEXT](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/nl2prog.pdf)
  - put pretrained word vectors in `/data`, uploaded to (Google Drive)[]
  - Problem: the paper said the the pretrained vectors are set untrainable to avoid over-fitting, but many special tokens in SQL will be labeld `UNK`

- no_pretrained brance: only data changed, the rest are the same as project 2
  - no need to put pretrained word vectors
