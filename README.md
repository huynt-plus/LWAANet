# ILWAANet
## A Interactive Lexicon-Aware Word-Aspect Attention Network

The implementation of the paper: "*LWAANet: A Interactive Lexicon-Aware Word-Aspect Attention Network for Aspect-level Sentiment Classification on Social Networking*"

### Word & Lexicon Embeddings

Creating *vec* folder and copying *Glove* and *Lexicon* embeddings into this folder.

[Glove 42B](https://nlp.stanford.edu/projects/glove/) is used for word embeddings

[Preconstructed-Lexicon Embeddings](https://drive.google.com/open?id=1CB1dyhsRGMk0El9ileUgLk49jepHoPjY)

### Pre-trained Models
[Pre-trained models](https://drive.google.com/open?id=1nGXusK8_wVX5n1oed81Us2818qZGKr_x)

### Hyper-parameters

*Random search* is used for choosing *Hyper-parameters*

| Hyper-parameters| #Latop #Restaurant #Twitter |
| ----------------|:---------------------------:|
| Mini-batch size | 100                         |
| Embedding dim   | 300                         |
| Lexicon dim     | 16                          |
| Epochs          | 300                         |
| RNN dim         | 100                         |
| Learning rate   | 2e-3                        |
| Dropout rate    | 0.5                         |
| L2 Constrain    | 1e-5                        |


### Running Models

Running AN model:

```
python AN.py
```

Running WAAN model:

```
python WAAN.py
```

Running LWAAN model:

```
python LWAAN.py
```

Running ILWAAN model:

```
python ILWAAN.py
```

