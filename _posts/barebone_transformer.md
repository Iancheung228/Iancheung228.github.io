---
title: transformer
date: 2024-04-01
---

## Introduction

Goal of post:
1) easy visual walkthrough of transformer with reference to functional code
2) TBD


Let's walk through what training one single piece of training data in a transformer looks like.

In NLP a single piece of training data is usually a sequence of words (tokens to be exact). The length of this sequence is determined by a hyperparameter we choose, called the context_length. We also have access to a pretrained database of vocabulary. You can think of this vocab database as a matrix that contains every word in the Oxford English dictionary. Unlike the Oxford English Dictionary, however, instead of defining a word by other words, we define a word by a vector of length embed_dim (non-contextual information). It is important to note that although this vector contains valuable information about the word when considered along, language is a very complex system where the meaning of a word in a sentence changes based on neighbouring words, or as we might call it, the context. Transformer aims to learn a richer vector representation, based on the context where this word appears in.

## data manipulation
The early bird eats the worm.
10892 11257 10202 4514 10892  8438
These are the corresponding indexes in our vocabulary

We then literally look up the vector at that index. Again this vector has dimension embed_dim.

We would also like to keep track of the order of the words in the sequence. We accomplish this by adding a positional encoding vector to each word.

## Single attention head
The attention head is where the magic happens. This is where the model learns how much, the other words in the sequence should influence the meaning of the current word. This involves the matrix multiplication of the key, query and value matrices. Many intuitions are provided in the following sources.

definition
Batch_size = 1
Context_len =
Embed_dim =
Head_size =
Num_head =



## Multiple attention heads

Note that in multiple attention heads, the model has the same number of parameters to learn as in a single attention head.

## Self-projection layer

## feedforward layer







