---
layout: post
title: transformer
date: 2024-04-20
---

## Introduction

Goal of post:
1) easy visual walkthrough of transformer with reference to functional code
2) TBD


Let's walk through what training 1 iteration on a single piece of data in a transformer looks like. In practice, we will of course train on a batch of data.

In NLP a single piece of training data is usually a sequence of words (tokens to be exact). The length of this sequence is determined by a hyperparameter we choose, called the context_length. A longer context_length means the model has a larger RAM.

We also have access to a pretrained database of vocabulary. You can think of this database of vocabulary as a matrix that contains every word in the Oxford English dictionary. Unlike the Oxford English Dictionary, however, instead of defining a word with other English words, we define a word by a vector with dimension length embed_dim. Each one of these vectors contains valuable information about the word when considered alone, language is a very complex system where the meaning of a word in a sentence changes based on neighbouring words, or as we might call it, the context. Transformer aims to learn a richer vector representation, based on the context where this word appears in. That is the database of vocabulary contains non-contextual information while Transformer aims to learn contextual information.

## Data preprocessing:
Suppose our raw input to our transformer model is the sequence: The early bird eats the worm.
These are the following steps:

<ol>
  <li>find index for each word</li>
  <li> retrieve the non contextual word embedding from our vocab</li>
  <li> incorporate positional information as we like to keep track of the order of words</li>
  <li> reshape</li>
</ol>


<img width="813" alt="Screenshot 2024-04-20 at 5 53 15 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/2c4dab14-4d8e-414f-b024-dced839813f7">

<img width="781" alt="Screenshot 2024-04-20 at 6 34 22 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/585bdd51-167a-41a7-8c38-2a343cfd5e0c">
Now the raw input is preprocessed and is ready to be fed into our transformer model.

## Single attention head:
The attention head is where the magic happens. This is where the model learns how much, the other words in the sequence should influence the meaning of the current word. On a high level, this involves the matrix multiplication of the key, query and value matrices. Many intuitions are provided in the following sources for interested readers.

definition
Batch_size = 1
Context_len =
Embed_dim =
Head_size =
Num_head =

- X = 1
- Wq = 2
- Wk =3 
- Wv = 4 

<img width="854" alt="Screenshot 2024-04-20 at 6 15 05 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/c8adc98c-7bc0-41ae-98f8-eaa2952c2c1f">


Elementwise non-linearity
## Multiple attention heads
Allows each head to learn different relationships between the sequence.
Requires Embed_dim % Num_head = 0

We split the 1)Query, 2)Key, and 3)Value matrices into num_head parts. The parameters in each of the multihead attentions are learned independently. Note the attention matrix from each multi-head attention is proportionally smaller. We would concatenate all these attention matrices at the end.

Note that in multiple attention heads, the model has the same number of parameters to learn as in a single attention head. Multi-head self-attention is no more expensive than single-head due to this low-rank property.

The attention block has dimension (context_len, embed_dim)

<img width="1010" alt="Screenshot 2024-04-20 at 6 21 30 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/5008eedb-f438-4e7c-a8dc-ac2dfe4d37ac">

<img width="781" alt="Screenshot 2024-04-20 at 6 34 22 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/d2b4dd9a-6763-4b14-8555-0196e6a3561b">



## Self-projection layer
(embed_dim, embed_dim)

## residual layer
x = x + (context_len, embed_dim)
## feedforward layer
takes it from embed_dim to 4*embed_dim then back to embed_dim

## logit layer
takes (context_len, embed_dim) to (context_len, vocab_size)






