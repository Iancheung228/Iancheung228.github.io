---
layout: post
title: transformer
date: 2024-04-20
---

## Goal of this post:
1) Visual walkthrough of coding up a barebone transformer

The majority of this post will be dedicated to providing a gentle visual introduction to each core mechanism within a transformer model. To focus on the core aspect of the transformer, this post will go through what training 1 iteration on a single piece of data in a transformer looks like. In practice, we will of course train on a batch of data.

## Introduction to the NLP problem
A common problem a transformer model can solve is predicting the next word that appears given a sequence of input.

In NLP a single piece of training data is usually a sequence of words (tokens to be exact). The length of this sequence is determined by a hyperparameter we choose, called the context_length. A longer context_length means the model is now able to leverage words from the distant past when predicting the next word.

We also have access to a pretrained database of vocabulary. You can think of this database of vocabulary as a matrix that contains every word in the Oxford English dictionary. Unlike the Oxford English dictionary however, instead of defining a word with other English words, we 'define' a word by a vector with dimension **embed_dim**. Each vector contains valuable information about the word when considered alone. However, language is a very complex system where the meaning of a word changes based on how it is used in a sentence. Transformer aims to learn a richer vector representation of each word, based on the context where this word appears. That is the database of vocabulary contains non-contextual information while the transformer aims to learn contextual information.

## 0) Data preprocessing:
Suppose our raw input to our transformer model is the sequence: The early bird eats the worm.
These are the following steps to preprocess our data:

<ol>
  <li> Find index for each word in our database of vocabulary</li>
  <li> Retrieve the non-contextual word embedding </li>
  <li> Incorporate positional information by simple vector addition</li>
  <li> Reshape (optional)</li>
</ol>


<img width="813" alt="Screenshot 2024-04-20 at 5 53 15 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/2c4dab14-4d8e-414f-b024-dced839813f7">

<img width="791" alt="Screenshot 2024-04-21 at 11 23 48 AM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/704abffa-dde4-45fb-afa1-d55a29d57197">



Now the raw input is preprocessed and is ready to be fed into our transformer model.

## Single attention head:
The attention head is where the magic happens. This is where the model learns how much, the other words in the sequence should influence the meaning of the current word. On a high level, this involves the matrix multiplication of the key, query and value matrices. There are many great resources on interpreting these mechanisms, like :.

definition
Batch_size = 1
Context_len = 6
Embed_dim = 128
Head_size = 64
Num_head = 2

- X = post-processed data
- Wq = Query matrix
- Wk = Key matrix 
- Wv = Value matrix 

<img width="816" alt="Screenshot 2024-04-21 at 10 43 21 AM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/d69b6cda-5cde-41f9-a6c7-844cd2eaf254">



Breakdown of the single-attention head
<ol>
  <li> Matrix multiplication between query and key matrices, the output size will be (context_len, context_len) </li>
  <li> Apply softmax to get a probability distribution that sums to 1. From this matrix, we can read off how much weight is given to the other words in the context. (we omit the scaled dot product attention in this post to keep it simple) </li>
  <li> Matrix multiply with value matrix to get attention vector which has size (context_len, embed_dim)</li>
  <li> Matrix multiply with a linear layer </li>
</ol>

Vector addition this attention vector to our original non-contextual word embedding in residual

Elementwise non-linearity

## 1a) Multiple attention heads
In practice, we are motivated to learn many of such single-attention heads (parameterized by the Query, Key and Value matrices). Each head will learn a different aspect of the complex relationship in the original sequence. We now have an additional hyperparameter which is **num_head**. For this post, let's set it to 2.

**Requires Embed_dim % Num_head = 0**


Specifically, we partition the **1)Query**, **2)Key**, and **3)Value** matrices into num_head parts. In the diagram each of the 3 original sized matrices are partitioned into green and pink portions. The parameters in each of the single-head attention are learned independently. Note the attention matrix outputed from each single-head attention is proportionally smaller. We have an extra step to concatenate all these attention matrices at the end.


<img width="1010" alt="Screenshot 2024-04-20 at 6 21 30 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/5008eedb-f438-4e7c-a8dc-ac2dfe4d37ac">

<img width="996" alt="Screenshot 2024-04-21 at 10 43 40 AM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/75ea073b-99ed-49f7-988b-e53980f6cf33">

Note that in multiple attention heads, the model has the same number of parameters to learn as in a single attention head. Multi-head self-attention is no more expensive than single-head due to this low-rank property.


## residual layer

x = x + (context_len, embed_dim)

We add
## 2) feedforward layer
takes it from embed_dim to 4*embed_dim then back to embed_dim

## logit layer
takes (context_len, embed_dim) to (context_len, vocab_size)






