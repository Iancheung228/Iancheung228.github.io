---
layout: post
title: Barebone transformer walkthrough
date: 2024-04-20
---

## Goal of this post:
The goal of this post is to provide a gentle visual introduction to each core mechanism within a transformer model. To focus on the core aspect of the transformer, this post will go through one training iteration on a single piece of data. In practice, we usually train on a batch of data.

## Introduction to the NLP problem
A common problem a transformer model can solve is predicting the next word that appears given a sequence of input.

In Natural Language Processing (NLP) a single piece of training data is usually a sequence of words (tokens to be exact). The length of this sequence is determined by a hyperparameter we choose, denoted as the **context_length**. A longer context_length means the model have access to words from further back in time when attempting to predict the next word.

### Data base of vocabulary
In addition, we also have access to a pretrained database of vocabulary. You can think of this database of vocabulary as a matrix that contains every word in the Oxford English dictionary. However, unlike the Oxford English Dictionary, we 'define' each word by a vector of dimension **embed_dim**, instead of defining a word with other English words. 

Considered alone, each of these vectors already contains valuable information about the word. However, the meaning of a word changes based on the context. For example, the word "Harry" refers to different people when the article is about the Pricne vs the fictional character. Transformer is designed to learn a richer vector representation of each word, based on the context where this word appears. **That is, the database of vocabulary contains non-contextual information while the transformer aims to learn contextual information.**

### Positional encoding vector
The position in which the words appear in the sequence is also important. We usually add something called a positional encoding vector to each word in the sequence. For the scope of this post, we can take it for granted that this positional encoding vector will give the model all the information it needs to figure out the position of each word within our sequence. For general purposes, we can assume the positional encoding vector is deterministic and could be retrieved from a position embedding table.

## 0) Data preprocessing:
Suppose our raw input to our transformer model is the sequence: The early bird eats the worm.


These are the following steps to preprocess our data:

<ol>
  <li> Find the index for each word in our database of vocabulary</li>
  <li> Retrieve the non-contextual word embedding </li>
  <li> Incorporate positional information by simple vector addition</li>
  <li> Reshape (optional)</li>
</ol>


<img width="827" alt="Screenshot 2024-04-27 at 6 54 07 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/870b077e-d15d-465e-bd54-62ad0ab19326">

<img width="815" alt="Screenshot 2024-04-27 at 6 54 18 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/968d8d0a-c61e-4a47-b8c7-53ddf6c182a1">

At this point, we have preprocessed the raw input and it is now ready to be fed into our transformer model.

## 1a) Single attention head:
The attention head is where the model learns the richer contextual embedding for each word. Specifically, the model learns how much, the other words in the sequence should influence the meaning of the current word. On a high level, this involves a matrix multiplication of the key, query and value matrices. There are many great resources on interpreting these mechanisms, like :.


| Variable | Description | In this post |
| --- | --- | --- |
| `Batch_size` | The number of training examples utilized in one iteration | 1 (for simplicity) |
| `Context_len` | The length of the input sequence to the model | 6 words |
| `Embed_dim` | Dimension of the word representation in our database of vocab | 128 |
| `Num_head` | Number of attention heads  | 2 |
| `Head_size` | dimension equals to (Embed_dim / Num_head) | 64 |

### Shorthand notation:
- X = post-processed data
- Wq = Query matrix
- Wk = Key matrix 
- Wv = Value matrix 


<img width="852" alt="Screenshot 2024-04-27 at 6 54 34 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/f682f05e-35a6-4b42-82bc-b8e52b1b5a71">



### Breakdown of the single-attention head
<ol>
  <li> Matrix multiplication between query and key matrices, the output size will be (context_len, context_len) </li>
  <li> Apply softmax to get a probability distribution that sums to 1. The way to interpret the matrix is that: each row corresponds to a word, and we can tell how much other words in the sequence modify the meaning of the current word based on the weights assigned. (For simplicity, we have omitted the scaled dot product attention version in this post) </li>
  <li> Matrix multiply with the value matrix to get attention vector which has size (context_len, embed_dim)</li>
  <li> Matrix multiply with a linear layer </li>
</ol>

Vector addition this attention vector to our original non-contextual word embedding in residual

## 1b) Multiple attention heads
In practice, there is a huge incentive to learn many of such single-attention heads (parameterized by the Query, Key and Value matrices) such that each head will learn a different aspect of the complex relationship in the original sequence. With multiple heads, we now have an additional hyperparameter denoted: **num_head**. For this post, let's set it to 2 for simplicity.

**Note: Requires Embed_dim % Num_head = 0**

The multiple attention heads operate largely similarly to the single head except that the size of the Query, Key and Value Matrices have now shrunk. Specifically, we partition the **1)Query**, **2)Key**, and **3)Value** matrices into num_head parts. In the diagram, each of the 3 full-sized matrices is partitioned into **a)** green and **b)** pink portions. The parameters in each of the single-head attention are learned independently. Note the attention matrix output from each single-head attention is proportionally smaller. We have an extra step in between step **3** and **4** to concatenate all these attention matrices at the end.


<img width="1010" alt="Screenshot 2024-04-20 at 6 21 30 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/5008eedb-f438-4e7c-a8dc-ac2dfe4d37ac">

<img width="1001" alt="Screenshot 2024-04-28 at 4 20 39 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/4b92d6bf-a869-4d08-9fda-05c0af7a80b7">



Note that in multiple attention heads, the model has the same number of parameters to learn as in a single attention head. Multi-head self-attention is no more expensive than single-head due to this low-rank property.


## Residual connection

Recall, the final output of the multiple attention heads layer is an attention matrix that has undergone a linear transformation layer. We will take that and do matrix addition with our initial post-processed data (matrix X). Note both matrices have the same dimensionality of (context_len, embed_dim).

As a quick recap, X contains the non-contextual vector embedding of each word, we are modifying our understanding of each word by adding our new contextual vector embedding, based on what the multi-attention head has learnt.

## 2) Feedforward layer
We then take the output from the last step and pass it through a feedforward layer. The importance of this step is to introduce **non-linearity** to our transformer model.


Specifically, this layer takes the input which lives in embed_dim to 4*embed_dim then project back to embed_dim.

## 3) Language_model_head_linear_layer + softmax
Finally, recall that our goal has always been to predict the next word, given the entire sequence.  Sensibly, we would want our final layer to output a vector of size vocab_size. We will apply softmax to this vector so that we can interpret it as a probability distribution that describes the probability of any words in our database of vocabulary being the next predicted word.

Practically, instead of being a vector of size vocab_size, we actually work with a matrix of size (context_len, vocab_size). That is for each word in our sequence, we are predicting the next word that comes after. Additional information could be found if you search for causal attention head.


Specifically, this layer takes an input of dimension (context_len, embed_dim) and applies a linear layer and softmax to output a matrix of dimension (context_len, vocab_size).


## Walk through of code
token is character
vocab size is 65

lookup character to index using char2idx

B is short hand for batch
T is short hand for context_len
C is short hand for embed_dim





