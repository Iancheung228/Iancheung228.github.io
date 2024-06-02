---
layout: post
title: Barebone transformer walkthrough
date: 2024-04-20
---

## Goal of this post:
This post documents my written notes that helped me better understand the core mechanism of the [transformer paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). I have benefited tremendously from [Andrej Karparthy's youtube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s)  and his [code which could be found here](https://github.com/karpathy/ng-video-lecture). Here is the [notebook](https://colab.research.google.com/drive/1nlZDMEVV6mAGixBIm6t2ZDJMGpqvcWf5#scrollTo=PfTS95CgJsUs) with my own annotation for reference. To focus on the core aspect of the transformer, this post will go through one training iteration on a single piece of data. Of course, in practice, we usually train on a batch of data.

## The problem
A common problem a transformer model can solve is predicting the next word that appears given a sequence of input.

In Natural Language Processing (NLP) a single piece of training data is usually a sequence of words (tokens to be exact). The length of this sequence is determined by a hyperparameter we choose, denoted as the **context_length**. A longer context_length means the model have access to words from further back in time when attempting to predict the next word.

### Data base of vocabulary
In addition, we also have access to a pretrained database of vocabulary. You can think of this database of vocabulary stored as a matrix that contains every word in the Oxford English dictionary. However, unlike the Oxford English Dictionary, the definition of a word is defined by a vector of dimension **embed_dim**, instead of defining a word with other English words. 

Considered alone, each of these vectors already contains valuable information about the word. However, the meaning of a word changes based on the context. For example, the word "Harry" refers to 2 completely different people when the article is about the Pricne versus the fictional character. The attention mechanism is designed to learn a richer vector representation of each word, based on the context where this word appears. **That is, the database of vocabulary contains non-contextual information while the transformer aims to learn contextual information.**

### Positional encoding vector
The position in which the words appear in the sequence is also important. We usually add a positional encoding vector to each word in the sequence. For the scope of this post, we can take it for granted that this positional encoding vector will give the model all the information it needs to figure out the position of each word within our sequence. The original transformer paper uses a deterministic sinusoidal function as the positional encoding vector.

With these preliminaries in mind, let's dive into an example by going through one training sequence.

## 0) Data preprocessing:
Suppose our raw input to our transformer model is the sequence: ** The early bird eats the worm. **


These are the following steps to preprocess our data:

<ol>
  <li> Find the index for each word in our database of vocabulary</li>
  <li> Retrieve the non-contextual word embedding vector</li>
  <li> Incorporate positional information by simple vector addition</li>
  <li> Reshape (optional)</li>
</ol>


<img width="827" alt="Screenshot 2024-04-27 at 6 54 07 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/870b077e-d15d-465e-bd54-62ad0ab19326">

<img width="815" alt="Screenshot 2024-04-27 at 6 54 18 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/968d8d0a-c61e-4a47-b8c7-53ddf6c182a1">

At this point, we have preprocessed the raw input, which is now ready to be fed into our transformer model.

## 1a) Single attention head:
The attention head is where the model learns the richer contextual embedding for each word. Specifically, the model learns how much, the other words in the sequence should influence the meaning of the current word. On a high level, this involves a matrix multiplication of the key, query and value matrices. There are many great resources on interpreting these mechanisms, like this post by [Luis Serrano](https://cohere.com/blog/what-is-attention-in-language-models).


| Variable | Description | In this post |
| --- | --- | --- |
| `Batch_size` | The number of training examples utilized in one iteration | 1 (for simplicity) |
| `Context_len` | The length of the input sequence to the model | 6 words |
| `Embed_dim` | Dimension of the word representation in our database of vocab | 128 |
| `Num_head` | Number of attention heads  | 2 |
| `Head_size` | Dimension (equals to (Embed_dim / Num_head)) | 64 |

### Notation for our diagrams to come:
- X = post-processed data (output of step 0)
- Wq = Query matrix
- Wk = Key matrix 
- Wv = Value matrix 


<img width="852" alt="Screenshot 2024-04-27 at 6 54 34 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/f682f05e-35a6-4b42-82bc-b8e52b1b5a71">



### Breakdown of the above single-attention head diagram

I had always been confused about the dimensionality of these matrix operations. I have deliberately over-annotated as it helped me better understand the full picture, and that my future self might come back to thank me.
<ol>
  <li> Matrix multiplication between query and key matrices, the output size will be (context_len, context_len) </li>
  <li> Apply softmax to get a probability distribution that sums to 1. The way to interpret the matrix is that: each row corresponds to a word, and we can tell how much other words in the sequence influence the meaning of the current word based on the weights assigned. (We have omitted the scaled dot product attention version in this post) </li>
  <li> Matrix multiply with the value matrix to get what we call in this post as: the attention vector, which has dimension (context_len, embed_dim)</li>
  <li> Matrix multiply with a linear layer </li>
</ol>

(Jumping slightly ahead, as we will see, the output of this step is our contextual word embedding, and we will add it to our original non-contextual word embedding in the residual layer.)

## 1b) Multiple attention heads
In practice, however, we do not use a single attention head. There is a huge incentive to learn many of such single-attention heads (parameterized by the Query, Key and Value matrices) so that each head will learn a different aspect of the complex relationship in the original sequence. With multiple heads, we now have an additional hyperparameter denoted: **num_head**. For this post, let's set it to 2 for simplicity.

A shortcoming of the single-attention head is likely that it is not very expressive when we only learn one big comprehensive set of Query, Key and Value matrices, where you can think of it as a "generalist". With multiple heads, we are facilitating the environment for more "specialists". But again this is only intuition and speculation.

**Note: Requires Embed_dim % Num_head = 0**

The multiple attention heads operate largely similarly to the single head except that the size of the Query, Key and Value Matrices have now shrunk. Specifically, we partition the **1)Query**, **2)Key**, and **3)Value** matrices into num_head parts. In the diagram, each of the 3, full-sized matrices is partitioned into **a) green boxed** and **b) pink boxed** portions. Virtually, we now have 2 single-head attention blocks, where the parameters in each single-head are learned independently. Note the attention matrix output from each single-head attention is proportionally smaller. We have an extra final step in between **step 3** and **step 4** to concatenate all these attention matrices into one of the original size.


<img width="1010" alt="Screenshot 2024-04-20 at 6 21 30 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/5008eedb-f438-4e7c-a8dc-ac2dfe4d37ac">

<img width="1016" alt="Screenshot 2024-05-06 at 8 43 17 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/ba46fe88-982c-4c03-854d-7217518604e9">


Aside: Note that in multiple attention heads, the model has the same number of parameters to learn as in a single attention head. Multi-head self-attention is no more expensive than single-head due to the low-rank property.

We have now fully explained the following diagram from the original paper.
![Screenshot 2024-06-01 at 5 26 48 PM](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/a58abe97-f1ee-4461-80f1-41e55bbd6393)


## 2) Residual connection

Recall, the final output from the previous step is the attention matrix that has undergone a linear transformation. Next, we will add our initial post-processed data, X (output of step 0). Note both matrices have the same dimensionality of (context_len, embed_dim).

Recall, X contains the non-contextual information, and now we are adding the contextual information on top. A question worth pondering is why we decided to add those 2 vectors together instead of concatenating so the signals of both vectors remain intact?

## 3) Feedforward layer
We then take the output from step 2 and pass it through a feedforward layer. The significance of this step is to introduce **non-linearity** to our transformer model.

Specifically, this layer takes the input which lives in embed_dim to 4*embed_dim and then projects it back to embed_dim. The output of this layer has dimension (context_length, embed_dim)

## 4) Language_model_head_linear_layer + softmax
Finally, recall that our original goal has always been to predict the next word, given the input sequence.  Sensibly, we would want our final layer to output a vector of probability for the next word to come. To accomplish this, we:

<ol>
  <li> Pass it through a linear layer which takes an input of dimension (context_len, embed_dim) and returns a matrix of dimension (context_length, vocab_size) </li>
  <li> Apply softmax so that it is a probability distribution </li>
</ol>

The output matrix could be interpreted as a probability distribution that describes the probability of any words in our database of vocabulary being the next predicted word.






