---
title: BatchNorm
date: 2024-04-01
---

## Introduction
Batch Normalization (BN) has been empirically shown to allow deep neural nets (NN) to train faster and more stably (less sensitive to the choice of initialization). However, the exact theoretical benefit of the batch norm layer has always been a topic of debate. The main difficulty perhaps comes from the fact that a NN has many moving parts so it is hard to put your finger down on the exact root problem the BN layer solves, let alone whether BN is the unique mechanism that solves it. The original paper from 2015 [^1] attributes the success to resolving the problem of internal covariate shift (ICS). In 2019, there is a new paper [^2] that argues that instead of ICS, BN's true benefit  is in making the optimization landscape smoother.


[^1]: [Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", 2015.](https://arxiv.org/abs/1502.03167)
[^2]: [Shibani Santurkar, "How Does Batch Normalization Help Optimization", 2019.](https://arxiv.org/pdf/1805.11604.pdf)

In this article, we will go through the list of items:
<ol>
  <li>What is batch norm and implement a simple neuron net with batch norm layer</li>
  <li> Discussion of first benefit: preventing dead or saturated units</li>
  <li> Discussion of second benefit: Resolving the Internal Covariate Shift problem (and why it is not entirely true) (2015 paper)</li>
  <li> Discussion of third benefit: Smoothening the loss landscape (2019 paper)</li>
</ol>

<br/><br/>

## Formal definition of batch normalization 
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during the training phase. Specifically, the batch norm layer converts the first two moments of the neuron's input (denoted as y) to 0 mean and unit variance. The mean and standard deviation are calculated based on the current batch.

<img width="729" alt="Screenshot 2024-02-25 at 11 00 54 AM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/312f5c2e-0dad-49fd-8882-384737fdc998">

In practice, a BN layer includes 2 learnable parameters (in green) for the output mean and variance. This is done to give back the expressive power of the original network. i.e. the NN is free to choose whether a non-zero mean is better suited for each layer.

<br/><br/>

Pseudocode of PyTorch code for a 2-layered neuron net with batch norm for binary classification task
```
#Suppose we have 10000 training data points each with dim 5
#i.e Xtr = (10000,5), Ytr(10000,1)
batch_size = 32
# minibatch construction
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# defining our layers of 100 neurons 
X_dim = Xb.shape[1]                   
n_hidden = 100              
W1 = torch.randn((X_dim , n_hidden))
W2 = torch.randn((n_hidden, 2))
b2 = torch.randn(2)

# BatchNorm parameters
bngain = torch.ones((1, n_hidden))          # in green in the diagram of BN def 
bnbias = torch.zeros((1, n_hidden))         # in green in the diagram of BN def 
bnmean_running = torch.zeros((1, n_hidden)) # not trained during back prop (used at inference)
bnstd_running = torch.ones((1, n_hidden))   # not trained during back prop (used at inference)

parameters = [W1, W2, b2, bngain, bnbias]
for p in parameters:
  p.requires_grad = True


Our Neural Network -------------------------------------------------------------
# Linear layer
pre_act = Xb @ W1
## We don't need a bias term as the BN layer will get rid of the bias here
## pre_act has shape (batch_size, 100) where 100 is the number of neurons.
## For each neuron, we want to find the mean and std across all 32 training examples.

# BatchNorm layer
bnmeani = pre_act.mean(0, keepdim=True)
bnstdi = pre_act.std(0, keepdim=True)
pre_act = bngain * (pre_act - bnmeani) / bnstdi + bnbias # each neuron will be unit gaussian for this batch of data
# running mean to use in validation, these running also get updated in the training phase, but these do not require any gradient
with torch.no_grad():
  bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
  bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

# Non-linearity
h = torch.tanh(pre_act)            # non linear layer
logits = h @ W2 + b2               # output layer
# -------------------------------------------------------------
loss = F.cross_entropy(logits, Yb) # loss function
```

## Discussion of first benefit: Preventing dead or saturated units

Many activation functions used in a NN, including Tanh are a so-called squashing function. Squashing functions like Tanh removes information from the original input. Specifically, if the input value (in absolute value) is too big, Tanh will return approximatly 1 or -1, which corresponds to the flat region in the tail ends of the function. From a gradient point of view, if the neuron lands on the flat region, the gradient would be 0, and virtually this will stop any gradient flowing through this neuron when updating the neuron's weight parameter. In other words, if the neuron's output (in absolute value) is too big, no matter how you perturb the value of the neuron, it will not have any impact on the final loss, and hence the neuron will not get updated. We call this a dead neuron.

By adding a batch norm layer before the activation layer, we would force the input to take on a zero mean and unit variance distribution which greatly prevents neurons landing on flat regions.


## Discussion of second benefit: Resolving the Internal Covariate Shift problem (and why it is not entirely true) (2015 paper)

ICS is closely related to the concept of covariate shift, which is when the input-data distribution shifts over time and hence making a trained model obsolete. For example, we could use pre-covid's stock data to train a stock price prediction model, however, chances are the model will not be effective in predicting returns for post-COVID era, as the data distribution has changed substantially.

Now, adding the word "Internal" before "covariate shift", describes a closely related phenomenon that occurs in the training of a neural network, where the distribution of input for an individual layer, changes due to the update of the previous layers' weights.

Let me introduce a useful framework for thinking about neuron nets. We can think of a neural net as a function parameterized by weights. This function takes a given datapoint as input and outputs a prediction. The training of neural nets can be seen as solving an optimization problem, where we attempt to learn the optimal weight for the function in order to map our datapoint to the true label, as closely as possible. 

In fact, we can break down the original optimization problem into solving a series of smaller, sequential optimization problems at a layer level. That is, each layer is also a function that takes in an input (received from the previous layer) and produces an output (feeds to the next layer). The layer-wise optimization problem has a similar flavour, where we try to find good weights that map the input to the desired output. Precisely speaking, the input here refers to the output from the previous layer, and the desired output is related to the accumulation of the gradient w.r.t the final loss from the later layers.


The ICS occurs when the input for the layer (output of the previous layer) changes drastically (due to weight update in the previous epoch) in every iteration of the training procedure.

Let's walk through an example for more clarity.

<br/><br/>
### Example
Consider a neural network with 3 layers (each layer has 1 neuron) with no nonlinearity. Let's walk through how the backpropagation algorithm will update the weights of the 3 layers. We will do this for 2 epochs **a)** $$i-1$$ and **b)** $$i$$.
#### Notation
* $$w_c$$ denotes the weight of neuron c

* $$z_c$$ denotes the output of neuron c

* L denotes the loss (common choice is $${\lvert\hat{y} - y \rvert}^2$$)

* $$\alpha $$ is the learning rate


<br/><br/>
#### Groundwork
The update rule for weight at neuron c is:
$$ w_c^{new} \leftarrow w_c^{old} - \alpha \color{red}{\frac{\delta L}{\delta w_c}}$$

Taking a closer look at the gradient term, we can rewrite it as:
$$ \color{red}{\frac{\delta L}{\delta w_c}} = \frac{\delta L}{\delta z_c} \frac{\delta z_c}{\delta w_c}$$

Recall, we also know that the output of neuron c is simply the dot product of the weight of neuron c and the output of neuron b:
$$z_c = w_c*z_b$$. This means that taking the derivative of $$ w_c $$ w.r.t $$ z_c $$ is $$z_b$$.

Incorporating what we discussed, we arrive at
**$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$** which we will use to update the neuron's weight.

Importantly, we see that the update of neuron c's weight depends on the output (hence the weight) of the previous neuron.


#### Stepping through an example
With the groundwork established, you can go through the diagram below by yourself, where the backpropagation algorithm updates based on the order of  **1c), 1b), 1a), 2c)** 

![IMG_10186360727C-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/e9991454-6224-4daf-8298-1740b1ebe9c9)

##### Commentary:

We see that in step **1c**) the output of neuron b, $$z_b$$, is a function of $$w_b^{t-2}$$, $$w_a^{t-2}$$ and $$x_{t-1}$$

At the end of step **1)**, we have updated the weight of all 3 neurons.

We see that in step **2c)** the output of neuron b, $$z_b$$, is a function of $$w_b^{t-1}$$, $$w_a^{t-1}$$ and $$x_{t}$$

**Importantly, $$w_b^{t-2} \neq w_b^{t-1}$$ and $$w_a^{t-2} \neq w_a^{t-1}$$. Hence, we see that in step 2c), the input to layer c  $$:= z_b ^{(t)}$$ would have a completely different distribution than the corresponding $$z_b^{(t-1)}$$ in step 1a). Even in the extreme case where the input data $$x_t = x_{t-1}$$! This is internal covariate shift.**


*Here we make the simplifying assumption that at each iteration we only train on 1 data point, in practice, we train on a mini-batch and the idea of distribution applies* 

*Aside: when updating a neural network within one training iteration, we have to first update the $${k+1}^{th}$$ layer, before we can update the $$k^{th}$$ layer (take it for granted if you are not familiar), this reverse order of update is dictated by the backpropagation algorithm.* 


The subtle thing is, when we update the weight of layer c at iteration i using the equation: $$\frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$, the output for node b, $$z_b$$, actually still uses the learned weights from the previous iteration i-1 . (Recall $$w_b$$ is optimized for the data point at iteration i-1 $$(x_{i-1},y_{i-1})$$.) In other words, at iteration i, the input of layer c's optimization problem assumes the distribution of $$x_i$$ is the same as the distribution of $$x_{i-1}$$ by using a "stale" weight $$w_b$$ at this exact point in the backpropagation algorithm. If $$x_{i-1}$$ has a significantly different value than $$x_i$$, then $$w_b^{i-1}$$ will not be good for minimizing the loss for the data point $$x_i$$.

This ICS problem was believed to be a huge problem if left unaddressed and the authors of the original paper hence suggested adding a BN layer after each layer of the original NN.


### Summary of Internal Covariate Shift

| Problem: | occurs at: | drastic change in: | 
| --- | --- | --- |
| **Covariate Shift** | train vs test time | input to model |
| **Internal Covariate Shift** | epoch i-1 vs epoch i | input to layer |


<br/><br/>
## Counter argument 1) 2019 Experiment shows contradictory results
The authors of the 2019 paper conducted a simple experiment where they intentionally added noise after the BN layer (we call it the Batch norm plus noise model). 

The rationale is: If the performance gain of the neural net is indeed attributable to resolving the internal covariate shift, adding back noise **after** the BN layer will erase any of the benefits.

The result is that they found that the Batch norm plus noise model has largely similar performance compared with the Batch norm model. This suggests that BN's main benefit does not come from resolving the ICS.

## Counter argument 2) Actual placement of BN layer is before the activation layer
Recall that ICS is the issue where the input distribution to a layer changes drastically between consecutive epochs. In theory, to resolve ICS, we would apply BN layer **right before** feeding the input to the next layer. This is not the case in practice, where the BN is placed before the activation layer, which is then fed as input to the next layer. This means we are **not guaranteed** that the input distribution after the activation layer is still non-zero mean and unit variance.

## Discussion of third benefit: Smoothening the loss landscape in 2 manifestations (2019 paper) 
The 2019 paper argues BN's main benefit is in reparameterizing the underlying optimization problem and smoothening the loss landscape. This benefit comes largely in 2 manifestations and heavily utilizes the concept of Liptschitzness.

### First manifestation: Improves Lipschitzness of loss function
> " the loss changes at a smaller rate and the magnitudes of the gradients are smaller too"

**Def:** 
a function f is L-Lipschitz if $$|f(x) - f(y)| \leq L\|x - y\| \forall$$ x, y 

Rearranging the definition, we get:

$$\frac{|{f(x) - f(y)}|}{||x-y||} \leq L $$

Interpretation: the gradient of the function f (L.H.S) at any point is always bounded by the constant L.

> "Recall loss function tends to have a large number of kinks and sharp minima, this makes gradient descent-based training algorithms unstable due to exploding gradients, highly sensitive to learning rate and initialization"

By applying BN layer, the authors found that the gradient of the loss landscape is bounded by a smaller constant L.
**Importantly, the first manifestation reduces the problem of exploding gradients.**

<br/><br/>
### Second manifestation: Improves the smoothness of loss function
The second manifestation is arguably the stronger benefit and relies on the concept of smoothness. Smoothness has pretty much the exact same definition as Lipschitzness with the addition of the gradient in its definition shown in the red color.

**Def:** a function f is L-smooth if 
:
$$|{\color{red}\nabla}$$ f(x) - $${\color{red}\nabla}$$ f(y) |  $$ \leq L\|x - y\| \forall$$ x,y

Interpretation: BN improves the Lipschitzness of the gradient (achieves smaller L).

We will try to build the intuition of what the benefit entails using a few examples.

### Let's take a look at what the convergence behavior in the loss landscape looks like without BN for 3 different initialization points

![IMG_B5CE78CEC40E-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/3c52d62d-b8b5-4ac6-9b6a-b675837e2bb6)


The key takeaway is that without batch norm, the convergence behavior is **i)** dependent on the choice initialization and **ii)** requires a smaller learning rate or risk overshooting. We see from the leftmost figure, we can not take a large stepsize (indicated by the hypothetical green arrow), otherwise, we will overshoot the minimum.

On the middle and rightmost figure, we got lucky by picking one of the better initializations and we only took 2 steps to reach minima.

### Now with BN, the loss landscape looks like:

<div style="text-align:center;">
  <img src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/18ad2903-c647-4a11-9eb2-7b2670e65908" style="max-width:50%;" alt="Description of the image">
</div>

Applying batch norm smoothens the loss landscape, and the gradient of the loss surface becomes more predictable for all initialization points, leading to stable convergence. Quoting from the paper: 
> "This gives us confidence that when we take a larger step in the direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step." (You can better understand this sentence with the diagram of initialization A) above.)

As a result, we can use a larger learning rate with faster convergence.

### Reference/Appendix


<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>



