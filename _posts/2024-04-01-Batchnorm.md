---
title: The story of BatchNorm
date: 2024-04-01
---

## Introduction
Batch Normalization (BN) has been empirically found to allow deep neural nets (NN) to train faster and more stably (less sensitive to the choice of initialization). However, the exact benefit of the batch norm layer has always been more blurry. The original paper from 2015 [^1] attributes the success to resolving the problem of internal covariate shift (ICS), but as we will see it fails to explain the full picture. Subsequently, in 2019, there is another paper [^2] that argues instead of resolving the ICS, BN's true benefit is making the optimization landscape smoother.






[^1]: [Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", 2015.](https://arxiv.org/abs/1502.03167)
[^2]: [Shibani Santurkar, "How Does Batch Normalization Help Optimization", 2019.](https://arxiv.org/pdf/1805.11604.pdf)


## In this post, we will go over:
1. What is batch norm and how to implement a simple neural net with a batch norm layer
2. First benefit: preventing dead or saturated units
3. Second benefit: ICS removal hypothesis (and why it fails to explain the full picture) (2015 paper)
    1. 1st counter argument
    2. 2nd counter argument
4. Third benefit: Smoothening the loss landscape (2019 paper)


## Formal definition of batch normalization 
Batch norm is a mechanism that aims to stabilize the distribution of inputs, to a network layer, during the training phase. Specifically, the batch norm layer converts the first two moments of the neuron's input (here denoted as y) to 0 mean and unit variance. The mean and standard deviation are calculated based on the current batch.

[![Screenshot 2024-02-25 at 11 00 54 AM](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/312f5c2e-0dad-49fd-8882-384737fdc998)](image-url)
*In practice, a BN layer includes 2 learnable parameters (in green) for the output mean and variance. This is done to give back the expressive power of the original network. i.e. the NN is free to choose whether a non-zero mean is better suited for each layer.*



## Pytorch pseudocode for a 2-layered neuron net with batch norm for a binary classification task.
Suppose we have 10k training data each with dim 500, batch size of 32 and hidden layer of 100 neurons
```
Setup -------------------------------------------------------------
batch_size = 32
batch_ix = torch.randint(0, Xtr.shape[0], (batch_size,))
Xb, Yb = Xtr[batch_ix], Ytr[batch_ix]

# defining our layers of 100 neurons 
X_dim = Xb.shape[1]                   
n_hidden = 100              
W1 = torch.randn((X_dim , n_hidden))
W2 = torch.randn((n_hidden, 2))
b2 = torch.randn(2)

# BatchNorm parameters
bngain = torch.ones((1, n_hidden))          # learnable
bnbias = torch.zeros((1, n_hidden))         # learnable
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
## For each neuron, find the mean and std across all 32 training examples.

# BatchNorm layer
bnmeani = pre_act.mean(0, keepdim=True)
bnstdi = pre_act.std(0, keepdim=True)

# each neuron will be unit gaussian for this batch of data
pre_act = bngain * (pre_act - bnmeani) / bnstdi + bnbias

# running mean to use in validation, these running also get updated
## in the training phase, but these do not require any gradient
with torch.no_grad():
  bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
  bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

# Non-linearity
h = torch.tanh(pre_act)            # non linear layer
logits = h @ W2 + b2               # output layer

Evaluation -------------------------------------------------------------
loss = F.cross_entropy(logits, Yb) # loss function
```
---
## First benefit of BN: Preventing dead or saturated units
Before going over the first benefit, we need to understand some properties of activation functions.

Many activation functions used in a NN, including Tanh are a so-called squashing function. Squashing functions like Tanh remove information from the original input. With Tanh, if the input value (in absolute value) is too big, Tanh will return approximately 1 or -1, which graphically lies on the flat region in the tail ends of the function. From a gradient point of view, if the neuron's output lands on the flat region of Tanh, the gradient would be 0, and virtually this will stop any gradient flowing through this neuron when updating the neuron's weight parameter. We call this a **dead neuron**, and it means that no matter how you perturb the weight of the neuron, it will have no meaningful impact on our final train loss.

![Screenshot 2024-06-02 at 3 16 45 PM](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/708f7e1f-5c3c-40ca-b49c-5034b215709a)


**First benefit of BN: By adding a batch norm layer before the activation layer, we would force the input to take on a zero mean and unit variance distribution which greatly prevents the chance of neurons landing on the flat regions. You can read up on why [dying neurons](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks) are sometimes undesirable**

---
## Second benefit: Resolving the Internal Covariate Shift (ICS) problem (and why it is not entirely true) (2015 paper)

### What is ICS?
ICS is closely related to the concept of covariate shift, which occurs when the input-data distribution shifts over time and as a result, leaves the trained model obsolete. For example, we could use pre-covid stock data to train a stock price prediction model, however, chances are the model will not be effective in predicting returns for post-covid era, as the data distribution has changed substantially.

Now, adding the word "Internal" before "covariate shift", describes a closely related phenomenon that occurs in the training of a neural network, where the distribution of input for an individual layer, changes due to the update of the previous layers' weights.

Let's introduce a useful framework for thinking about neuron nets. We can think of a neural net as a function parameterized by weights. This function takes a given data point as input and outputs a prediction. The training of neural nets can be seen as solving an optimization problem, where we attempt to learn the optimal weight for this gigantic function, with the goal of mapping our datapoint to the true label, as closely as possible. 

In fact, we can break down the original optimization problem into solving a series of smaller, sequential optimization problems at a layer level. That is, each layer is also a function that takes in an input (received from the previous layer) and produces an output (feeds to the next layer). The layer-wise optimization problem has a similar goal, where we try to find good weights that map the input to the desired output. Where it slightly differs is that the input here refers to the output from the previous layer, and the desired output is related to the accumulation of the gradient w.r.t the final loss from the later layers.


**The ICS occurs when the input of the layer (or equivalently, the output of the previous layer) changes drastically (due to weight update in the previous epoch) in every iteration of the training procedure.**

This definition did not make sense for me when I first read it so let's walk through an example for more clarity.

<br/><br/>
### Example to illustrate the problem of internal covariate shift
Consider a neural network with 3 layers (each layer has 1 neuron) with no nonlinearity. Let's walk through how ICS shows up in the backpropagation algorithm when we update the weight of neuron c.
#### Notation
* $$w_c$$ denotes the weight of neuron c

* $$z_c$$ denotes the output of neuron c

* L denotes the loss (common choice is $${\lvert\hat{y} - y \rvert}^2$$)

* $$\alpha $$ is the learning rate


<br/><br/>
### Groundwork


![IMG_6E5E0C290475-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/e0361fcc-cbc0-4c4e-ae31-98bcf04cc10b)

### Commentary:
The update rule for weight at neuron c is:
$$ w_c^{new} \leftarrow w_c^{old} - \alpha \color{red}{\frac{\delta L}{\delta w_c}}$$

Taking a closer look at the gradient term, we can rewrite it as:
$$ \color{red}{\frac{\delta L}{\delta w_c}} = \frac{\delta L}{\delta z_c} \frac{\delta z_c}{\delta w_c}$$

Recall, we also know that the output of neuron c is simply 
$$z_c = w_c*z_b$$. This means that taking the derivative of $$ z_c $$ w.r.t $$ w_c $$ is $$z_b$$.

Incorporating what we discussed, we arrive at
**$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$** which we will use to update the neuron's weight.

Importantly, we see that the weight update of neuron c depends on the output, and hence the weight of neuron b and a. And since the weight of neuron b and a gets updated during each training epoch, the input being fed into layer c changes as well. From the diagram, the weights of a and b in the previous step **(Green)**, could be significantly different from the weights of the same neurons in the current step **(Blue)**. If left untreated we could see why the input distribution to layer c could vary significantly, especially during early epochs when the model is "learning the ropes". Intuitively, and plausibly speaking, if the input distribution keeps changing, it makes sense that it would be hard for layer c to learn any meaningful pattern.

*Aside: when updating a neural network within one training iteration, we have to first update the $${k+1}^{th}$$ layer, before we can update the $$k^{th}$$ layer (take it for granted if you are not familiar), this reverse order of update is dictated by the backpropagation algorithm.* 

### Summary of Internal Covariate Shift

| Problem: | occurs at: | drastic change in: | 
| --- | --- | --- |
| **Covariate Shift** | train vs test time | input to model |
| **Internal Covariate Shift** | epoch i-1 vs epoch i | input to layer |


## Now that we know what the Internal Covariate Shift problem is, does resolving it actually improve our NN training?


<br/><br/>
## Counter argument 1) 2019 Experiment shows contradictory results
The authors of the 2019 paper conducted a simple experiment where they intentionally added noise after the BN layer. 

In theory, if the performance gain of the neural net is indeed attributable to resolving the ICS, deliberately adding back noise **after** the BN layer will erase any of the benefits.

**Result:** They found that the Batchnorm-plus-noise model has similar performance compared with the Batchnorm model. This suggests that BN's main benefit is not in resolving the ICS.

## Counter argument 2) Actual placement of BN layer is before the activation layer
Recall that ICS is the issue where the input distribution to a layer changes drastically between consecutive epochs. In theory, to resolve ICS, we would apply BN layer **right before** feeding the input to the next layer. This is **not** the case in practice, where the BN is actually placed before the activation layer, which is then fed as input to the next layer. This means we are **not guaranteed** that the input distribution after the activation layer is still non-zero mean and unit variance.

---



## Third benefit: Smoothening the loss landscape in 2 manifestations (2019 paper) 
The 2019 paper proposes a new perspective and argues that BN's main benefit is in reparameterizing the underlying optimization problem and smoothening the loss landscape. This benefit comes largely in 2 manifestations and heavily utilizes the concept of Liptschitzness. In this post, I will only give the background and a visual intuition.

### First manifestation: Improves Lipschitzness of loss function
**Def:** 
a function f is L-Lipschitz if $$|f(x) - f(y)| \leq L\|x - y\| \forall$$ x, y 

Rearranging the definition, we get:

$$\frac{|{f(x) - f(y)}|}{||x-y||} \leq L $$

Interpretation: the gradient of the function f (L.H.S) at any point is always bounded by the constant L.

> "Recall loss function tends to have a large number of kinks and sharp minima, this makes gradient descent-based training algorithms unstable due to exploding gradients, highly sensitive to learning rate and initialization"

By applying BN layer, the authors found that the gradient of the loss landscape is bounded by a smaller constant L, meaning that the loss changes at a smaller rate and the magnitudes of the gradients are smaller too.
**Importantly, the first manifestation reduces the problem of exploding gradients.**

<br/><br/>
### Second manifestation: Improves the smoothness of loss function
The second manifestation is arguably the stronger benefit and it relies on the concept of smoothness. Smoothness has pretty much the exact same definition as Lipschitzness with the addition of the gradient in its definition shown in red.

**Def:** a function f is L-smooth if 
:
$$|{\color{red}\nabla}$$ f(x) - $${\color{red}\nabla}$$ f(y) |  $$ \leq L\|x - y\| \forall$$ x,y

Interpretation: BN improves the Lipschitzness of the gradient (achieves smaller L).

### We will try to build the intuition of what the benefit entails using a few examples. Let's take a look at what the convergence behaviour in the loss landscape looks like without BN for 3 different initialization points

![IMG_B5CE78CEC40E-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/3c52d62d-b8b5-4ac6-9b6a-b675837e2bb6)


The key takeaway is that without batch norm, the convergence behaviour is **i)** dependent on the choice initialization and **ii)** we are forced to use a smaller learning rate or risk overshooting the minima. 

Recall what we have shown here is a level curve of the loss surface and the minimum loss is achieved at the center of the ellipse. At each iteration, we will make a step in the orthogonal direction of the level set.

In the leftmost figure, we could not take a large stepsize (indicated by the hypothetical green arrow), otherwise, we will overshoot the minimum. In the middle and rightmost figure, we got lucky by picking one of the better initializations and we only took 2 steps to reach minima.

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



