## Introduction
Batchnorm has been empirically shown to allow deep neural nets to train faster and more stably (less sensitive to the choice of initialization). However, the exact theoretical benefit of the batch norm layer has always been a topic of debate. The main difficulty perhaps comes from the fact that NN has many moving parts so it is hard to put your finger down on the exact root problem the BN layer solves, and whether BN is the unique mechanism that solves it. The original paper from 2015 attributes the success to resolving the problem of internal covariate shift. In 2019, there is a new paper that argues, that instead of ICS, it is the fact that batch norm layer makes the optimization landscape smoother that justifies BN's success.





In this blog, we will go through the list of items:
<ol>
  <li>What is batch norm and implement a simple neuron net with batch norm layer</li>
  <li>First benefit: preventing dead or saturated units</li>
  <li>Second benefit: Resolving the Internal Covariate Shift problem (and why it is not entirely true) (2015 paper)</li>
  <li>Third benefit: Smoothening the loss landscape (2019 paper)</li>
</ol>

<br/><br/>

## Formal definition of batch normalization 
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during the training phase. Specifically, the batch norm layer converts the first two moments of the input to mean 0 and variance 1. 

<img width="729" alt="Screenshot 2024-02-25 at 11 00 54 AM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/312f5c2e-0dad-49fd-8882-384737fdc998">

In practice, the BN operation includes 2 learnable parameters (in green) for the output mean and variance for each column. This is done so that BN to maintain the expressive power of the original network.

<br/><br/>

Let us walk through an example of a 2 layered neuron net
```
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
X_dim = 5 # dimension of a training data point
g = torch.Generator().manual_seed(2147483647) # for reproducibility
W1 = torch.randn((X_dim , n_hidden), generator=g)
W2 = torch.randn((n_hidden, X_dim),          generator=g)
b2 = torch.randn(X_dim,                      generator=g)




# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden)) # not trained using back prop (used at inference)
bnstd_running = torch.ones((1, n_hidden)) # not trained using back prop (used at inference)

parameters = [W1, W2, b2, bngain, bnbias]
for p in parameters:
  p.requires_grad = True

batch_size = 32
# minibatch construct
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# Linear layer
hpreact = embcat @ W1 # we don't need a bias term as the BN layer will get rid of the bias here

##^ want hpreact to be gaussian, hpreact has shape (batch_size, 100) where 100 is the number of neurons, and we want to find the mean and std for each neuron, 
  ## across all 32 examples. i.e take average of the 1st neuron's value (100 of those in total) for the 32 datapoints.

# BatchNorm layer-------------------------------------------------------------
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias # each neuron will be unit gauusian for this batch of data
# running mean to use in validation, these running also get updated in trainning phase, but these do not require any gradient
with torch.no_grad():
  bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
  bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
# -------------------------------------------------------------
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
logits = h @ W2 + b2 # output layer
loss = F.cross_entropy(logits, Yb) # loss function

```

## First benefit: preventing dead or saturated units

Many activation functions, including Tanh are a squashing function, which means Tanh will remove information from the given input. Specifically, if the input value in absolute terms is too big, tanh will return 1/-1, which corresponds to the flat region in the tail end of the function. From a gradient pov, if we land on the flat region, the gradient would be 0, and virtually this will stop any gradient flowing through this neuron. In other words, if the neuron's output in absolute terms is too big, no matter how you perturb the value of the neuron, it will not have an impact on the final loss, and hence the neuron will not get updated. We call this a dead neuron.

By adding a batch norm layer before the activation layer, we would force the input to take on a zero mean and unit variance distribution which prevents the landing on flat regions and subsequently dead neurons.


## Second benefit: Resolving the Internal Covariate Shift problem (and why it is not entirely true) (2015 paper)

ICS is closely related to the concept of covariate shift, which is when the input-data distribution shifts over time. For example, we could use pre-covid's stock data to train a stock price prediction model, however, chances are the model will not be effective in predicting returns for post-COVID time, as the data distribution has changed substantially.

Now, adding the word "Internal" before "covariate shift", describes a closely related phenomenon where the distribution of input for an individual layer, changes due to the update of the previous layers' weights.

Let me introduce a useful framework to think about Neuron Nets. We can view the optimization of the entire deep neural network as solving a series of smaller, sequential optimization problems at a layer level. A 10-layer NN could be seen as solving 10 smaller optimization problems. Each of these smaller optimization problems is separate, and the only 2 ingredients we need  **a)** the output of the previous layer and **b)** accumulation of the gradient w.r.t final loss of the next layer. 

The ICS occurs when the output of the previous layer (input for current layer) changes drastically at each training step, due to the updates of weight in previous layers, stemming from the previous training iteration. Let's walk through an example.

<br/><br/>
### Example
Consider a neural network with 3 layers (each with 1 neuron) with no nonlinearity. Let's walk through how backpropagation will update the weights of the 3 layers, for epoch $$i-1$$ and $$i$$.
#### Notation
* $$w_c$$ denotes the weight of neuron c

* $$z_c$$ denotes the output of neuron c

* L denotes the loss (common choice is $${\lvert\hat{y} - y \rvert}^2$$)

* $$\alpha $$ is the learning rate


<br/><br/>
Recall, that the update rule for weight at layer c is:
$$ w_c^{new} \leftarrow w_c^{old} - \alpha \(\color{red}{\frac{\delta L}{\delta w_c}}\)$$

Taking a closer look at the gradient term, we can rewrite it as:
$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} \frac{\delta z_c}{\delta w_c}$$

Recall
$$z_c = w_c*z_b$$

and hence plugging that in 
$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$

Importantly, we see that the update of the layer c's weight depends on the output of layer b.




You can go through the diagram below, where we go through in order of **1a), 1b), 1c), 2a)**
![IMG_40CEE668B383-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/8257cd68-d16c-4d04-8a7e-dbe80649f3b9)

We see that in step **1a**) the output of neuron b, $$z_b$$, is a function of $$w_b^{t-2}$$, $$w_a^{t-2}$$ and $$x_{t-1}$$

At the end of step **1)**, we have updated the weight of all 3 neurons.

We see that in step **2a)** the output of neuron b, $$z_b$$, is a function of $$w_b^{t-1}$$, $$w_a^{t-1}$$ and $$x_{t}$$

where $$w_b^{t-2} \neq w_b^{t-1}$$ and $$w_a^{t-2} \neq w_a^{t-1}$$

Hence, we see that in step **2a)**, the input to layer c  $$:= z_b$$ would have a completely different distribution than the corresponding $$z_b$$ in step **1a)**.


*Here we make the simplifying assumption that at each iteration we only train on 1 data point, in practice, we train on a mini-batch and the idea of distribution applies* 

*Aside when updating a neural network within one training iteration, we have to first update the $${k+1}^{th}$$ layer, before we can update the $$k^{th}$$ layer (take it for granted if you are not familiar), this reverse order of update is dictated by the backpropagation algorithm.* 


The subtle thing is, when we update the weight of layer c at iteration i using the equation: $$\frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$, the output for node b, $$z_b$$, actually still uses the learned weights from the previous iteration i-1 . (Recall $$w_b$$ is optimized for the data point at iteration i-1 $$(x_{i-1},y_{i-1})$$.) In other words, at iteration i, the input of layer c's optimization problem assumes the distribution of $$x_i$$ is the same as the distribution of $$x_{i-1}$$ by using a "stale" weight $$w_b$$ at this exact point in the backpropagation algorithm. If $$x_{i-1}$$ has a significantly different value than $$x_i$$, then $$w_b^{i-1}$$ will not be good for minimizing the loss for the data point $$x_i$$.

This ICS problem was believed to be a huge problem if left unaddressed and the authors of the original paper hence suggested to add a BN layer after each layer of the original NN.

<br/><br/>
## Counter argument by the experiment in 2019
The authors of the 2019 paper conducted a simple experiment where they intentionally added noise after the BN layer (we call it the Batch norm plus noise model). 

The rationale is: if the performance gain is indeed attributable to resolving the internal covariate shift, adding back noise will erase any of the benefit.

In the end, they found that the Batch norm plus noise model has largely similar performance compared with the Batch norm model. This suggests that BN's main benefit does not come from resolving the ICS.

## Argument 2: The benefit of batch norm improves the loss landscape and comes in 2 manifestations
The thesis of the paper is that BN's main benefit is that it reparametrizes the underlying optimization problem and smoothens the loss landscape. This benefit comes largely in 2 manifestations and utilizes the concept of Liptschitzness.

### First manifestation: Improves Lipschitzness of loss function
> " the loss changes at a smaller rate and the magnitudes of the gradients are smaller too"



Definition : 
a function f is L-Lipschitz if $$|f(x) - f(y)| \leq L\|x - y\| \forall$$ x, y 

Rearranging the definition, we get:

$$\frac{|{f(x) - f(y)}|}{||x-y||} \leq L $$

Interpretation: the gradient of the function f at any point is bounded by constant L.

> "Recall loss function tends to have a large number of kinks and sharp minima, this makes gradient descent-based training algorithms unstable due to exploding gradients, highly sensitive to learning rate and initialization"

**The first manifestation reduces the problem of exploding gradients.**

<br/><br/>
### Second manifestation: Improves the smoothness of loss function
The second manifestation is arguably the stronger benefit and relies on the concept of smoothness. Smoothness has the exact same definition as Lipschitzness with the addition of the red gradient.

Definition : a function f is L-smooth if 
:
$$|{\color{red}\nabla}$$ f(x) - $${\color{red}\nabla}$$ f(y) |  $$ \leq L\|x - y\| \forall$$ x,y




Interpretation: BN improves the Lipschitzness of the gradient.

### Let's take a look at what the convergence behaviour looks like without BN for 3 different initialization points

![Ps2 3](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/23cc5783-2f3a-4013-84b6-2b2cded5f519)


The key takeaway is that without batch norm, the convergence behavior is dependent on the choice initialization. We see from the leftmost figure, we can not take a large stepsize (indicated by green arrow), otherwise, we will overshoot the minimum.

On the rightmost figure, we got lucky by picking a rare initialization that takes 2 steps to reach minima.

### Now with BN:
<img src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/a9256d60-6d7e-409c-ba75-4781d50677fa" alt="Ps2 2" width="300"/>

With BatchNorm, the gradient of the loss surface becomes more predictable for all initialization points, leading to stable convergence. Quoting from the paper: 
> "This gives us confidence that when we take a larger step in the direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step."
As a result, we can use a larger learning rate with faster convergence.

### Not a simple rescale!

In fact, the author of the 2019 paper noted that this is not a simple rescale.
<img width="994" alt="Screen Shot 2024-01-10 at 1 48 38 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/c2186a71-c45f-451c-8c68-0d5ec9c861f6">






<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
