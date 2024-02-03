## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during the training phase. Specifically, the batch norm layer converts the first two moments of the input to mean 0 and variance 1. Batch norm has long been believed to be successful due to resolving the problem of internal covariate shift. The paper in 2019 argues otherwise and claims the main benefit of batch norm lies in making the optimization landscape more smooth.

Empirically, neural networks with BatchNorm layers tend to train faster and are less sensitive to the choice of hyperparameters initialization. The theoretical benefit of batch norm


## Formal definition of batch normalization 
$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^{2} + \varepsilon}} + \beta $$

where $$\hat{\mu} = \frac{1}{B} \sum_{i=1}^{B} y_i $$
and $$\hat{\sigma}^{2} = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{\mu})^{2} $$





## What is the problem: internal covariate shift (ICS)

ICS is closely related to the concept of covariate shift, which refers to the problem where the input distribution shifts over time. For example, we could use pre-covid's stock data to train a stock price prediction model, however, chances are the model will not be effective in predicting returns for post-COVID time, as the data distribution has changed substantially.

Adding the word "Internal" before "covariate shift", describes a closely related phenomenon where the distribution of input for an individual layer, changes from one training epoch to the next epoch.

Before diving deeper, recall we can view the optimization of the entire deep neural network as solving a series of smaller, sequential optimization problems at a layer level. Each of these smaller optimization problems is independent, GIVEN the output of the previous layer. 

Namely, at each layer, we have a) the input (output of the previous layer), we are also given some b) target output, and we wish to find the best set of weights that transform the input to the desired output as closely as possible (desired output for the final layer will be the true label, the desired output for any layers before is less interpretable for us humans). 

The ICS occurs when the output of the previous layer (input for current layer) changes drastically at each training step, due to the updates of weight in previous layers, stemming from the previous training iteration. Let's walk through an example.

Consider a neural network with 3 neurons with no nonlinearity. Let's walk through how backpropagation will update the weights for epoch $$i-1$$ and epoch $$i$$.
w_c denotes the weight of neuron c
z_c denotes the output of neuron c
L denotes the loss ($|y_hat - y|$)

For instance, the update rule with learning rate $$\alpha $$ for weight at layer c is:
$$ w_c^{new} \leftarrow w_c^{old} - \alpha \frac{\delta L}{\delta w_c}$$

Taking a closer look at the gradient term, we can rewrite it as:
$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} \frac{\delta z_c}{\delta w_c}$$

Recall
$$z_c = w_c*z_b$$

and hence plugging that in 
$$ \frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$

Importantly, we see that the update of the layer c's weight depends on the output of layer b.

You can go through the diagram below.
![IMG_40CEE668B383-1](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/8257cd68-d16c-4d04-8a7e-dbe80649f3b9)

We see that in step 1a) the output of neuron b is a function of $$w_b^{t-2}$$, $$w_a^{t-2}$$ and $$x_{t-1}$$

At the end of step 1), we have updated the weight of all 3 neurons.

We see that in step 2a) the output of neuron b is a function of $$w_b^{t-1}$$, $$w_a^{t-1}$$ and $$x_{t}$$

where $$w_b^{t-2} \neq w_b^{t-1}$$ and $$w_a^{t-2} \neq w_a^{t-1}$$


Aside when updating a neural network within one training iteration, we have to first update the ${k+1}^{th}$ layer, before we can update the $k^{th}$ layer (take it for granted if you are not familiar), this reverse order of update is dictated by the backpropagation algorithm. 


The subtle thing is, when we update the weight of layer c at iteration i using the equation: $$\frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$, the output for node b, $$z_b$$, actually still uses the learned weights from the previous iteration i-1 . (Recall $$w_b$$ is optimized for the data point at iteration i-1 $$(x_{i-1},y_{i-1})$$.) In other words, at iteration i, the input of layer c's optimization problem assumes the distribution of $$x_i$$ is the same as the distribution of $$x_{i-1}$$ by using a "stale" weight $$w_b$$ at this exact point in the backpropagation algorithm. If $$x_{i-1}$$ has a significantly different value than $$x_i$$, then $$w_b^{i-1}$$ will not be good for minimizing the loss for the data point $$x_i$$.

## Empirically,does batch norm's main benefit come from internal covariate shift
The authors conducted a simple experiment where they intentionally added noise after the BN layer. The rationale is: if the performance gain is indeed attributable to resolving the internal covariate shift, adding back noise will erase any of the gains during validation.

In the end, they found that the Batch norm plus noise model has largely similar performance compared with the Batch norm model. This suggests that BN's main benefit does not come from resolving the ICS.

## Benefit of batch norm and the 2 manifestations
The thesis of the paper is that BN's main benefit is that it reparametrizes the underlying optimization problem and smoothens the loss landscape. This benefit comes largely in 2 manifestations and utilizes the concept of Liptschitzness.

### First manifestation: Improves Lipschitzness of loss function
" the loss changes at a smaller rate and the magnitudes of the gradients are smaller too"

Definition : a function f is L-Lipschitz if $$|f(x) - f(y)| \leq L\|x - y\| \forall$$ x, y 
rearranging, we get:

$$\frac{|{f(x) - f(y)}|}{||x-y||} \leq L $$

Interpretation: the gradient of the function f at any point is bounded by constant L.

"Recall loss function tends to have a large number of kinks and sharp minima, this makes gradient descent-based training algorithms unstable due to exploding gradients, highly sensitive to learning rate and initialization"

The benefit is that the gradient does not explode.

### Second manifestation: improves the smoothness of loss function
The second manifestation is the stronger effect.

Definition : a function f is L-smooth if 
a
$$|{\color{red}\nabla}$$ f(x) - $${\color{red}\nabla}$$ f(y) |  $$ \leq L\|x - y\| \forall$$




Takeaway: BN improves the Lipschitzness of the gradient

![Ps2 3](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/23cc5783-2f3a-4013-84b6-2b2cded5f519)


Without batch norm, the convergence behavior is sensitive to the initialization. We see from the leftmost figure, we can not take a large stepsize (indicated by green arrow), otherwise, we will overshoot the minimum.

On the rightmost figure, we got lucky by picking a rare initialization that takes 2 steps to reach minima.


![Ps2 2](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/a9256d60-6d7e-409c-ba75-4781d50677fa)

<img src="[https://your-image-url.type](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/a9256d60-6d7e-409c-ba75-4781d50677fa)" width="100" height="100">


With BatchNorm, the gradient of the loss surface becomes more predictable for all initialization points, leading to stable convergence. This "gives us confidence that when we take a larger step in the direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step." As a result, we can use a larger learning rate with faster convergence.

### Not a simple rescale!

In fact, the author of
<img width="994" alt="Screen Shot 2024-01-10 at 1 48 38 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/c2186a71-c45f-451c-8c68-0d5ec9c861f6">






<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
