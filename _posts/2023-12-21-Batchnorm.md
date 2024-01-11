## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, the batch norm layer convert the first two moments to mean 0 and variance one.


## Formal definition of batch normalization 
$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^2 + \varepsilon}} + \beta $$

where $$\hat{\mu} = \frac{1}{B} \sum_{i=1}^{B} y_i $$
and $$\hat{\sigma}^2 = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{\mu})^2 $$

## What is the internal covariate shift

ICS is closely related to the concept of covariate shift, which refers to the problem where the input distribution shifts over time. For example, we could use pre-covid's stock data to train a model, however, chances are the model will not be effective in predicting returns when covid hits, as the data distribution has changed substantially.

Adding the word "Internal" before "covariate shift", describes another phenomenon where the input data distribution of an individual layer, changes from one training epoch to the next epoch.

Recall we can view the optimization of the deep neural network as solving a series of smaller optimization problems at each layer. Each of these smaller optimization problems are independent, conditioned on knowing what the output of the previous layer is. We are given some input, we are also given some output, and we wish to find the best set of learnable weights that transform the input to the desired output as closely as possible (desired output for the final layer will be the true label, the desired output for any layers before is less interpretable). The ICS occurs when the output of the previous layer (input for current layer) changes drastically at each training step, due to the updates of weight of previous layers. I know this sentence probably does not make sense so let's walk through an example.
In neural networks, the output of the first layer feeds into the second layer, the output of the second layer feeds into the third, and so on. Simply put, when the weight parameters of the previous layer change, the output of the previous layer changes, equivalently so does the distribution of inputs of the next layer.

Consider a neural network with 3 neurons with no nonlinearity. Suppose we want to update the weight of layer b. The update rule with learning rate $$\alpha $$ is:

$$ w_b^{new} \leftarrow w_b^{old} - \alpha \frac{\delta L}{\delta w_b}$$

Taking a closer look at the gradient term, and refactoring it:
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} \frac{\delta z_b}{\delta w_b}$$

Recall
$$z_b = w_b*z_a$$

and hence plugging that in 
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} z_a$$

We see importantly, the update of layer b's weight depends on the output of layer a. 

WLOG let's assume we train the network with n epochs, with each epoch sampling one datapoint from the dataset to train on. At each training iteration, we need to first update the weight of layer c, then layer b, and finally layer a (by finding $$ \frac{\delta L}{\delta w_c},\frac{\delta L}{\delta w_b}, \frac{\delta L}{\delta w_a}$$), this rule is dictated by backpropagation.

Now suppose we are at the end of training iteration i-1, which means we have completed the backpropagation procedure to find (in order)
$$ \frac{\delta L}{\delta w_c},\frac{\delta L}{\delta w_b}, \frac{\delta L}{\delta w_a}$$ and updated the weight of layer c, b, and a using the derivatives respectively. Note, the loss in the derivative in our calculations is with respect to the training data at $${(i-1)}^{th}$$ iteration $$(x_{i-1},y_{i-1})$$

We have completed the optimization procedure for iteration i-1 and hence let's move on to iteration i.
In general, we know that within 1 training epoch of backpropagation, we have to first update the k+1 layer, before we can update the k layer (take it for granted if you are not familiar). 

The subtle thing is, when we update the weight of layer c at iteration i using the equation: $$\frac{\delta L}{\delta w_c} = \frac{\delta L}{\delta z_c} z_b$$, the output for node b, $$z_b$$, actually still uses the learned weights from the previous iteration i-1 . (Recall $$w_b$$ is optimized for the data point at iteration i-1 $$(x_{i-1},y_{i-1})$$.) In other words, at iteration i, the input of layer c's optimization problem assumes the distribution of $$x_i$$ is largely similar to the distribution of $$x_{i-1}$$ as they share the weight $$w_b$$ at this exact point in the backpropagation algorithm. Naturally, if $$x_{i-1}$$ has a significantly different value than $$x_i$$, then $$w_b^{i-1}$$ will not be good for minimizing the loss for the data point $$x_i$$.

## Does batch norm's main benefit come from internal covariate shift
The authors conducted a simple experiment where they intentionally added noise after the BN layer. The rationale is: if the performance gain is indeed attributable to resolving the internal covariate shift, adding back noise will erase any of the gains in validation performance.

In the end, they found that the Batch norm plus noise model has largely similar performance compared with the Batch norm model.

## Benefit of batch norm and the 2 manifestations
The thesis of the paper is that BN's main benefit is that it reparametrizes the underlying optimization problem and smoothens the loss landscape.

### First manifestation: Improves Lipschitzness of loss function
" the loss changes at a smaller rate and the magnitudes of the gradients are smaller too"


Definition : a function f is L-Lipschitz if $$|f(x) - f(y)| \leq L\|x - y\| \forall$$ x, y 
rearranging, we get:

$$\frac{|{f(x) - f(y)}|}{||x-y||} \leq L $$

The gradient of the function f at any point is bounded by constant L.

"Recall loss function tends to have a large number of kinks and sharp minima, this makes gradient descent-based training algorithms unstable due to exploding gradients, highly sensitive to learning rate and initialization"

The benefit is that the gradient does not explode.

### Second manifestation: improves the smoothness of loss function
The second manifestation is the stronger effect.

Definition : a function f is L-smooth if 


|$${\color{red}\nabla}$$ f(x) - $${\color{red}\nabla}$$ f(y)| $$\leq L\|x - y\| \forall$$ x,y

Takeaway: BN improves the Lipschitzness of the gradient

![Ps2 3](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/23cc5783-2f3a-4013-84b6-2b2cded5f519)


Without batch norm, the convergence behavior is sensitive to the initialization. We see from the leftmost figure, we can not take a large stepsize (indicated by green arrow), otherwise, we will overshoot the minimum.

On the rightmost figure, we got lucky by picking a rare initialization that takes 2 steps to reach minima.


![Ps2 2](https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/a9256d60-6d7e-409c-ba75-4781d50677fa)


With BatchNorm, the gradient of the loss surface becomes more predictable for all initialization points, leading to stable convergence. This "gives us confidence that when we take a larger step in the direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step." As a result, we can use a larger learning rate with faster convergence.

### Not a simple rescale!

In fact, the author of
<img width="994" alt="Screen Shot 2024-01-10 at 1 48 38 PM" src="https://github.com/Iancheung228/Iancheung228.github.io/assets/37007362/c2186a71-c45f-451c-8c68-0d5ec9c861f6">






<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
