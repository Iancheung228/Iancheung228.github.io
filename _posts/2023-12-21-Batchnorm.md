## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, they convert the first two moments to mean 0 and variance one.


## What is batch norm
$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^2 + \varepsilon}} + \beta $$

$$\hat{\mu} = \frac{1}{B} \sum_{i=1}^{B} y_i $$
$$\hat{\sigma}^2 = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{\mu})^2 $$

## What is the internal covariate shift

ICS is closely related to covariate shift, which is referred to as the problem where the input distribution shifts over time. For example, we could use pre-covid's stock data to train a model, however, chances are the model will not be effective in predicting returns in 2020, as the data distribution changed quite substantially.

ICS describes the phenomenon where the data distribution from one layer to the next changes during training.

Recall each layer can be interpreted to be an optimization problem. We are given some input, we are also given some output, we wish to find the best weights that transform the input to the desired output as closely as possible. The ICS occurs when the output of the previous layer (input for current layer) changes drastically at each training step, due to the updates of weight of previous layers.

Let us illustrate this with a simple example
Consider a neural network with 3 neurons with no nonlinearity. Suppose we want to update the weight of layer b. The update rule is:

$$ w_b^{new} \leftarrow w_b^{old} - \alpha \frac{\delta L}{\delta w_b}$$

Taking a look at the gradient term:
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} \frac{\delta z_b}{\delta w_b}$$

Recall
$$z_b = w_b*z_a$$
hence 
$$ \frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} z_a$$

We see importantly, the update of layer b depends on the output of layer a. 

In neural networks, the output of the first layer feeds into the second layer, the output of the second layer feeds into the third, and so on. When the parameters of a layer change, so does the distribution of inputs to subsequent layers.

Suppose we are at the end of training of iteration i-1 , which means we have completed the backpropagation procedure to find (in order)
$$ \frac{\delta L}{\delta w_c},\frac{\delta L}{\delta w_b}, \frac{\delta L}{\delta w_a}$$

and updated the weight using the derivative respectively. The loss in the derivative is with respect to training data at ${(i-1)}^{th}$ iteration (x_{i-1},y_{i-1})

Now for this illustration, suppose we are in the back propagation procedure in the ith iteration, for layer b.
In general, we know that within 1 training epoch of backpropagation, we have to first update the k+1 layer, before we can update the k layer (take it for granted if you are not familiar). 

The subtle thing is, when we update the weight of layer b at iteration i using the equation: $\frac{\delta L}{\delta w_b} = \frac{\delta L}{\delta z_b} z_a$, the output for node a, $z_a$, actually still uses the learned weights from the previous iteration i-1 . (Recall $w_a$ is optimized for the data point at iteration i-1 $(x_{i-1},y_{i-1})$.) In other words, at iteration i, the input of layer b's optimization problem assumes the distribution of $x_i$ is similar to the distribution of $x_{i-1}$ as they share the weight $w_a$ at this point in the back propagation. Clearly, if $x_{i-1}$ is significantly different than $x_i$, then $w_a^{i-1}$ will not be good for minimizing the loss for the data point $x_i$.



<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
