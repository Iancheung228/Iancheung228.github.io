## Introduction
Batch norm is a mechanism that aims to stabilize the distribution of inputs to a network layer during training. Neural networks with BatchNorm tend to train faster and are less sensitive to the choice of hyperparameters. Specifically, they convert the first two moments to mean 0 and variance one.


## what is batch norm
hi $$\sqrt{3x-1}+(1+x)^2$$ bye


$$ \hat{y} = \gamma \frac{(y - \hat{\mu})}{\sqrt{\hat{\sigma}^2 + \varepsilon}} + \beta $$

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>
