---
layout: post
title: Causal Inference Notes
date: 2024-08-25
---


Plan:
practical

connect with bayesian, what does trustworthy


The business team comes up with a new idea for a feature and hypothesize that adding this feature will improve conversion $$ (\theta) $$. They come to you and ask for your help to collect sample data to empirically evaluate whether the hypothesis is reasonable.

In a perfect world, we could measure the true effect of a treatment by giving the entire population on Earth the control condition and recording the outcome. Then, on a perfect clone of Earth, we would give everyone the treatment and measure that outcome as well. Comparing these two results would give us a definitive, black-and-white answer.

Of course, this isn’t possible in reality — we can’t run experiments on parallel Earths. So instead, we randomly assign individuals within a sample population to either control or treatment groups. This randomness introduces variability in the results, which is why we use statistical distributions to model and analyze the uncertainty in our estimates.


Since the business team is concerned with whether the new feature is an improvement, we need to use a **one-sided test**. 

H0 and HA are always compliments to each other

$$
\begin{aligned}
H_0\!:\ & \theta_0 \leq \theta_A \quad \text{(null hypothesis)} \\
H_A\!:\ & \theta_A > \theta_0 \quad \text{(alternative hypothesis)}
\end{aligned}
$$

If we denote $$ \delta = \theta_0 - \theta_A $$
$$
\begin{aligned}
H_0\!:\ & \delta \geq 0 \quad \text{(null hypothesis)} \\
H_A\!:\ & \delta < 0 \quad \text{(alternative hypothesis)}
\end{aligned}
$$

Our final goal is always the same: After collecting observed data, we will decide whether we **reject $$H_0$$** (and accept $$H_A$$) OR **do not reject $$H_0$$** (and decline $$H_A$$).

Below are the high-level steps we need to perform
- We define a test statistic T, which is a random variable (r.v.)
- We conduct the one-sided test, **under the assumption that the NULL hypothesis is true**; consequently, the test statistic T follows a particular distribution that we call the **null distribution**
- We calculate t, the observed value of the test statistic r.v.
- We evaluate the extremity of t relative to the null distribution
- If the observed t is unlikely to be generated from the null distribution, we have more evidence to believe against $$H_0$$, and if it passes a certain threshold (significance level), we will decide to reject the null hypothesis. The threshold we pick is denoted as $$ \alpha $$. Note that even when we reject the null hypothesis, it is possible that in reality, the observed outcome came from the Null distribution, in which case we have incorrectly rejected the null hypothesis; this is denoted as a Type I error, which we will discuss later.
- Statistically, we formalise the extremity of t using the p-value. 


Before we define p-value formally, recall by design, either the Null hypothesis (H0) is true or the alternative hypothesis (HA) is true. At the same time, we can decide to reject the Null or reject the Alternative. We can treat these as 2 random variables and visualize them in a grid.

<img width="760" alt="Screenshot 2025-04-16 at 1 55 38 PM" src="https://github.com/user-attachments/assets/cf9fac54-9451-45af-8de7-8b6bcbff7fba" />

In our one sided test however, we operate under the assumption that the Null hypothesis is true, that is, we condition on it. And the grid now becomes 

<img width="804" alt="Screenshot 2025-04-16 at 1 48 25 PM" src="https://github.com/user-attachments/assets/046f20b6-21ac-42c1-9f04-f1099bc67946" />

### There are exactly 4 possible outcomes 
1. Given H₀ is true and we correctly do not reject it  
2. Given H₀ is true and we incorrectly reject it (Type I error)
3. Given H₀ is false and we incorrectly do not reject it  (Type II error)
4. Given H₀ is false and we correctly reject it


we control the type 1 and 2 error with the significance level and power of the test



We now formally define the following:

**P-value** is defined as the probability of observing a more extreme test statistic, under the assumption that the **null hypothesis is true**. 

$$ \text{p-value} = P( |Z| \geq z_{\alpha/2} |\text{Null hyp is assumed to be true}) $$

***Type 1 error ($$alpha$$):*** Incorrectly rejecting the Null hypothesis when it is indeed true.

***Power (1-beta):*** is defined as the probability of correctly rejecting the null hypothesis under the assumption that **Alternative hypothesis is true**. Equivalently, it is the prob of not making a Type II error.

$$ \text{power} =  P(\text{p-value} < \alpha | \text{Alternative hyp is assumed to be true}) $$

**Type 2 error:** We fail to reject H0 when HA is indeed true


Let's incorporate everything we have discussed into 2 visual diagrams. Where $$ * $$ is the threshold of our decision, $$\delta_{H_0}$$ and $$\delta_{H_A}$$ is 0 and the true improvement (usually the minimum detectable effect in practice), respectively. 



<div style="display: flex; justify-content: space-between; align-items: flex-start;">

  <!-- First Plot (Null hypothesis) -->
  <figure id="null-graph" style="text-align: center; width: 45%; margin: 0;">
    <h3 style="text-align: center;">Type I Error</h3>
    <img src="https://github.com/user-attachments/assets/e292e422-1b9c-4af0-8786-fd9137c174f1" alt="IMG_E39688680BFC-1" width="100%" />
    <figcaption> Under the assumption that null is true (i.e our observation is generated by the null distribution). </figcaption>
  </figure>
  
  <!-- Second Plot (Alternative hypothesis) -->
  <figure id="alternative-graph" style="text-align: center; width: 45%; margin: 0;">
    <h3 style="text-align: center;">Type II Error</h3>
    <img src="https://github.com/user-attachments/assets/3f8fbd92-a02f-4395-b395-595d60499c36" alt="IMG_5665639446F5-1" width="100%" />
    <figcaption> Under the assumption that the alternative is true (i.e our observation is generated by the alternative distribution). </figcaption>
  </figure>

</div>

In the figure on the left, an error is made if we decide to reject the Null, which occurs when our observation lies to the right of the critical value - $$*$$. While in the right figure, an error is made if we decide to not reject the Null, which occurs when our observation lies to the left of $$*$$. 

**Remark:** Observe that we can reduce the probability of a Type II error by shifting the decision boundary (i.e., the value of $$\alpha$$ to the right. However, doing so would increase the probability of a Type I error, since a larger critical region would make it easier to reject the null hypothesis.








## 3 levers that affect power

1) $$ \alpha $$: To increase power, we can increase our $$ \alpha $$, but again there is a trade-off between making a Type I error (false positive) and a Type II error (false negative).

2) $$ n $$: You can increase the sample size. Both the Null and Alternative distributions will become narrower in shape, leading to less overlap between the 2 distributions. 

3) $$ \delta $$: If the beliefed true parameter is further away from the null hypothesis value, there will again be less overlap between the 2 distributions and power increases.

### Practically how do we pick what value of $$\delta$$ to use for the alternative hypothesis in the calculation of power?

This corresponds to placing $$ \delta$$ in the diagram
Given a significance level α, we are typically interested in determining the sample size required to achieve a desired statistical power. However, the specific value of the alternative hypothesis, which is needed in our power calculation, is unknown.

A common choice to plug in is the **minimum detectable effect (MDE)**. This is the minimum effect size 𝛿 that would be considered meaningful for the business problem at hand.In using this value for delta, we’ve assured ourselves that the calculated sample size is big enough to detect the smallest effect we care about (as well as anything bigger).



### Example of how to calculate the sample size needed for a desired power, given $$\alpha$$ and true alternative of $$ \delta_A >0 $$

Consider the following setup:
- The metric is conversion - where the object either successfully converted or failed to convert, and this follows a Bernoulli
- The sample variance for a Bernoulli variable would be $$ \sigma^2 =  \frac{\mu (1-\mu)}{N} $$
- The true $$\Delta_A$$ is $$\delta_A$$ and is greater than 0
- Let the Null and alternative distributions be parameterized by $$N(\mu_1, \sigma_1^2)$$ and $$N(\mu_2, \sigma_2^2)$$ with sample sizes $$N_1$$ and $$N_2$$. 
- $$\Delta$$ = $$N(\mu_1, \sigma_1^2) - N(\mu_2, \sigma_2^2) = N\left(\mu_2 - \mu_1, \sigma_1^2 + \sigma_2^2 \right)$$


**Goal** We wish to obtain a power of 1- $$\beta$$, we want to reject the null at least 100(1- $$\beta$$)% of the time assuming the alternative hypothesis is true.

$$
1 - \beta \leq P\left( Z > z_{\alpha} \, \Big| \, \text{alternative} \right) 
= P\left( \frac{ \widehat{\mu_1} - \widehat{\mu_0} }{ \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right) } } > z_{\alpha} \, \Big| \, \text{alternative} \right)
$$



**Note 1** The z score follows a standard normal only under the Null hypothesis, and not under the alternative. We have to rearrange the above expression to obtain something normally distributed.
   
**Note 2** Under alternative $$ \text{Var}(\widehat{\mu_1} -  \widehat{\mu_0}) = \frac{\mu_1(1- \mu_1)}{n} +  \frac{\mu_0(1-\mu_0)}{n} $$

Continuing ...

$$
\begin{aligned}
1 - \beta &\leq P\left( \widehat{\mu}_1 - \widehat{\mu}_0 > z_{\alpha} \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right)} \Big| \, \text{alternative} \right) \\
          &= P\left( \frac{(\widehat{\mu}_1 - \widehat{\mu}_0) - \delta_A}{ \sqrt{\frac{\mu_1(1 - \mu_1)}{n} + \frac{\mu_0(1 - \mu_0)}{n}} } > \frac{z_{\alpha} \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right)} - \delta_A}{ \sqrt{\frac{\mu_1(1 - \mu_1)}{n} + \frac{\mu_0(1 - \mu_0)}{n}} } \Big| \, \text{alternative} \right) \\
          &= P\left( Z^* > \frac{z_{\alpha} \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right)} - \delta_A}{ \sqrt{\frac{\mu_1(1 - \mu_1)}{n} + \frac{\mu_0(1 - \mu_0)}{n}} } \Big| \, \text{alternative} \right) \\
\end{aligned}
$$


Now the quantity on the left of the inequality, $$Z^* \sim N(0, 1)$$ under the alternative hypothesis



$$
z_{1 - \beta} \leq \frac{
    z_{\alpha} \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right)} - \delta_A
}{ 
    \sqrt{\frac{\mu_1(1 - \mu_1)}{n} + \frac{\mu_0(1 - \mu_0)}{n}}
} \quad \text{since} \quad Z^* \sim N(0, 1)
$$

Then, you can simply continue the algebraic manipulation and isolate for n.




### Aside $$ \text{P-value} \neq P(H_0 | data) $$
A common misconception is that the p-value represents the probability that the null hypothesis ($$H_0$$) is true, given the observed data. I.e. a p-value of $$5\%$$ means the null hypothesis has $$5\%$$ chance of being true $$ P(H_0 \mid \text{p-value} = 0.05) = 0.05 $$. 

Or something like: a p-value of $$5\%$$ means there is a $$5\%$$ chance of us making a false positive. 
$$ P(H_0 \mid \text{p-value} = 0.05) = 0.05 $$

The above interpretations are wrong because the p-value doesn't tell us the probability of the Null being true.

While this is an expression of great interest, it is important to note that we cannot directly calculate this probability without knowing the prior probability of $$H_0$$.

$$P(H_0 \mid \text{data})$$ and the p-value are related by the Bayes' theorem:

$$
\begin{aligned}
P(H_0 \mid \text{data}) &= \frac{ P(\text{data} \mid H_0) \times P(H_0)}{P(\text{data})} \\
&= \text{p-value} \times \frac{P(H_0)}{P(\text{data})}
\end{aligned}
$$

$$ P(H_0 |Data) = \frac{ P(Data|H_0)P(H_0)}{P(Data|H_0)P(H_0) + P(Data|H_1)P(H_1)} $$

It does align with our intuition: a lower p-value suggests that the null hypothesis is less likely to be true.




In reality, the p-value is the probability of observing a result as or more extreme than what was observed, under the assumption that the null hypothesis is true.
$$ \text{p-value} = P(\delta \text{ observed or more extreme} \mid H_0 \text{ is true}) $$


### What is so bad about low power, does it matter as long as the p-value is low? ###

Even if the p-value is statistically significant, if the experiment is low-powered to begin with there is still a high probability of the Null hypothesis to be true.

**False positive risk (FPR)** is the probability that the Null hypothesis is true while we decide to reject the null hypothesis (p-value < 5%). Mathematically it is represented by $$ P(\text{Null is True} \mid \text{p-value} < 0.05) $$


Let
- SS be a statistically significant result i.e. p-value < 0.05
- $$\alpha$$ be the threshold to declare statistical significance.
- $$\beta$$ be the Type II error rate.
- $$\pi$$ be the prior probability of the null hypothesis.


$$
\begin{aligned}
P(H_0 \mid SS) &= P(SS \mid H_0) \cdot \frac{P(H_0)}{P(SS)} \\
&= \frac{ P(SS \mid H_0) P(H_0)}{P(SS \mid H_0) P(H_0) + P(SS \mid H_1) P(H_1)} \\
&= \frac{\alpha \cdot \pi}{\alpha \cdot \pi + (1-\beta)(1-\pi)}
\end{aligned}
$$


If the power $$1- \beta$$ is low the denominator will be small and the entire fraction (FPR) will be high. This means when power is low the probability that the Null hypothesis is true is small (even when p-value is statistically significant)

We see that with low power even if you have a very small p-value, the FPR could be very high.


















<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




