---
layout: post
title: Why Power Matters in AB Testing- A Dive into Statistical Significance
date: 2025-04-02
---

In this post, we will first go over the setup of a one-sided hypothesis test, then we will provide intuition and practical significance of concepts like Type I error, power, minimum detectable effect, etc.

**Imagine this**: the business team comes up with an idea for a new feature and hypothesises that adding this feature will improve the conversion rate $$ (X) $$. They come to you and ask for your help to collect sample data to empirically evaluate whether the hypothesis is reasonable.

In a perfect world, we could measure the true effect of a treatment by giving the entire population on Earth the control condition and recording the eventual outcome. Then, on a perfect clone of Earth, we would give everyone the treatment and measure that outcome one more time. Comparing these two results would give us a definitive, black-and-white answer.

Of course, this isn’t possible in reality — we can’t run experiments on parallel Earths. So instead, we randomly assign individuals within a sample population to either control or treatment groups. This randomness introduces variability in the results, which is why we use statistical distributions to model and analyze the uncertainty in our estimates.

&nbsp;


Now, since the business team is concerned with whether the new feature is an improvement, we need to use a **one-sided test**. We first form the Null ($$H_0$$) and Alternative hypothesis ($$H_A$$), and note that $$H_0$$ and $$H_A$$ are always compliments to each other. That is if Null is false, Alternative must be true.



$$
\begin{aligned}
H_0\!:\ & X_A \leq X_0 \quad \text{(null hypothesis)} \\
H_A\!:\ & X_A > X_0 \quad \text{(alternative hypothesis)}
\end{aligned}
$$

We can denote the difference in the 2 conversion rate as: $$ \Delta = X_0 - X_A $$. Then the hypothesis becomes


$$
\begin{aligned}
H_0\!:\ & \Delta \geq 0 \quad \text{(null hypothesis)} \\
H_A\!:\ & \Delta < 0 \quad \text{(alternative hypothesis)}
\end{aligned}
$$

&nbsp;
Our final goal is always the same: After collecting data, we will decide whether we **1)  reject $$H_0$$** (and implicitly accept $$H_A$$) OR **2)  do not reject $$H_0$$** (and implicitly decline $$H_A$$).

&nbsp;
#### Below are the high-level steps we need to perform
- We define a test statistic T, which is a random variable (r.v.)
- We conduct the one-sided test, **under the assumption that the Null hypothesis is true**; consequently, the test statistic T follows a particular distribution that we call the **null distribution**
- We calculate t, the observed value for the test statistic r.v.
- We evaluate the extremity of t relative to the null distribution
- If the observed t is unlikely to be generated from the null distribution, we have more evidence to believe against $$H_0$$, and if it passes a certain threshold ($$\alpha$$), we will decide to reject the null hypothesis.

**Remark** even when we reject the null hypothesis, it is possible that the observed outcome actually came from the Null distribution, in which case we have incorrectly rejected the null hypothesis; this is denoted as a Type I error, which we will discuss later.
- Statistically, we formalise the extremity of t using the p-value. 

&nbsp;

Before we define p-value formally, recall by design

1. Either the Null hypothesis $$H_0$$ is true or the Alternative hypothesis $$H_A$$ is true.

2. We can either decide to reject the Null or reject the Alternative.


We can treat these as 2 random variables and visualise them with a 2 by 2 grid.

<img width="760" alt="Screenshot 2025-04-16 at 1 55 38 PM" src="https://github.com/user-attachments/assets/cf9fac54-9451-45af-8de7-8b6bcbff7fba" />

In our one sided test however, we operate under the assumption that the Null hypothesis is true, that is, we condition on it. This corresponds to vertically summing the probabilities.

<img width="804" alt="Screenshot 2025-04-16 at 1 48 25 PM" src="https://github.com/user-attachments/assets/046f20b6-21ac-42c1-9f04-f1099bc67946" />

### There are exactly 4 possible outcomes 
1. Given H₀ is true and we correctly do not reject it  
2. Given H₀ is true and we incorrectly reject it (**Type I error**)
3. Given H₀ is false and we incorrectly do not reject it  (**Type II error**)
4. Given H₀ is false and we correctly reject it

### We now formally define the following:

1. **P-value** is defined as the probability of observing a more extreme test statistic, under the assumption that the **null hypothesis is true**.  

   $$
   \text{p-value} = P\left( |T| \geq t_{\alpha} \mid \text{Null hyp is true} \right)
   $$

2. **Type I error ($$ \alpha $$):** Incorrectly rejecting the null hypothesis when it is indeed true.

3. **Power (1−β):** The probability of correctly rejecting the null hypothesis under the assumption that the **alternative hypothesis is true**. Equivalently, it is the probability of **not** making a Type II error.  

   $$
   \text{Power} = P\left( \text{p-value} < \alpha \mid \text{Alternative hyp is true} \right)
   $$

4. **Type II error (β):** Failing to reject the null hypothesis when the alternative hypothesis is actually true.


&nbsp;
&nbsp;
#### Let's incorporate everything we have discussed into the 2 visual diagrams below. 

- $$ * $$ is the threshold for our decision to reject the null hypothesis or not
- $$\Delta_{H_0}$$ is 0 
- $$\Delta_{H_A}$$ is the "true improvement" (usually the minimum detectable effect in practice), respectively. 



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

In the left figure, an error is made if we decide to reject the Null hypothesis, which occurs when our observation lies to the right of our threshold  $$*$$. While in the right figure, an error is made if we decide to NOT reject the Null, which occurs when our observation lies to the left of our threshold $$*$$. 

**Remark:** Observe that we can reduce the probability of Type II error by shifting the threshold/decision boundary to the right. However, doing so would increase the Type I error, since a larger critical region would make it easier to reject the null hypothesis.

&nbsp;


#### In the remaining post, we will focus on understanding the properties of power, how to calculate sample size given a desired level of power, and why a low power would be worrisome for hypothesis testing.




&nbsp;
## 3 levers that affect power

1) $$ \alpha $$: To increase power, we can increase $$ \alpha $$ (shift the decision boundary), but again, this is a trade-off between making a Type I error (false positive) and a Type II error (false negative).

2) $$ n $$: You can increase the sample size. Geometrically, what this means is both the Null and Alternative distributions in the above diagram will become narrower in shape, leading to less overlap between the 2 distributions and greater power.

3) $$ \Delta_{H_A} $$: As $$ \Delta_{H_A} $$ gets further away to the right from 0, the area to the left of the alternative distribution decreases, and power increases.

&nbsp;
### Practically, how do we pick what value of $$\Delta$$ to use for the alternative hypothesis in the calculation of power?

Thus far, we have omitted the discussion of what the value of $$\Delta_{H_A}$$ should be. All we know is that $$\Delta_{H_A}$$ should be greater than 0. But as we saw in the 2 diagrams above, we need a specific value to determine the power of the hypothesis test.

In practice, what happens is we are given a significance level α, and we are interested in determining the sample size required to achieve a desired statistical power. However, the specific value of the alternative hypothesis, which is needed in our power calculation, is unknown.

A common choice is to use the **minimum detectable effect (MDE)**. This is the minimum effect size $$\Delta_{H_A}$$ that would be considered meaningful for the business problem. 
So in our conversion improvement example, suppose the business team told you that a 1% of conversion improvement (absolute) is needed to breakeven for the associated cost with implementing this new feature, our MDE would be 1%. Then we would calculate the sample size required, substituting the MDE as the value of $$\Delta_{H_A}$$. 

If the "true $$\Delta$$" is larger than this MDE, by property 3), our test will still have enough power to detect the effect size. On the other hand, if the "true $$\Delta$$" is smaller than this MDE, our test will NOT have enough power, but we do not care about this small improvement anyways.


All that is to say: by using this value for $$\Delta_{H_A}$$ , we’ve assured ourselves that the calculated sample size is big enough to detect the minimum effect we care about (and vacously, any effect size that is bigger). For any effect size that are smaller than then MDE, we would have insufficient power when drawing our conclusions.

We will leave the sample size calculation in the appendix of this post for readers that are interested.

&nbsp;
### Distinction of $$P(\text{data} | H_0)$$ vs $$P(H_0 | \text{data})$$

#### **1) $$ \text{P-value} \neq P(H_0 | data) $$**
It is a common misconception that a p-value of $$5\%$$ means the null hypothesis has a $$5\%$$ chance of being true. In mathematical terms: $$ P(H_0 \mid \text{p-value} = 0.05) = 0.05 $$. 

We will show how the 2 probabilities $$P(H_0 \mid \text{data})$$ and the p-value are related through the use of Bayes' theorem. Importantly, we would need to know the prior probability of the null hypothesis being true.

$$
\begin{aligned}
P(H_0 \mid \text{data}) &= \frac{ P(\text{data} \mid H_0) \times P(H_0)}{P(\text{data})} \\
&= \text{p-value} \times \frac{P(H_0)}{P(\text{data})} \\
P(H_0 |Data) &= \frac{ P(Data|H_0)P(H_0)}{P(Data|H_0)P(H_0) + P(Data|H_1)P(H_1)}
\end{aligned}
$$


**Note**: It does align with our intuition, though, that a lower p-value suggests that the null hypothesis is less likely to be true.


Geometrically, what we have done is sum the probabilities across the horizontal direction.

<img width="805" alt="Screenshot 2025-04-23 at 11 23 58 AM" src="https://github.com/user-attachments/assets/ce485c3b-572d-4f11-8496-6f7127f97703" />

&nbsp;
#### **2) Why is power a big deal?** 

It is because experiments with low statistical power are NOT trustworthy. That is, even if the p-value is statistically significant, if the experiment is low-powered to begin with there is still a high probability that the Null hypothesis is true.

We will introduce the concept of False positive risk, and show how it relates to p-value.

**False positive risk (FPR)** is the probability that the Null hypothesis is true while we decide to reject the null hypothesis (p-value < 5%). Mathematically it is represented by $$ P(H_0 \mid \text{p-value} < 0.05) $$. Clearly, a higher FPR is worse.


Let
- SS be a statistically significant result i.e. p-value < 0.05
- $$\alpha$$ be the threshold to declare statistical significance.
- $$\beta$$ be the Type II error rate.
- $$\pi$$ be the prior probability of the null hypothesis.


$$
\begin{aligned}
\text{FRP} &= \\
P(H_0 \mid SS) &= P(SS \mid H_0) \cdot \frac{P(H_0)}{P(SS)} \\
&= \frac{ P(SS \mid H_0) P(H_0)}{P(SS \mid H_0) P(H_0) + P(SS \mid H_1) P(H_1)} \\
&= \frac{\alpha \cdot \pi}{\alpha \cdot \pi + (1-\beta)(1-\pi)}
\end{aligned}
$$


If the power ($$1- \beta$$) is low, the denominator will be small and the entire fraction (FPR) will be big. This implies that when power is low, the probability that the Null hypothesis is true is small (even when p-value is statistically significant). 

We see that the p-value could be extremely small where we do decide to reject the Null hypothesis, yet this decision might not be trustworthy.

&nbsp;
&nbsp;
## Appendix:
### Derivation of sample size needed for a desired power, given $$\alpha$$ and a specific choice of $$ \delta_A >0 $$ (which as discussed earlier would be the MDE)

Consider the following setup:
- We are still interested in the conversion rate $$X_0$$ , $$X_1$$ - the customer either successfully converted or failed to convert, and this follows a Bernoulli
- We substitute the value of MDE for the true $$\Delta_A$$
- $$ \bar{X}_0 \sim N(\mu_0, \sigma_0^2) $$ with sample size $$n_0$$
- $$ \bar{X}_1 \sim  N(\mu_1, \sigma_1^2)$$ with sample size $$n_1$$.
- In this example, we assume 50/50 split, so $$ n_0 =n_1 = n$$
- $$\Delta$$ = $$\bar{X}_1$$- $$\bar{X}_0$$ $$\sim$$ $$N(\mu_1, \frac{\sigma_1^2}{n_1}) - N(\mu_0, \frac{\sigma_0^2}{n_0}) = N\left(\mu_1 - \mu_0, \frac{\sigma_0^2}{n_0} + \frac{\sigma_1^2}{n_1} \right)$$
- Recall in general, the sample variance for the mean of a Bernoulli variable would be $$ \sigma^2 =  \frac{\mu (1-\mu)}{n} $$
- $$\mu_0$$ is the baseline conversion
- $$\mu_1$$ = $$\mu_0 + \text{absolute MDE}$$
- $$\mu$$ = $$ \frac{\mu_0+\mu_1}{2} $$


**Goal** Again, we wish to obtain a power of 1- $$\beta$$. That is, we want to reject the null at least 100(1- $$\beta$$)% of the time, under the assumption that the alternative hypothesis is true.

$$
1 - \beta \leq P\left( Z > z_{\alpha} \, \Big| \, \text{alternative} \right) 
= P\left( \frac{ \widehat{\mu_1} - \widehat{\mu_0} }{ \sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right) } } > z_{\alpha} \, \Big| \, \text{alternative} \right)
$$



**Note 1** The z score follows a standard normal only under the Null hypothesis, and not under the alternative. We have to rearrange the above expression to obtain something normally distributed.
   
**Note 2** Under the alternative hypothesis we can't use the pooled variance $$\sqrt{\mu(1 - \mu) \left( \frac{1}{n} + \frac{1}{n} \right)}$$  but instead $$ \text{Var}(\widehat{\mu_1} -  \widehat{\mu_0}) = \frac{\mu_1(1- \mu_1)}{n_1} +  \frac{\mu_0(1-\mu_0)}{n_0} $$.

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

Afterwards, we continue the algebraic manipulation and isolate for n to arrive at the following final expression:


$$n = \frac{ \left( Z_{1 - \alpha} \cdot \sqrt{2\mu(1 - \mu)} + Z_{1 - \beta} \cdot \sqrt{\mu_1(1 - \mu_1) + \mu_2(1 - \mu_2)} \right)^2 }{(\text{MDE})^2} $$



&nbsp;
## Reference 

1. [A/B Testing intuition Busters](https://openreview.net/pdf?id=tC3reuvALn)

2. [Hypothesis Testing Explained (How I Wish It Was Explained to Me)](https://medium.com/data-science/hypothesis-testing-explained-how-i-wish-it-was-explained-to-me-14379f7a41c3)








<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




