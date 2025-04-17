---
layout: post
title: AB Testing
date: 2025-04-02
---

In this post, we will first go over the set up of a one-sided hypothesis test, then we will provide intuition and practical significance of concepts like Type I error, power, minimum detechtable effect etc.

Imagine this: the business team comes up with an idea for a new feature and hypothesises that adding this feature will improve the conversion rate $$ (\theta) $$. They come to you and ask for your help to collect sample data to empirically evaluate whether the hypothesis is reasonable.

In a perfect world, we could measure the true effect of a treatment by giving the entire population on Earth the control condition and recording the eventual outcome. Then, on a perfect clone of Earth, we would give everyone the treatment and measure that outcome one more time. Comparing these two results would give us a definitive, black-and-white answer.

Of course, this isn’t possible in reality — we can’t run experiments on parallel Earths. So instead, we randomly assign individuals within a sample population to either control or treatment groups. This randomness introduces variability in the results, which is why we use statistical distributions to model and analyze the uncertainty in our estimates.

&nbsp;


Now, since the business team is concerned with whether the new feature is an improvement, we need to use a **one-sided test**. We first form the Null ($$H_0$$) and Alternative hypothesis ($$H_A$$), and note that $$H_0$$ and $$H_A$$ are always compliments to each other. That is if Null is false, Alternative must be true.



$$
\begin{aligned}
H_0\!:\ & \theta_A \leq \theta_0 \quad \text{(null hypothesis)} \\
H_A\!:\ & \theta_A > \theta_0 \quad \text{(alternative hypothesis)}
\end{aligned}
$$

We can denote the difference in the 2 conversion rate as: $$ \Delta = \theta_0 - \theta_A $$. Then the hypothesis become


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

1) either the Null hypothesis ($$H_0$$) is true or the alternative hypothesis ($$H_A$$) is true.
2) We can either decide to reject the Null or reject the Alternative.

We can treat these as 2 random variables and visualise them with a 2 by 2 grid.

<img width="760" alt="Screenshot 2025-04-16 at 1 55 38 PM" src="https://github.com/user-attachments/assets/cf9fac54-9451-45af-8de7-8b6bcbff7fba" />

In our one sided test however, we operate under the assumption that the Null hypothesis is true, that is, we condition on it. This corresponds to vertically summing the probabilities.

<img width="804" alt="Screenshot 2025-04-16 at 1 48 25 PM" src="https://github.com/user-attachments/assets/046f20b6-21ac-42c1-9f04-f1099bc67946" />

### There are exactly 4 possible outcomes 
1. Given H₀ is true and we correctly do not reject it  
2. Given H₀ is true and we incorrectly reject it (**Type I error**)
3. Given H₀ is false and we incorrectly do not reject it  (**Type II error**)
4. Given H₀ is false and we correctly reject it

We now formally define the following:

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

$$ * $$ is the threshold for our decision, $$\Delta_{H_0}$$ and $$\Delta_{H_A}$$ is 0 and the true improvement (usually the minimum detectable effect in practice), respectively. 



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

In the left figure, an error is made if we decide to reject the Null hypothesis, which occurs when our observation lies to the right of our threshold - $$*$$. While in the right figure, an error is made if we decide to NOT reject the Null, which occurs when our observation lies to the left of our threshold $$*$$. 

**Remark:** Observe that we can reduce the probability of Type II error by shifting the threshold / decision boundary to the right. However, doing so would increase the Type I error, since a larger critical region would make it easier to reject the null hypothesis.







&nbsp;
## 3 levers that affect power

1) $$ \alpha $$: To increase power, we can increase $$ \alpha $$, but again, this is a trade-off between making a Type I error (false positive) and a Type II error (false negative).

2) $$ n $$: You can increase the sample size. Both the Null and Alternative distributions in the above diagram will become narrower in shape, leading to less overlap between the 2 distributions. 

3) $$ \Delta_{H_A} $$: If the believed true parameter is further away from 0, the area to the left of the alternative distribution decreases, and as a result power increases.

&nbsp;
### Practically, how do we pick what value of $$\Delta$$ to use for the alternative hypothesis in the calculation of power?

Thus far, we have omitted the discussion of what $$\Delta_{H_A}$$ is. Given a significance level α, we are typically interested in determining the sample size required to achieve a desired statistical power. However, the specific value of the alternative hypothesis, which is needed in our power calculation, is unknown.

A common choice is to plug in the **minimum detectable effect (MDE)**. This is the minimum effect size $$\Delta_{H_A}$$ that would be considered meaningful for the business problem on hand. By using this value for $$\Delta_{H_A}$$ , we’ve assured ourselves that the calculated sample size is big enough to detect the smallest effect we care about (and importantly, as well as anything bigger). That is, if the true $$\Delta_{H_A}$$ is smaller than the MDE we selected, our test design would not have sufficient power. In the other case where the true $$\Delta_{H_A}$$ is bigger than the MDE we selected, our test will have a surplus of power.

We will leave the sample size calculation in the appendix of this post for interested readers.

&nbsp;
### Distinction of $$P(\text{data} | H_0)$$ vs $$P(H_0 | \text{data})$$

#### 1) $$ \text{P-value} \neq P(H_0 | data) $$
It is a common misconception that a p-value of $$5\%$$ means the null hypothesis has a $$5\%$$ chance of being true $$ P(H_0 \mid \text{p-value} = 0.05) = 0.05 $$. 

We will show how the 2 probabilities $$P(H_0 \mid \text{data})$$ and the p-value are related through the use of Bayes' theorem:

$$
\begin{aligned}
P(H_0 \mid \text{data}) &= \frac{ P(\text{data} \mid H_0) \times P(H_0)}{P(\text{data})} \\
&= \text{p-value} \times \frac{P(H_0)}{P(\text{data})}
\end{aligned}
$$

$$ P(H_0 |Data) = \frac{ P(Data|H_0)P(H_0)}{P(Data|H_0)P(H_0) + P(Data|H_1)P(H_1)} $$

It does align with our intuition, though, that a lower p-value suggests that the null hypothesis is less likely to be true.

&nbsp;
#### 2) Why is power a big deal? 

It is because experiments with low statistical power are NOT trustworthy. That is, even if the p-value is statistically significant, if the experiment is low-powered to begin with there is still a high probability that the Null hypothesis is true.

We will introduce the concept of False positive risk, and show how it relates to p-value.

**False positive risk (FPR)** is the probability that the Null hypothesis is true while we decide to reject the null hypothesis (p-value < 5%). Mathematically it is represented by $$ P(H_0 \mid \text{p-value} < 0.05) $$ Clearly, a higher risk is worse.


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


If the power ($$1- \beta$$) is low, the denominator will be small and the entire fraction (FPR) will be big. This implies that when power is low, the probability that the Null hypothesis is true is small (even when p-value is statistically significant). Yes the p value is small enough where we will decide to reject the Null hypothesis, but this decision might not be trustworthy.


## Appendix:
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
















<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




