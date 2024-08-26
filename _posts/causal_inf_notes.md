---
layout: post
title: Causal Inference Notes
date: 2024-08-25
---


Consistency:
for each individual, one of the counterfactual outcomes is actually factual (one that corresponds to the treatment value the indiv received)
The counterfactual outcome is equal to his observed outcome and can be expressed as $Y =Y^A$ where $Y^A$ denotes the counterfactual $Y^a$ evaluated at the observed treatment value a.

Problem with I.C.E
ICE is defined as a contrast of the values of counterfactual outcomes, but only one of those outcomes is observed for each individual. All counterfactual outcomes other than the one 
corresponding to the treatment value actually experienced remain unobserved. Hence individual effects cant be identified

We go to 1.2 average causal effects
Average Causal Effect in the population: An ACE of treatment A on outcome Y is present if 
$$ Pr[Y^{a=1} =1] \ineq Pr[Y^{a=0} = 1] $$ in the population of interest (this is only for binary outcomes)

the more general one is $$ E[y^{a=1}] \ineq E[y^{a=0}] $$

Aside: absence of ACE does not imply absence of ICE

1.4 Random variability (havnt read)

1.5 Association vs causation

Association: 2 variables are associated if information about one tells you something about the value of other
Causation: 2 variables are causally related if manipulating the level of one has the potential to change the level of others (holding all else fixed)

causal effect for an individual: The treatment A has a causal effect on an individual's outcome Y if $Y^{a=1} \ineq Y^{a=0}$ for the individual

$ E[Y^{1}_{i}]$ is generally distinct from $ E[Y_i|A_i =1] $
former is defined as a prior and averaged across the entire population (only god knows)
Latter is defined after treatment assignment,  and the average among those who receive treatment A_i = 1




