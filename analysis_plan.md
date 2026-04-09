# Can we use machine learning techniques to learn the behavioural response to an epidemic?

**Aim:** See if we can use machine learning techniques to learn the underlying behavioural response to an epidemic.

**Objectives:**
- Phase 1: Create ideal conditions that allow universal differential equations and symbolic regression to recover a feedback loop of simulated endogenous behavioural dynamics.
- Phase 2 
	- Part 1: Investigate how and at what threshold the framework degrades and the the UDE framework can no longer successfully recover the behavioural dynamics.
	- Part 2: Investigate if it is possible to restore performance via hyperparameter tuning or adjusting the methodology.

**Background**

In contrast to statistical models, that primarily aim to fit the observed data well, mechanistic models aim to mathematically represent the dynamic processes within a system by using known physical, biological or chemical laws. In the context of infectious disease forecasting, mechanistic models are used to explain how and why a disease is spreading [17]. 

During the COVID-19 pandemic, many mechanistic models were developed, but their success in accurately forecasting the pandemic trajectory was varied [4]. It became clear that there was an intrinsic link between the population's behavioural response to the pandemic and how the disease was spreading [6], but determining the underlying relationship and how to incorporate this response into the models was challenging. Typically, the behaviour was included exogenously via public health guidance or mobility data, however, when incorporated in this way, the behavioural response to the epidemic dynamics are ignored. Accounting for the behavioural response endogenously (when the behaviour is represented as a function of another time-dependent variable within the model and is able to alter the epidemiological dynamics [4]) was identified as an important way to improve forecasting in the post-COVID evaluation [11].

This coupled behaviour-disease dynamic is complex to model and there is a lack of data around how the population actually responds [13]. However, it has been shown that models which do not incorporate behavioural response typically underestimate $R_0$ and overestimate the final epidemic size. When behaviour is incorporated the model fits better to observed data and synthetic data experiments showed that this better fit, estimation of $R_0$ and epidemic size is a direct result of the inclusion of behavioural response [4, 9]. Behavioural dynamics are also important for non-pharmaceutical interventions (NPIs). During the first wave of the COVID-19 pandemic, NPIs were more effective than when introduced during the second wave of the pandemic and this is likely due to the population's behaviour not returning to the pre-pandemic baseline after the first wave. This highlights that people's behaviour changes even when it is not enforced [15]. 

There are three main approaches to incorporate behaviour endogenously; introducing a feedback loop ( a time-varying transmission rate that depends on other states), using game/utility theory (the transmission rate is affected by the introduction of interventions and the populations' choice of whether to adhere to them), and considering information/opinion spread (movement of individuals to and from a protected class if they adhere/reject an NPI, the choice they make can depend on the risk perception) [4].

Introducing a feedback loop, where a behavioural response is triggered by the prevalence of the disease, is a relatively straightforward approach to incorporating endogenous behaviour. Analyses have shown that a relatively simple compartmental model with a feedback loop could predict COVID-19 deaths as well as the CDC ensemble [11], and that including the infection rate as a function of prevalence resulted in less error than when this feedback loop was excluded [7].

In recent years, there has been a particular focus on exploring the opportunity to integrate machine learning into mechanistic models, retaining the mechanistic structure of the system but enabling the model to use multiple data streams and big data to improve the accuracy of its prediction. Attempts to integrate machine learning into epidemiological forecasting include using physics-informed neural networks (PINNs), epidemiology-aware AI models (EAAMs), synthetically-trained AI models, and AI-augmented epidemiological models. However, most methods attempt to enhance epidemiological parts of the model, with limited attention given to socio-behavioural mechanisms [17]. 

Neural networks have been used to leverage the vast amounts of historical data at later stages of an epidemic to estimate the epidemiological parameters which are then used in the model predictions [8]. *add more examples here*

However, a relatively new approach to embed neural networks within the mechanistic structure of the dynamical system via a universal differential equation (UDE) framework has emerged. UDEs are differential equation systems that have universal approximators, such as neural networks, embedded within them. Known dynamics of the system can be explicitly included, and the neural networks can use data during training to then learn and approximate unknown components or processes [10]. This allows the advantages of machine learning and mechanistic modelling to be combined and to complement the flaws of one another by learning unknown components of the system in a data-driven way, while still being able to incorporate prior epidemiological knowledge, interpret the dynamics and assign epidemiological meaning to the parameters and transmission.

Their use has been investigated in the context of infectious disease forecasting; for example, mapping noisy wastewater surveillance data to reported case counts [14], and approximating the force of infection from neighbouring regions to a target region [12]. *add more examples here e.g. [39] and [53] from [6]*

To further improve the utility of the UDE framework, it is possible to use symbolic regression to represent the neural network approximation in a functional form [1, 10]. This enhances interpretability and reduces the black box effect of traditional ML techniques. Moreover, it can give us insights into the unknown underlying mechanisms that we are investigating, allowing us to learn some process or component within the system. 

The exploration of the coupled behaviour-disease dynamic using the UDE framework has not been extensively researched. The only example at the time of writing is in [6], they wanted to investigate the link between the population's response during the COVID-19 pandemic, and the various pandemic waves observed. The timing of interventions made it clear that a large first wave led to strict interventions being implemented, the number of infections decreasing and then restrictions being relaxed leading to a second wave. They used UDEs to learn the interaction between mobility and infections to predict the future trajectory of the epidemic and people's mobility patterns by training the UDE on mobility and infection prevalence data. They then qualitatively compared the performance of the framework when learning biases were included as additional objectives for the neural network to optimise over.

We want to further investigate this behaviour-disease interplay by introducing a purely endogenous feedback loop to represent the risk-driven behaviour of the population, and seeing if, and under what conditions, it is possible to learn the behavioural response via symbolically recovering the time-varying transmission rate. The primary objective is to correctly learn a mechanistically meaningful expression for the behavioural response, as a result, we would expect this to improve predictive ability of the model, however, this is not explored in depth here.

To learn the prevalence-dependent behaviour we will use a UDE framework, then we will use symbolic regression techniques to recover a functional form for the behavioural response, which will provide information about the underlying behavioural response during an epidemic. We will only attempt to represent the endogenous behavioural response as a feedback loop represented by a transmission rate that depends on other time-varying states within the model. We will not attempt to simulate more complicated theoretical behavioural models *add more examples here*, however, we believe it would be possible to introduce a UDE framework in these cases. For example, in [13] they use a health belief model to describe factors that contribute towards an individual undertaking a protective behaviour. Future work could investigate if it is possible to learn the rate of behaviour uptake and abandonment in the health belief model, and recover the symbolic form of this relationship with parameter values for the factors contributing towards performing protective behaviour.

**Methods**

***Phase 1: create ideal conditions for a symbolic UDE framework***

1. Create a compartmental model framework that incorporates endogenous behaviour by introducing a feedback loop. This feedback loop will be represented by a time-varying transmission rate that varies depending on disease prevalence. We will model the transmission rate as a monotonically decreasing function of the number of infections, reflecting adaptive protective behaviours in the population. 
2. Generate multiple epidemic trajectories using various epidemiological parameters and initial conditions, keeping the functional form of the transmission rate constant across all simulations.
3. Train a neural network approximator for the transmission rate within a UDE framework; optimising a combined loss across the multiple synthesised trajectories. This will allow the neural network to learn a general and transferable representation of the underlying behavioural mechanism without learning the transmission rate specific to a particular outbreak trajectory. It will give the neural network more opportunities to see how behaviour responds to the disease dynamics, improving its ability to learn the coupled behaviour-disease interplay.
4. Use symbolic regression to recover the symbolic form of the transmission rate.
5. Compare the recovered functional form for the transmission rate to the function we chose to generate the trajectories [2].
6. *Is it possible to then recover the parameters for each individual trajectory? in [[chesebroScientificMachineLearning2025]] they conclude that converge to 3dp after an additional optimisation step to tune the parameters (can do further optimisation if parameters are still incorrect e.g. simulation based inference)*
7. Adjust quality of data, initial conditions and optimise the hyperparameters to create ideal conditions for the neural network training. If the neural network is unable to recover the functional form of the transmission rate in these ideal conditions, then we will conclude that an alternative approach for recovering the feedback loop should be used.
	We will adjust:
	1. the infectious disease model generating the data and the neural network hyperparameters until the functional form learned by the neural network visually recovers the unknown dynamics, with a sufficiently low mean-squared error (MSE) between the neural network approximation and the true transmission rate. We may also encode hard constraints on the neural network [5] or introduce learning biases as additional optimiser objectives during training [6].*do I need to specify now what I mean by sufficiently low...?*
	2. the symbolic regression methodology choice and hyperparameters until the output of the regression matches the functional form for the transmission rate that we used to generate the synthetic data.

***Phase 2: investigate limitations***

In Phase 2, we will take the ideal conditions established in Phase 1 as a baseline and systematically investigate how the framework degrades as we move into a real-world conditions.

We will undertake an ablation study, focusing on gaining insight into the problem in addition to trying to maximise the neural network's ability to recover the time-varying transmission rate. The ablation will consist of two parts. The first will look at the data quality and structure, and how adjusting this affects the UDE framework's ability to recover the true functional form for the transmission rate. The second will investigate whether targeted fixes, including hyperparameter tuning in the neural network training and symbolic regression, can restore performance.

We will structure the ablation into a sequence of rounds. Each round will be centred around a specific goal that will help us gain insight into the problem, whilst incrementally making improvements to the configuration. 

1. *Data quality and structure*

We propose investigating the effect of the following and recording at what point the symbolic form recovered by regression does not match the generating function (in each round, we will keep all other data quality and structure and neural network architecture and hyperparameters the same as the ideal conditions found in Phase 1):
- Increasing the noisiness of data to establish the minimum noise level that the framework is still able to recover the functional form.
- Varying the amount of synthetic epidemic trajectories given in training to establish the minimum amount of example interactions the neural network requires to correctly recover the underlying behaviour-disease relationship.
- Varying the amount of observed time points for each trajectory to establish the minimum amount of data required for accurate symbolic recovery.
- Using different functional forms for the transmission rate e.g. exponential / rational / mixed, to establish whether the framework can identify different behavioural mechanisms.
- Varying the strength of the behavioural response e.g. $\zeta$ in [9], to establish the minimum detectable signal.

When the framework degrades, we will then need to identify whether this is due to the neural network approximation of the transmission rate, or the method of symbolic regression used. We will determine this by evaluating the mean squared error between the neural network's learned transmission rate and the true generating function, and comparing the recovered functional form to the true functional form.

*Potentially investigate pairwise combinations, maybe assessing importance using `ANOVA` to decide which pairs to evaluate? [14]*

2. *Hyperparameter tuning to improve the neural network approximation of the transmission rate*

Taking the rounds from Phase 2 Part 1 where the performance of the framework degraded due to a failure in the neural network's ability to accurately approximate the transmission rate, we will investigate whether optimising the neural network configuration is able to restore the UDE's ability to recover the functional form. We will follow the guidance given in [5]. However, due to the computationally intensive process of training the UDE, we will not undertake a quasi-random search of the nuisance parameter search space, but rather move straight to Bayesian optimisation techniques, as in [14].

In each round we will:
1. Define a clear goal e.g. "Can changing the activation function allow us to recover  the transmission rate when the behavioural response is weak"
2. Select the scientific hyperparameter that we want to investigate e.g. the activation function.
3. Set all other nuisance hyperparameters to their value determined in Phase 1.
4. Define the nuisance hyperparameter search space.
5. Optimise over the nuisance hyperparameters using [Optuna.jl](https://github.com/una-auxme/Optuna.jl) (Julia package that provides an API interface for the `Optuna` hyperparameter optimisation framework) [16].
6. For each scientific hyperparameter, select the best performing nuisance hyperparameter configuration and compare the best trial for each scientific hyperparameter.

We will evaluate performance using mean squared error (MSE) between the neural network's learned transmission rate and the true generating function. We will also record the computational time required to recover the symbolic form in each setting to assess feasibility of real-time use.

If the fix improves performance and we are able to recover the true transmission rate, we will adopt the fix going into future rounds. If the proposed fix does not improve the approximation, we will state the limitations and opportunity for future work or refinements of the method.

We will investigate adjusting the following as proposed fixes:
- Neural network regularisation
- Neural network input
- Neural network depth
- Neural network width
- Activation function
- Final activation function
- ODE solver
- Loss function
- Optimiser
- Encoded hard constraints
- Inclusion of learning biases

3. *Hyperparameter tuning to improve the symbolic regression performance*

If in Phase 2 Part 1, the neural network was accurately recovering the transmission rate but the symbolic regression was failing to recover the true functional form, then we will investigate whether tuning the symbolic regression hyperparameters or using alternative regression methods improves performance.

We will use the same framework outlined in Phase 2 Part 2, but investigate the following proposed fixes identifying a successful fix as when the structural form output of symbolic regression and the true functional form match:
- Shrinking cut-off
- Basis
- Optimiser
- BIC selector
- GPSINDy
- Genetic algorithm

**Results**

*Phase 1:*
- Graph of simulated epidemic vs NN prediction
- Graph with transmission rate learned by NN vs actual known transmission rate function against time
- Graph of effectiveness of SINDy approximation e.g. in [10]
- Figure of comparison of synthesised data, just the UDE model, and the UDE and SINDy model

*Phase 2 Part 1:*
- Experimental grid with thresholds where performance degrades

| Condition enforced                                               | Threshold at which symbolic regression is unable to recover functional form | MSE between neural network approximation of transmission rate and true generating function at threshold | MSE between recovered transmission rate and true generating function at threshold |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Noisiness of data                                                |                                                                             |                                                                                                         |                                                                                   |
| Number of synthetic trajectories used in neural network training |                                                                             |                                                                                                         |                                                                                   |
| Number of time points used in neural network training            |                                                                             |                                                                                                         |                                                                                   |
| Different behavioural functional form                            |                                                                             |                                                                                                         |                                                                                   |
| Varying the strength of the behavioural response                 |                                                                             |                                                                                                         |                                                                                   |

*Phase 2 Part 2 & 3*:
Experimental grid with quantitative comparisons 

| Condition under which the framework failed | Scientific hyperparameter | Studies  | Best performing trial configuration | MSE between neural network approximation and true generating function |
| ------------------------------------------ | ------------------------- | -------- | ----------------------------------- | --------------------------------------------------------------------- |
| e.g. Varying the behavioural response      | e.g. Activation function  | sigmoid  |                                     |                                                                       |
|                                            |                           | tanh     |                                     |                                                                       |
|                                            |                           | gelu     |                                     |                                                                       |
|                                            |                           | softplus |                                     |                                                                       |
|                                            |                           |          |                                     |                                                                       |
|                                            |                           |          |                                     |                                                                       |

Summary of targeted fixes that recover performance lost in Phase 2 Part 1.

| Condition enforced | Description of fix that recovers performance | MSE before fix | MSE after fix |
| ------------------ | -------------------------------------------- | -------------- | ------------- |
|                    |                                              |                |               |
|                    |                                              |                |               |
# References

[1] Brunton, S.L., Proctor, J.L. and Kutz, J.N. (2016) ‘Discovering governing equations from data by sparse identification of nonlinear dynamical systems’, _Proceedings of the National Academy of Sciences_, 113(15), pp. 3932–3937. doi:10.1073/pnas.1517384113.

[2] Chesebro, A.G. _et al._ (2025) _Scientific machine learning of chaotic systems discovers governing equations for neural populations_. Available at: https://arxiv.org/html/2507.03631v3#S8 (Accessed: 08 April 2026).

[3] Godbole, V. _et al._ (2023) _A playbook for systematically maximizing the performance of deep learning models._, _GitHub_. Available at: https://github.com/google-research/tuning_playbook/tree/main (Accessed: 09 April 2026).

[4] Hamilton, A. _et al._ (2024) ‘Incorporating endogenous human behavior in models of COVID-19 transmission: A systematic scoping review’, _Dialogues in Health_, 4, p. 100179. doi:10.1016/j.dialog.2024.100179.

[5] Hyunho, K. (2025) _A Novel Architecture for Integrating Shape Constraints in Neural Networks_ [Preprint]. Available at: https://openreview.net/forum?id=Nd0dt1B5Ec.

[6] Kuwahara, B. and Bauch, C.T. (2024) ‘Predicting covid-19 pandemic waves with biologically and behaviorally informed universal differential equations’, _Heliyon_, 10(4). doi:10.1016/j.heliyon.2024.e25363. 

[7] Menda, K. _et al._ (2021) ‘Explaining COVID-19 outbreaks with reactive SEIRD models’, _Scientific Reports_, 11(1). doi:10.1038/s41598-021-97260-0.

[8] Nguyen, D.Q. _et al._ (2022) ‘Becaked: An explainable artificial intelligence model for covid-19  forecasting’, _Scientific Reports_, 12(1). doi:10.1038/s41598-022-11693-9. 

[9] Pant, B. _et al._ (2025) ‘The paradox of neglecting changes in behavior: How standard epidemic models Misestimate both transmissibility and final epidemic size’, _MedRxiv_ [Preprint]. doi:10.64898/2025.12.07.25341782. 

[10] Rackauckas, C. _et al._ (2020) ‘Universal differential equations for scientific machine learning’, _arXiv_ [Preprint]. doi:10.21203/rs.3.rs-55125/v1. 

[11] Rahmandad, H., Xu, R. and Ghaffarzadegan, N. (2022) ‘Enhancing long-term forecasting: Learning from covid-19 models’, _PLOS Computational Biology_, 18(5). doi:10.1371/journal.pcbi.1010100.

[12] Rojas-Campos, A., Stelz, L. and Nieters, P. (2023) ‘Learning COVID-19 Regional Transmission Using Universal Differential Equations in a SIR model’, _arXiv_ [Preprint]. doi:arXiv:2310.16804. 

[13] Ryan, M. _et al._ (2024) ‘A behaviour and disease transmission model: Incorporating the health belief model for human behaviour into a simple transmission model’, _Journal of The Royal Society Interface_, 21(215). doi:10.1098/rsif.2024.0038. 

[14] Schmid, N. _et al._ (2026) _Wastewater-informed neural compartmental model for long-horizon case number projections_[Preprint]. doi:10.64898/2026.02.10.26345731. 

[15] Sharma, M. _et al._ (2021) ‘Understanding the effectiveness of government interventions against the resurgence of COVID-19 in Europe’, _Nature Communications_, 12(1). doi:10.1038/s41467-021-26013-4. 

[16] Una-Auxme (no date) _Optuna.jl_, _GitHub_. Available at: https://github.com/una-auxme/Optuna.jl (Accessed: 09 April 2026).

[17] Ye, Y. _et al._ (2025) ‘Integrating artificial intelligence with mechanistic epidemiological modeling: A scoping review of opportunities and challenges’, _Nature Communications_, 16(1). doi:10.1038/s41467-024-55461-x.

