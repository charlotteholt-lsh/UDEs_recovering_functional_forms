# Can we use machine learning techniques to learn the behavioural response to an epidemic?

**Aim:** See if we can use machine learning techniques to learn the behavioural response to an epidemic.

**Objectives:**
- Phase 1: Use universal differential equations and symbolic regression to recover simulated behavioural dynamics
- Phase 2: Investigate what training conditions improve performance of the framework, and how little data or noisy the data can be for the UDE to successfully recover the behavioural dynamics?

**Background**
In contrast to statistical models, that primarily aim to fit the observed data well, mechanistic models aim to mathematically represent the dynamic behaviour within a system by using known physical, biological or chemical laws. In the context of infectious disease forecasting, mechanistic models are used to explain how and why a disease is spreading. There is the opportunity to integrate machine learning into mechanistic models, retaining the mechanistic structure of the system but enabling the model to use multiple data streams and big data to improve the accuracy of its prediction [10].

Attempts to integrate machine learning into epidemiological forecasting include using physics-informed neural networks (PINNs), epidemiology-aware AI models (EAAMs), synthetically-trained AI models, and AI-augmented epidemiological models [10]. Neural networks have been used to leverage the vast amounts of historical data at later stages of tan epidemic to estimate the epidemiological parameters which are then used in the model predictions [3]. ~={red} add more examples here=~

However, most methods attempt to enhance epidemiological parts of the model, with limited attention given to socio-behavioural mechanisms [10], despite the link between the population's behavioural response to an epidemic and how the disease was spreading becoming clear during the COVID-19 pandemic [2]. This coupled behaviour-disease dynamic is complex to model and there is a lack of data around how the population actually responds [7]. However, models that do not incorporate behavioural response typically underestimate $R_0$ and overestimate the final epidemic size. When behaviour is incorporated the model fits better to observed data and synthetic data experiments showed that this better fit, estimation of $R_0$ and epidemic size is a direct result of the inclusion [4]. Behavioural dynamics are also important for non-pharmaceutical interventions (NPIs). During the first wave of the COVID-19 pandemic, NPIs were more effective than when introduced during the second wave of the pandemic and this is likely due to the population's behaviour not returning to the pre-pandemic baseline after the first wave. This highlights that people's behaviour changes even when it is not enforced [9]. One method of recovering and incorporating the behavioural feedback loop, is through universal differential equation (UDE) framework.

UDEs are differential equation systems that have universal approximators, such as neural networks, embedded within them. Known dynamics of the system can be explicitly included, and the neural networks can use data during training to then learn and approximate unknown components or processes [5]. This allows the advantages of machine learning and mechanistic modelling to be combined and to complement the flaws of the other by learning unknown components of the system in a data-driven way, but still being able to incorporate prior epidemiological knowledge, interpret the dynamics and assign epidemiological meaning to the parameters and transmission.

Their use has been investigated in the context of infectious disease forecasting; for example, mapping noisy wastewater surveillance data to reported case counts [8], approximating the force of infection from neighbouring regions to a target region [6], and learning the interaction between mobility and infections to predict the future trajectory of the epidemic and people's mobility patterns [2]. There is still scope to further investigate this coupled behaviour-disease dynamic and see if it is possible to learn the behavioural response using UDEs. Furthermore, we aim to represent this behaviour-disease interplay symbolically, to enhance interpretability and reduce the black box effect of traditional ML techniques.

**Methods**

*Phase 1: symbolic UDE framework*

1. Generate multiple epidemic trajectories using various epidemiological parameters and initial conditions, choosing the time-varying transmission rate to be a function that reflects behavioural response to infections (constant across all simulations).
2. Train a neural network approximator for the transmission rate within a UDE framework; optimising a combined loss across the multiple synthesised trajectories ~={red}need to find an example in the literature - is this what [7] did with the multistart maybe?=~.
3. Use symbolic regression to recover the symbolic form of the transmission rate.
4. Compare the recovered functional form for the transmission rate to the function we chose to generate the trajectories ~={red}comparison method here currently unclear, need to investigate)=~

*Phase 2: investigate limitations*

If the results of Phase 1 are successful ~={red}depending on what we define as 'successful'=~, then we will want to investigate at what point the framework 'breaks' and we are unable to recover the functional form that we input. If the results of Phase 1 are unsuccessful ~={red}again, what does 'unsuccessful' mean?=~, then we want to see if and how we can create ideal conditions for the neural network to learn the function well and how the symbolic regression method can be improved to better recover the transmission rate's functional form. 

We are looking to investigate the limitations of the framework:
- if it isn't working, what can we introduce to improve performance?
- when it breaks, what can we introduce to fix it?

Potential things to investigate:
- Noisiness of data
- Effect of using different functional forms for the transmission rate e.g. exponential / rational / mixed
- Vary the strength of the behavioural response e.g. $\zeta$ in [3]
- Evaluate loss against each state (rather than just e.g. mortalities/infections)
- Optimise hyperparameters using the Julia equivalent of `Optuna` (tree-structured Parzen estimator e.g. [7])
- Encode hard constraints e.g. $\beta(t)$ monotonically decreasing in $\delta I(t)$ 
- Encode learning biases (as hard constraints and as biases) e.g. in [1]
- Do genetic programming vs GPSINDy vs SINDy (compare different types of symbolic regression)
- Vary the amount of data given in training to find minimum amount of data required to do symbolic regression

**Results**
- Table with quantitative comparisons e.g. the different functional forms
- Graph of simulated epidemics
- Graph of simulated epidemic vs NN prediction
- Graph with transmission rate learned by NN vs actual known transmission rate function against time
- Graph of effectiveness of SINDy approximation e.g. in [5]
- Figure of comparison of synthesised data, just the UDE model, and the UDE and SINDy model
# References

[1] Hyunho, K. (2025) _A Novel Architecture for Integrating Shape Constraints in Neural Networks_ [Preprint]. Available at: https://openreview.net/forum?id=Nd0dt1B5Ec.

[2] Kuwahara, B. and Bauch, C.T. (2024) ‘Predicting covid-19 pandemic waves with biologically and behaviorally informed universal differential equations’, _Heliyon_, 10(4). doi:10.1016/j.heliyon.2024.e25363. 

[3] Nguyen, D.Q. _et al._ (2022) ‘Becaked: An explainable artificial intelligence model for covid-19  forecasting’, _Scientific Reports_, 12(1). doi:10.1038/s41598-022-11693-9. 

[4] Pant, B. _et al._ (2025) ‘The paradox of neglecting changes in behavior: How standard epidemic models Misestimate both transmissibility and final epidemic size’, _MedRxiv_ [Preprint]. doi:10.64898/2025.12.07.25341782. 

[5] Rackauckas, C. _et al._ (2020) ‘Universal differential equations for scientific machine learning’, _arXiv_ [Preprint]. doi:10.21203/rs.3.rs-55125/v1. 

[6] Rojas-Campos, A., Stelz, L. and Nieters, P. (2023) ‘Learning COVID-19 Regional Transmission Using Universal Differential Equations in a SIR model’, _arXiv_ [Preprint]. doi:arXiv:2310.16804. 

[7] Ryan, M. _et al._ (2024) ‘A behaviour and disease transmission model: Incorporating the health belief model for human behaviour into a simple transmission model’, _Journal of The Royal Society Interface_, 21(215). doi:10.1098/rsif.2024.0038. 

[8] Schmid, N. _et al._ (2026) _Wastewater-informed neural compartmental model for long-horizon case number projections_[Preprint]. doi:10.64898/2026.02.10.26345731. 

[9] Sharma, M. _et al._ (2021) ‘Understanding the effectiveness of government interventions against the resurgence of COVID-19 in Europe’, _Nature Communications_, 12(1). doi:10.1038/s41467-021-26013-4. 

[10] Ye, Y. _et al._ (2025) ‘Integrating artificial intelligence with mechanistic epidemiological modeling: A scoping review of opportunities and challenges’, _Nature Communications_, 16(1). doi:10.1038/s41467-024-55461-x.
