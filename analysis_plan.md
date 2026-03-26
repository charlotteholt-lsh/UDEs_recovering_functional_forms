# Can we use machine learning techniques to learn the behavioural response to an epidemic?

**Aim:** See if we can use machine learning techniques to learn the behavioural response to an epidemic.

**Objectives:**
- Use universal differential equations and symbolic regression to recover simulated behavioural dynamics
- Investigate what/how much training data/conditions improve performance

**Background**
In contrast to statistical models, that primarily aim to fit the observed data well, mechanistic models aim to mathematically represent the dynamic behaviour within a system by using known physical, biological or chemical laws. In the context of infectious disease forecasting, mechanistic models are used to explain how and why a disease is spreading. There is the opportunity to integrate machine learning into mechanistic models, retaining the mechanistic structure of the system but enabling the model to use multiple data streams and big data to improve the accuracy of its prediction [Ye 2025].

Attempts to integrate machine learning into epidemiological forecasting include using physics-informed neural networks (PINNs), epidemiology-aware AI models (EAAMs), synthetically-trained AI models, and AI-augmented epidemiological models [Ye 2025]. Neural networks have been used to leverage the vast amounts of historical data at later stages of tan epidemic to estimate the epidemiological parameters which are then used in the model predictions [Nguyen 2022]. ~={red} add more examples here=~

However, most methods attempt to enhance epidemiological parts of the model, with limited attention given to socio-behavioural mechanisms [Ye 2025], despite the link between the population's behavioural response to an epidemic and how the disease was spreading becoming clear during the COVID-19 pandemic [Kuwahara 2023]. This coupled behaviour-disease dynamic is complex to model and there is a lack of data around how the population actually responds [Ryan 2024]. However, models that do not incorporate behavioural response typically underestimate $R_0$ and overestimate the final epidemic size. When behaviour is incorporated the model fits better to observed data and synthetic data experiments showed that this better fit, estimation of $R_0$ and epidemic size is a direct result of the inclusion [Pant 2025]. Behavioural dynamics are also important for non-pharmaceutical interventions (NPIs). During the first wave of the COVID-19 pandemic, NPIs were more effective than when introduced during the second wave of the pandemic and this is likely due to the population's behaviour not returning to the pre-pandemic baseline after the first wave. This highlights that people's behaviour changes even when it is not enforced [Sharma 2021]. One method of recovering and incorporating the behavioural feedback loop, is through universal differential equation (UDE) framework. → 

UDEs are differential equation systems that have universal approximators, such as neural networks, embedded within them. Known dynamics of the system can be explicitly included, and the neural networks can use data during training to then learn and approximate unknown components or processes [Rackauckas 2021]. This allows the advantages of machine learning and mechanistic modelling to be combined and to complement the flaws of the other by learning unknown components of the system in a data-driven way, but still being able to incorporate prior epidemiological knowledge, interpret the dynamics and assign epidemiological meaning to the parameters and transmission.

Their use has been investigated in the context of infectious disease forecasting; for example, mapping noisy wastewater surveillance data to reported case counts [Schmid 2021], approximating the force of infection from neighbouring regions to a target region [Rojas-Campos 2023], and learning the interaction between mobility and infections to predict the future trajectory of the epidemic and people's mobility patterns [Kuwahara 2023]. There is still scope to further investigate this coupled behaviour-disease dynamic and see if it is possible to learn the behavioural response using UDEs. Furthermore, we aim to represent this behaviour-disease interplay symbolically, to enhance interpretability and reduce the black box effect of traditional ML techniques.

**Methods**

Phase 1 - *practice/learning*:
- Generate synthesised data using an exponential form of beta
- Train UDE on synthesised data
- Tune hyperparameters so NN fits the data as best as possible
- Apply symbolic regression to recover symbolic form
- Compare to exponential form

Phase 2 - *method structure*:
- Generate multiple synthesised trajectories using different initial states/parameters values but keeping the beta functional form consistent *beta is a function of these initial parameters so beta will vary between trajectories*
- ~={red} Jointly (?) train the UDE on these synthesised trajectories ==multistart Schmid? and Rojas-Campos== **UNSURE HOW TO DO THIS**=~
- Use symbolic regression to recover symbolic form

Phase 3 - *comparisons*
- Use different functional forms of beta e.g. exponential / rational / mixed
- Vary the strength of the behavioural response e.g. $\zeta$ in [Pant 2025](file:///Users/lsh2502304/Zotero/storage/JWUM6YSZ/Pant%20et%20al.%20-%202025%20-%20The%20Paradox%20of%20Neglecting%20Changes%20in%20Behavior%20How%20Standard%20Epidemic%20Models%20Misestimate%20Both%20Transmi.pdf)
- Evaluate loss against each state (rather than just e.g. mortalities/infections)
- Optimise hyperparameters using the Julia equivalent of `Optuna` (tree-structured Parzen estimator [Schmid Page 17](zotero://open-pdf/library/items/TDC2ZXGG?page=17&annotation=LR827SDY))
- Encode hard constraints e.g. $\beta(t)$ monotonically decreasing in $\delta I(t)$ 
- Encode learning biases (as hard constraints and as biases)
- Do genetic programming vs SINDy (compare different types of symbolic regression)
- Vary the amount of data given in training to find minimum amount of data required to do symbolic regression
- Noisy data vs non-noisy data

**Results**

~={red} How am I going to present the comparisons? What tables would I include?

- Graph of simulated epidemics
![[Pasted image 20260325181733.png]]
[Rojas-Campos 2023, Page 6](zotero://open-pdf/library/items/WY2V2C8W?page=6&annotation=7ZC8S3AH)

- Graph of simulated epidemic vs NN prediction
![[Pasted image 20260325182156.png]]
[Kuwahara 2023, Page 4](zotero://open-pdf/library/items/FGMAEDAV?page=4&annotation=5TCXU392)

- Graph with beta learned by NN vs actual known beta function against *time* **not mobility as shown here**
![[Pasted image 20260325182335.png]]
[Kuwahara 2023, Page 5](zotero://open-pdf/library/items/FGMAEDAV?page=5&annotation=FZ8VNTGP)


- Graph of effectiveness of SINDy approximation:
![[Pasted image 20260316155307.png]]
[Rackauckas 2021, Page 9](zotero://open-pdf/library/items/Y5CAKV2E?page=9&annotation=MVIVQFVF)

- Graph of comparison of just UDE model, and UDE model and SINDy approximation:
![[Pasted image 20260325181921.png]]
[Rojas-Campos 2023, Page 14](zotero://open-pdf/library/items/WY2V2C8W?page=14&annotation=PIJU6YNP)

**Timeline**

Phase 1 - **10/04/26***
Question answered: Was the UDE able to learn the exponential form of beta from one model that was performed well on the training data?

Phase 2 - **30/04/26**
Question answered: can we use multiple trajectories to help the NN learn the beta function better? And can we perform SINDy on multiple trajectories at once?

Phase 3 - **15/07/26**
Question answered: In what situations is the model able to better recover the behavioural dynamics?
