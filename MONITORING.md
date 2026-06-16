# Model Monitoring and Data Drift Analysis

This document evaluates production data integrity using the automated Evidently monitoring module to compare the baseline training distribution against simulated production data.

### 1. Which features showed drift and why?
The features that consistently exhibit significant statistical data drift are:
* **`age`**: The p-value dropped far below the 0.05 alpha threshold, signaling clear drift. This occurred due to an intentional demographic change simulating an aging population sample.
* **`cholesterol`**: Showed a structural upward distribution shift, simulating elevated cardiovascular baseline risks in incoming patient groups.
* **`chest_pain_type`**: Showed massive categorical distribution shift. The ratio of `asymptomatic` admissions jumped from 25% to 50% of overall throughput, significantly altering the composition of the categorical feature space.

Other parameters (e.g., `resting_blood_pressure`, `st_depression`) remained statistically stable, confirming the simulation isolated specific demographic changes.

### 2. Would this drift likely affect model performance?
**Yes, heavily.** The features experiencing data drift—specifically `age`, `cholesterol`, and `chest_pain_type` (asymptomatic status)—are the exact variables configured with high weights in our log-odds equations for predicting heart disease. 
Because the AdaBoost and RandomForest classifiers built their internal tree boundaries around younger, lower-cholesterol training boundaries, this upward distribution shift means production data falls outside the model's learned distribution. The model will likely experience severe underestimation biases, leading to an increase in dangerous false-negative classification errors.

### 3. What action would you recommend?
**Immediate Model Retraining is recommended.** 
Because the overall drift share reached **~27%–36%** (exceeding or sitting on the edge of our critical 30% pipeline gate), this is classified as a severe data drift event rather than an isolated anomaly. Continuing to monitor or investigating without updating the weights would expose the system to performance degradation. 

We should immediately schedule a retraining pipeline run (`src/train.py`) that incorporates a fresh, balanced training block containing these new patient profiles. This will update the model's decision boundaries before it encounters live production traffic.
