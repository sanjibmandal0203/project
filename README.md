# Cardiovascular disease detection using Multi-Model data and Important Feature Identification using XAI 

Cardiovascular disease requires urgent diagnostic solutions, with deep learning models like CNNs offering powerful automatic feature learning from multi-modal cardiac data including ECG and PCG signals, though their black-box nature limits clinical adoption. Explainable AI techniques such as SHAP address this challenge by providing interpretable insights into model decision-making, enabling clinicians to understand which features most significantly influence predictions and thereby increasing trust in AI-assisted diagnostics. 

**Keywords: ML, CNN, MS-CNN**

# Table of Contents

- [1. Data set](#Data_set)
- [2. Methodology](#Methodology)
    - [2.1 Experiment 1](#Experiment_1)
    - [2.2 Experiment 2](#Experiment_2)
    - [2.3 Experiment 3](#Experiment_3)
- [3. Results and Discussion](#Results_and_Discussion)
    - [Results of the Implementation](#Results_of_the_Implementation)
    - [Comparison with the previous models](#Comparison_with_the_previous_models)
 
## 1. Data set <a name="Data_set"> </a>

* The RAW ECG sample: The initial procedure was acquiring raw ECG data. Each signal records the value of 1975 patients and each signal is 8 seconds at the sampling frequency of 2000 Hertz. This implies that the signal of any one patient has 16,000 data points (because 8 seconds is the same as 2000 samples/sec divided by 8, or 16,000 samples). Thus, the first ECG data matrix is 16000 x 1975 
<img width="70%" alt="Raw ECG Sample" src="https://github.com/sanjibmandal0203/project/blob/project1/info/ECG/RAW%20ECG%20sample.png">

* The RAW PCG sample: The duration of each recording was 8 s and 1000 Hz sampling frequency used, that is, there were 8000 samples per channel per patient. Thus, per channel, the total data matrix was of dimensions 8000 x 1975 with time points as row and different patients as a column.
<img width="70%" alt="Raw PCG Sample" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20RAW.png">

<a href="https://doi.org/10.5281/zenodo.4263528"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4263528.svg" alt="DOI"></a>


## 2. Methodology <a name="Methodology"> </a>

**Work Flow**

<div align="center">
    <img width="90%" alt="work flow" src="https://github.com/sanjibmandal0203/project/blob/project1/info/Model/flow.jpg">
</div> 
exm

### 2.1 Experiment 1 (On ECG data)<a name="Experiment_1"> </a>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precidance

**2.1.1 Resample The RAW ECG sample:**

<img width="70%" alt="Resample The RAW ECG sample" src="https://github.com/sanjibmandal0203/project/blob/project1/info/ECG/ECG%20Resample.png">

**2.1.2 Implement Machine Learning Algorithm for classification on ECG feature (2048 feature):**

| Algorithm	             | Training Accuracy | Testing Accuracy  |
|------------------------|-------------------|-------------------|
| Logistic Regression	 | 92.72%	         | 52.66%            |
| K-Nearest Neighbors	 | 48.73%	         | 52.15%            |
| Support Vector Machine | 97.34%	         | 52.66%            |
| Naïve Bayes	         | 70.95%	         | 51.65%            |
| Decision Tree	         | 100.00%	         | 52.91%            |
| Random Forest	         | 100.00%	         | 59.24%            |
| XGBoost	             | 100.00%	         | 61.01%            |

<img width="50%" alt="classification on ECG feature (2048 feature)" src="https://github.com/sanjibmandal0203/project/blob/project1/info/ECG/ECG%20Resample%20classification%20results.png">

**2.1.3 Implement 8-layer Convolution Nural Network (CNN) for feature extraction on ECG feature (2048 feature):**

<div align="center">
    <img width="40%" hight ="50%" alt="ECG feature extraction" src="https://github.com/sanjibmandal0203/project/blob/project1/info/Model/ECG%20feature.png">
</div>

**2.1.4 Implement Machine Learning Algorithm for classification on ECG feature (128 feature):**

| Algorithm	             | Training Accuracy | Testing Accuracy  |
|------------------------|-------------------|-------------------|
| Logistic Regression	 | 60.13%	         | 58.48%            |
| K-Nearest Neighbors	 | 86.08%	         | 77.22%            |
| Support Vector Machine | 53.61%	         | 50.63%            |
| Naïve Bayes	         | 62.28%	         | 59.75%            |
| Decision Tree	         | 100.00%	         | 62.28%            |
| Random Forest	         | 100.00%	         | 70.89%            |
| XGBoost	             | 100.00%	         | 72.66%            |

<img width="50%" alt="classification on ECG feature" src="https://github.com/sanjibmandal0203/project/blob/project1/info/ECG/Ecg%20new%20feature%20classification%20results.png">
exm1

### 2.2 Experiment 2 <a name="Experiment_2"> </a> 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
precidance 2

**2.2.1 Resample The RAW PCG sample:**

Channel 1<div align="center">
<img width="70%" alt="PCG_Channel_1" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20channel_1.png">
</div>

Channel 2<div align="center">
<img width="70%" alt="PCG_Channel_2" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20channel_2.png">
</div>

Channel 3<div align="center">
<img width="70%" alt="PCG_Channel_3" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20channel_3.png">
</div>

Channel 4<div align="center">
<img width="70%" alt="PCG_Channel_4" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20channel_4.png">
</div>

**2.2.2 Implement 5-layer Convolution Nural Network (CNN) for feature extraction on PCG feature (2048 feature):**

The PCG channels, resampled, were given to fully differentiated 1D Convolutional Neural Networks (CNN) with a depth of 5 layers. The time-domain and frequency-domain features were precisely obtained by these networks by automatically extracting them in the signals. Every CNN was set to take as input 2048 samples and give out a compressed feature vector with 32 features per patient.
<div align="Center">
    <img width="40%" hight ="50%" alt="PCG feature extraction" src="https://github.com/sanjibmandal0203/project/blob/project1/info/Model/PCG%20feature.png">
</div>

**2.2.3 Implement Machine Learning Algorithm for classification on channel_1 PCG feature (32 feature):**

| Algorithm	            | Training Accuracy     |Testing Accuracy
|-----------------------|-----------------------|---------------
|Logistic Regression	| 59.49%	            |55.19%
|K-Nearest Neighbors	| 73.23%	            |52.91%
Support Vector Machine	| 59.87%	            |53.16%
Naïve Bayes	            | 58.54%	            |54.43%
Decision Tree	        | 100.00%	            |47.09%
Random Forest	        | 100.00%	            |56.96%
XGBoost	                | 100.00%	            |57.72%

<img width="50%" alt="classification on channel_1 PCG feature (32 feature)" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20channel_1%20classification.png">

**2.2.4 Implement Machine Learning Algorithm for classification on PCG feature (128 feature):**

| Algorithm	                | Training Accuracy| Testing Accuracy    
|---------------------------|------------------|-----------------
| Logistic Regression	    |60.51%	           | 55.19%
| K-Nearest Neighbors	    |73.23%	           | 52.91%
| Support Vector Machine	|60.76%	           | 55.44%
| Naïve Bayes	            |58.54%	           | 54.43%
| Decision Tree	            |100.00%	       | 48.10%
| Random Forest	            |100.00%	       | 57.22%
| XGBoost	                |100.00%	       | 57.72%

<img width="50%" alt="classification on PCG feature (128 feature)" src="https://github.com/sanjibmandal0203/project/blob/project1/info/PCG/PCG%20all%20channel%20128%20classification.png">

### 2.3 Experiment 3 <a name="Experiment_3"> </a>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 3. Results and Discussion <a name="Results_and_Discussion"> </a>
exm
### 2.1 Results of the Implementation <a name="Results_of_the_Implementation"> </a>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

exm1
### 2.2 Comparison with the previous models <a name="Comparison_with_the_previous_models"> </a>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
exm2
