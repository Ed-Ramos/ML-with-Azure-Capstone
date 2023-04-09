
# Using Azure Machine Learning to predict mortality by heart failure

This capstone project is part of the Udacity Machine Learning with Azure nano degree. 
The goal of the project is to submit a machine learning experiment using the Azure hyperdrive functionality
and the AutoML functionality. Using the most accurate model, i then use the python SDK to deploy that model via
an azure container instance(ACI) service. The resulting endpoint is then tested to prove functionality.

## Dataset

### Overview

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million
lives each year, which accounts for 31% of all deaths worlwide.

Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to 
predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one 
or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease)
need early detection and management wherein a machine learning model can be of great help. The dataset was obtained 
from the kaggle website. The authors are Davide Chicco and Giuseppe Jurman.

### Task

The task is to create a model for predicting mortality caused by Heart Failure. The features are:
* Age: Age of person
* Anaemia: Decrease of red blood cells or hemoglobin (boolean)
* creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
* creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
* diabetes: If the patient has diabetes (boolean)
* ejection_fraction: Percentage of blood leaving the heart at each contraction (percentage)
* high_blood_pressure: If the patient has hypertension (boolean)
* platelets: Platelets in the blood (kiloplatelets/mL)
* serum_creatinine: Level of serum creatinine in the blood (mg/dL)
* serum_sodium: * Level of serum sodium in the blood (mEq/L)
* sex: Woman or man (binary)
* smoking: If the patient smokes or not (boolean)
* time: Follow-up period (days)

### Access

I access the data by first uploading it to my Github repository and then bringing it into the project
from Github. 

## Automated ML

The autoML setting were chosen as follows:
- experiment_timeout_minutes: to save on compute resources, using 20 minutes as this is sufficient for the purpose of this project
- primary_metric: Since this is a classification problem, using the accuracy metric which matches the metric used in hyperdrive experiment
- n_cross_validations: Using cross validation as a resampling method to use different portion of data to test and train the model on different iterations. Using n=5 is sufficient for this data size
- iteration_timeout_minutes: This setting guards against iterations that have issues and are taking too long to complete and thus saves compute resources
- max_concurrent_iterations:  This setting provides compute efficiency. 

### Results

When using Azure AutoML to obtain the best model given the provided dataset and parameters, it determined 
that the best model was the VotingEnsemble model. The accuracy obtained was .8730. This model combines
the predictions from five other models and assigns a weight of .2 to each of those models. However, the 
individual model with highest accuracy was an XGBoostClassifier model which used SparseNormalizer for data 
transformation. This model has quite a bit of hyperparameters like eta, gamma, max_depth_max_leaves and
n-estimators which must be tuned.

To improve results, one can try and increase the experiment_timeout parameter. This will allow the 
model more time to find better algorithms and parameters. Also, you can increase the iteration timeout
parameter if you see that some iterations were not completed. Lastly, one can exclude some models that
are noticed to not provide good results. By doing this, more time is spent on finding and tuning better
models. In the end, one must consider and understand the compute costs incurred by these changes and 
decide if its worth it.


RunDetails screenshot

![rundetails.png](..%2FScreenshots%2Frundetails.png)

best model screenshot with details

![best automl model trained with details.png](..%2FScreenshots%2Fbest%20automl%20model%20trained%20with%20details.png)

best model showing runid

![best model with its runid.png](..%2FScreenshots%2Fbest%20model%20with%20its%20runid.png)

## Hyperparameter Tuning

A LogisticRegression algorithm was used with dataset along with three user defined parameters, C, max_iter,
and intercept_scaling to develop the ML model. The C parameter was set with a uniform range of (.05, 1.0),
the max_iter was set with a choice between (5, 200), and the intercept_scaling was set with a 
uniform range of (.1, 1.0). A train script file is used along with other parameters and policies to 
create a Hyperdrive configuration.

The configuration is then used to run an experiment in Hyperdrive. The best run which maximized 
the accuracy primary metric is then saved to a file.

The parameter sampler method chosen is the RandomParameterSampling method. The advantage of this 
method is that it ensures that results obtained from the sample should approximate what would have 
been obtained if the entire population had been measured

In order to save on compute resources and cost, a Bandit early stopping policy was chosen. 
The policy early terminates any runs where the primary metric is not within the specified slack 
factor amount with respect to the best performing training run. An evaluation interval and delay 
evaluation argument was also specified. These affect the frequency for applying the metric and the 
number of intervals to delay the first evaluation, respectively

### Results

The model accuracy with the hyperdrive run was .767. 
To improve results, additional model parameters can be included in the tuning process. 
This may help in that not only the default values are used. Also a different parameter sampler can 
be tried. Lastly, one can change the early termination policy to make sure that runs are not 
being terminated prematurely.

Rundetails screenhot

![hyperdrive Rundetails.png](..%2FScreenshots%2Fhyperdrive%20Rundetails.png)

Best hyperdrive model

![best trial of hyperdrive job.png](..%2FScreenshots%2Fbest%20trial%20of%20hyperdrive%20job.png)

completed hyperdrive job

![completed hyperdrive job.png](..%2FScreenshots%2Fcompleted%20hyperdrive%20job.png)


## Model Deployment

The model that was deployed was the votingEnsemble model obtained from the AutoML run as described above.

The endpoint can be consumed by submitting a get request to the endpoint service and providing the 
appropriate input data as shown below:

![input data to test endpoint.png](..%2FScreenshots%2Finput%20data%20to%20test%20endpoint.png)


## Screen Recording

https://youtu.be/vV0sizDGnew


