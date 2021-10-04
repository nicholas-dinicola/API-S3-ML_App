# Model Card
* Model Developed by: Nicholas Di Nicola
* Date: 04-10-2021 
* Version: 0.0.1
* Type: Clasifier
* Dataset Used: https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details
* The model is a Random Forest Classifier with 200 estimators and a maximum depth of 5. 
Its hyperparameters have been tuned using the Grid Search - CV technique. 

## Intended Use
The model's objective is to predict if the salary/income of an individual is above or below 
a threshold of 50k $, given fifteen different variables: 
* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Training Data
The training set consists of 26048 instances. Furthermore, after applying the One-Hot Encoder 
we ended up having 108 features as predictors, while the outcome variable (ie, lable) is represented by the 
salary. 

## Evaluation Data
The testing set consists of 6513 instances. Furthermore, after applying the One-Hot Encoder 
we ended up having 108 features as predictors, while the outcome variable (ie, lable) is represented by the 
salary. 

## Metrics
The metrics the model have been evaluated on are the F-score, precision and recall. 
Therefore, the trained and fine-tuned model obtained the following score on the respective metrics: 
* Precision: 0.798810703666997
* Recall: 0.798810703666997
* Fbeta: 0.6299335677999218

## Ethical Considerations

## Caveats and Recommendations


For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf