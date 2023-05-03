# Databricks-Machine-learning
For the preparation of the dataset, a machine learning cluster called ML FLOW CLUSTER was create the cluster has a Machine Learning implementation runtime.  
 
 
The notebook for Machine Learning implementation is call “Oluwatobiloba_vaughan_ML FLOW” the language was select as default (PYTHON) and it is attached to the currently created machine learning runtime cluster which gives access this notebook.
  


FaultDataset .csv containing the twenty vibration sensor readings was imported to databricks file system.

#  Section 1
 
The FaultDatasetDF data frame reads a CSV file named "FaultDataset.csv" located in the "/FileStore/tables" directory. The spark variable is assumed to be the SparkSession object which is the entry point to programming Spark with the Dataset and DataFrame API. The read.csv function is called on the spark object with the following parameters: "/FileStore/tables/FaultDataset.csv": the path to the CSV file to be read, header="True": specifies that the first row in the CSV file contains the column names, inferSchema="True": specifies that Spark should infer the schema of the DataFrame by inspecting the data in the CSV file.
The resulting DataFrame will have the column names and data types inferred from the CSV file, and can be used for further data processing and analysis in PySpark.
 
RFormula transformer was used to pre-process the FaultDataset so that it is in the correct format to train an MLlibmodel., RFormula transformer it is a feature engineering tool that specifies a formula using R-like syntax to define the relationship between input features and output labels. The formula used in this case is "fault_detected ~ .", which means that the column "fault_detected" is the output label, and all other columns in the DataFrame are input features.The code first creates an instance of the RFormula transformer with the specified formula. Then, it fits the transformer to the FaultDatasetDF DataFrame and transforms the data using the formula. The transformed DataFrame is stored back into the FaultDatasetDF variable. Finally, the show() method is called to display the first five rows of the transformed DataFrame. This method is a convenient way to quickly inspect the data and ensure that the transformation was performed correctly.
 
The next step is to split the data into a training dataset and a test dataset. Some of the data set will be held back during the learning training of this model. so that some labelled data that the model hasn’t yet seen can be used to make predictions for this dataset in order for its performance on new data can be documented. In this case data was randomly split into 70% which is allocated to the training Data Frame and 30% to be held back as test data.




# Section 2
 
For the FaultDataset training a decision tree classifier model using the DecisionTreeClassifier estimator is used. First, the code imports the DecisionTreeClassifier estimator from the pyspark.ml.classification module. Then, an instance of the DecisionTreeClassifier estimator is created and stored in the dt variable. The labelCol parameter is set to "label", which specifies the name of the column that contains the labels or target variable, and the featuresCol parameter is set to "features", which specifies the name of the column that contains the input features.
After that, the estimator is used to train the model by calling the fit() method on it. The training data, represented by the trainingDF variable, is passed as an argument to the fit() method. The fit() method trains the decision tree classifier model on the training data and returns an instance of the model, which is stored in the model variable.
Once the model is trained, it is used to make predictions based on the dataset provided Having the resulting predictions stored in the predictions variable. The predictions DataFrame contains the original columns of the test data as well as additional columns added by the model, including a prediction column containing the predicted labels. 

 

# Section 3

Selection of hyperparameters and model training and evaluation and MLflowexperiment tracking.
 
So as to train the model with different values of the hyperparameters, the dataset is been split up again into training and validation datasets. The reason for splitting the second time is to train the model with each combination of parameters on the training dataset and then pick the best model using the results on the validation dataset.
 
 
For performing the hyperparameter tuning the fit() method use acheive this, after performing the hyperparameter tuning, the grid search is then performed finding the best performing model out of the models which was built .
 
# Brief discussion of result 


FaultDataset.csv. row contains twenty vibration sensor readings, and the final column identifies whether there was a fault with the machine at the time of the readings. In this column, the aim  of this report is to be able to classify whether there is a fault with the machine based on the readings from the vibration sensors, this will answer the following question: Load the dataset into a Spark Data Frame, Use MLlib to train a Decision Tree classification model (DecisionTreeClassifieralgorithm) on the provided data and evaluate its performance, Track your experiment with ML flow. The accuracy is 0.952432, which means that 95% of the predictions made by our model on the test dataset are correct, which seems like a good result, for the best hyperparameters there is a marginal improvement in the performance of the model after grid search to select the best hyperparameters was performed giving 0.9650450450450451 as the result.
