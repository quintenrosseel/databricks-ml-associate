# Exam Information 

See more info [here](https://www.databricks.com/learn/certification/machine-learning-associate). 
## Objectives
- Describe the learning context, format, and structure behind the exam.
- Describe the topics covered in the exam.
- Recognize the different types of questions provided on the exam.
- Identify resources to learn the material covered in the exam.

## Target Audience
- Data scientist
- Beginner-level, practitioner certification
- Assess candidates at a level equivalent to six months of experience with machine learning with Databricks

## Expectations
-   Use Databricks Machine Learning and its capabilities within machine learning workflows
-   Implement correct decisions in machine learning workflows
-   Implement machine learning solutions at scale using Spark ML and other tools
-   Understand advanced scaling characteristics of classical machine learning models


## Not expected 
- Advanced ML Operations: Webhooks, Automation, Deployment, Monitoring, CLI/REST APIs
- Advanced ML Workflows: Target Encoding, Embeddings, Deep Learning, Natural Language Processing

## Notes
- Through Kryterion
- Automatically graded

## Basic Details
-  Time allotted to complete exam = 1.5 hours (90 minutes)
-  Passing scores = At least 70% on the overall exam
-  Exam fee = $200
-  Retake policy = As many times as you want, whenever you want
-  Number of Ouestions = 45
-  More info. on the Databricks Academy FAQ:  http://files.training.databricks.com/Ims/docebo/databricks-academy-fag.odf
Select all of the following resources that will be available during the exam.
1. Databricks documentation
2. Paper and pencil
3. A single, index card of pre-written notes
4. A running Spark session
5.  **None of the above**


# Certification topics

**Use Databricks Machine Learning and its capabilities within machine learning workflows (29%), including:** 
-  Databricks Machine Learning (clusters, Repos, Jobs)
	- Clusters
		- Types of clusters
		- Single-node use cases
		- Standard cluster use cases
	- Repos
		- Connect a repo from an external git provider
		- Commit changes
		- Create a new branch
		- Pull changes from an external Git provider
	- Jobs
		- Orchestrate multi-task ML workflows
-  Databricks Runtime for Machine Learning (basics, libraries)
	- Basics
		- Create a cluster with MLR
		- Identify differences between the standard DBR and MLR
	- Library Usage
		- Install a Python library in notebook scope
		- Install a Python library in cluster scope
-  AutoML (classification, regression, forecasting)
	- Classification
		- Common steps in the workflow
		- How to locate source code
		- Evaluation metrics
		- Data exploration
	- Regression
		- Common steps in the workflow
		- How to locate source code
		- Evaluation metrics
		- Data exploration
	- Forecasting
		- Common steps in the workflow
		- How to locate source code
-  Feature Store (basics)
	- Basics
		- Benefits
		- Create a feature store table
		- Write data to a feature store table
	- Pipelines
		- Train a model with features from a feature store table
		- Score a model using features from a feature store table
-  MLflow (Tracking, Model Registry)
	- Experiment tracking
		- Querying past runs
		- Logging runs
		- UI information
	- Model registry
		- Register a model
		- Transition a model across stages

Note, there are self assessment questions to check your understanding. 

**ML Workflows (29%) Implement correct decisions in machine learning workflows, including**:
-   Exploratory data analysis (summary statistics, outlier removal)
	- Summary stats
		- Compute summary statistics for a Spark DataFrame
		-  DataFrame.summary)
		-  dbutils.data.summarize()
	- Outlier removal
		- Removing outlier features
		-  Filtering records in a Spark DataFrame
-   Feature engineering (missing value imputation, one-hot-encoding)
	- Missing Values
		- Binary indicator features
		- Identifying the optimal replacement value
		- Imputer with Spark ML
	- One-hot-encoding
		- Complications with certain algorithms
		- OHE with Spark ML
-   Tuning (hyperparameter basics, hyperparameter parallelization)
	- Hyperparameter basics
		- Grid Search vs. Random Search
		- Tree of Parzen Estimators
		- Scikit-learn
	- Hyperparamter parallelization
		- Hyperopt applications
		- Relationship between selection algorithm and parallelization
-   Evaluation and selection (cross-validation, evaluation metrics)
	- Cross-validation
		- Number of trials
		- Train-validation split vs. cross-validation
		- Using Spark ML to accomplish the above 
	- Evaluation metrics
		- Recall
		- Precision
		- F1
		- Log-scale interpretation
 
Note, there are self assessment questions to check your understanding. 

**Usage of SparkML (33%) Implement machine learning solutions at scale using Spark ML and other tools, including**:
- Distributed ML Concepts
	- Difficulties
		- Data location and shuffling
		- Data fitting on each core for parallellization
	- Spark ML
		- No UDF requirement - when to use UDF; when to use Spark ML. Relationship with other libraries.
- Spark ML Modeling APIs (data splitting, training, evaluation, estimators vs. transformers, pipelines)
	- Prep
		- Splitting data
		- Reproducible splits
	- Modeling
		- Fitting
		- Feature vector columns
		- Evaluators
		- Estimators vs. transformers
	- Pipelines
		- Relationship with Cross validation (e.g. inside or outside pipeline)
		- Relationship with training and test data 
- Hyperopt
	- Basics
		- Bayesian hyperparameter abilities
		- Parallelization abilities
	- Applications
		- SparkTrials sv Trials
		- Relationship between number of evaluations and level of parallelization 
- Pandas API on Spark
	- Concepts
		- InternalFrame
		- Metadata storage
	- Benefits
		- Easy refactoring for scale
		- Pandas API
	- Usage
		- Importing 
		- Converting between DF types
- Pandas UDFs and Pandas Function APIs
	- Conversion
			- Apache arrow (why is this efficient?)
			- Vectorization
	- Pandas UDFs - why would you use them? 
		- Iterator UDF benefits
		- For scaled prediction
	- Pandas Function APIs
		- Group-specific training
		- Group-specific inference


Note, there are self assessment questions to check your understanding. 

**Scaling ML models (9%) - Understand advanced scaling characteristics of classical machine learning models, including **:
- Distributed Linear Regression
	- Identify what type of solver is used for big data and linear regression
	- Identify the family of techniques used to distribute linear regression
- Distributed Decision Trees
	- Describe the binning strategy used by Spark ML for distributed decision trees
	- Describe the purpose of the maxBins parameter
- Ensembling Methods (bagging, boosting)
	- Basics
		- Combining models
		- Implications of multi-model solutions
	- Types
		- Bagging
		- Boosting 
		- Stacking

Note, there are self assessment questions to check your understanding. 



# Practice & preparation
- Practice exam is coming soon
- Some practice questions are shown at the end of the course. 