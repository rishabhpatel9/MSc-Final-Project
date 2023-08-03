Here is the rewritten code in a code block with proper comments explaining what each code block is doing:

```python
# Import the necessary libraries
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Load the FERET dataset and create a dataframe with image paths and labels
dataset_path = "FERET/"
images = []
labels = []
for folder in os.listdir(dataset_path):
  for file in os.listdir(dataset_path + folder):
    images.append(dataset_path + folder + "/" + file)
    labels.append(folder)
df = pd.DataFrame({"image": images, "label": labels})

# Define a function to preprocess the images: resize, grayscale, histogram equalization
def preprocess_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (64, 64))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.equalizeHist(image)
  image = image / 255.0
  image = np.expand_dims(image, axis=2)
  return image

# Apply the preprocessing function to the images and convert the labels to numeric values
X = np.array([preprocess_image(image) for image in df["image"]])
y = np.array([0 if label == "male" else 1 for label in df["label"]])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a binary label dataset from the train set for aif360
train_dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                   df=pd.DataFrame(X_train.reshape(-1, 64*64)),
                                   label_names=["label"],
                                   protected_attribute_names=["gender"],
                                   labels=y_train,
                                   unprivileged_protected_attributes=[0])

# Compute the disparate impact and statistical parity difference metrics for the train set
metric_dataset = BinaryLabelDatasetMetric(train_dataset,
                                          unprivileged_groups=[{"gender": 0}],
                                          privileged_groups=[{"gender": 1}])
print("Disparate impact:", metric_dataset.disparate_impact())
print("Statistical parity difference:", metric_dataset.statistical_parity_difference())

# Apply the reweighing algorithm to mitigate bias in the train set
RW = Reweighing(unprivileged_groups=[{"gender": 0}], privileged_groups=[{"gender": 1}])
train_dataset_rw = RW.fit_transform(train_dataset)

# Compute the metrics for the reweighted train set
metric_dataset_rw = BinaryLabelDatasetMetric(train_dataset_rw,
                                             unprivileged_groups=[{"gender": 0}],
                                             privileged_groups=[{"gender": 1}])
print("Disparate impact after reweighing:", metric_dataset_rw.disparate_impact())
print("Statistical parity difference after reweighing:", metric_dataset_rw.statistical_parity_difference())

# Extract the features and labels from the reweighted train set
X_train_rw = train_dataset_rw.features.reshape(-1, 64, 64, 1)
y_train_rw = train_dataset_rw.labels.ravel()

# Define a convolutional neural network model for gender classification
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile and fit the model on the reweighted train set
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
history = model.fit(X_train_rw, y_train_rw,
                    batch_size=32,
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[early_stopping])

# Plot the training and validation accuracy and loss curves
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(X_test).round().ravel()
test_acc = accuracy_score(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)
print("Test accuracy:", test_acc)
print("Test confusion matrix:")
print(test_cm)

# Create a binary label dataset from the test set for aif360
test_dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                  df=pd.DataFrame(X_test.reshape(-1, 64*64)),
                                  label_names=["label"],
                                  protected_attribute_names=["gender"],
                                  labels=y_test,
                                  unprivileged_protected_attributes=[0])

# Compute the classification metrics for the test set
metric_classifier = ClassificationMetric(test_dataset,
                                         test_dataset.copy(),
                                         unprivileged_groups=[{"gender": 0}],
                                         privileged_groups=[{"gender": 1}])
print("Balanced accuracy:", metric_classifier.balanced_accuracy())
print("Equal opportunity difference:", metric_classifier.equal_opportunity_difference())
print("Average odds difference:", metric_classifier.average_odds_difference())
print("Theil index:", metric_classifier.theil_index())
```

This code is for a gender classification project using face recognition. It uses the FERET dataset and creates a dataframe with image paths and labels. The images are preprocessed by resizing, grayscaling, and histogram equalization. The data is then split into train and test sets. A binary label dataset is created from the train set for aif360 to compute the disparate impact and statistical parity difference metrics. The reweighing algorithm is applied to mitigate bias in the train set and the metrics are computed again for the reweighted train set. A convolutional neural network model is defined for gender classification and compiled and fit on the reweighted train set. The training and validation accuracy and loss curves are plotted. The model is evaluated on the test set and a binary label dataset is created from the test set for aif360 to compute classification metrics.

I hope this helps! 😊








--------------------------------------------------------------------------------------------------------






1. To detect algorithmic bias in your model, you need to measure how well your model performs on different subgroups of your data, such as different genders, races, ages, etc. 
You can use various fairness metrics to quantify the bias, such as accuracy, precision, recall, false positive rate, false negative rate, etc. 
You can also use tools like SageMaker Clarify or AI Fairness 360 to automate the bias detection process. 
For example, using SageMaker Clarify, you can define a `BiasConfig` object that specifies the sensitive attributes (such as gender) and the label attribute (such as male or female) of your data.
Then you can use the `clarify_processor.run_pre_training_bias` method to run a pre-training bias analysis on your data and generate a report that shows the bias metrics for each sensitive attribute group. Here is some sample code:

```python
# Import the required libraries
import sagemaker
from sagemaker import clarify

# Define the input and output locations
data_s3_uri = 's3://your-bucket/your-data.csv' # The location of your data file on S3
bias_report_output_path = 's3://your-bucket/bias_report/' # The location where you want to save the bias report

# Define the sensitive and label attributes
bias_config = clarify.BiasConfig(label_values_or_threshold=[1], # Assuming 1 is male and 0 is female in your data
                                 facet_name='Gender', # Assuming Gender is the column name for the sensitive attribute in your data
                                 facet_values_or_threshold=[1]) # Assuming 1 is male and 0 is female in your data

# Define the Clarify processor
clarify_processor = clarify.SageMakerClarifyProcessor(role=sagemaker.get_execution_role(),
                                                      instance_count=1,
                                                      instance_type='ml.m5.xlarge',
                                                      sagemaker_session=sagemaker.Session())

# Run the pre-training bias analysis
clarify_processor.run_pre_training_bias(data_config=clarify.DataConfig(s3_data_input_path=data_s3_uri,
                                                                       s3_output_path=bias_report_output_path,
                                                                       label='Label', # Assuming Label is the column name for the label attribute in your data
                                                                       headers=['Gender','Label','Other Features'], # Assuming these are the column names in your data
                                                                       dataset_type='text/csv'),
                                        data_bias_config=bias_config,
                                        methods='all', # This will run all the available bias metrics
                                        wait=True,
                                        logs=True)
```

2. To reduce algorithmic bias in your model, you can apply various bias mitigation techniques at different stages of your machine learning pipeline, such as data collection, data preprocessing, model training, and model evaluation.
Some of these techniques are:

- Data collection: You can try to collect more representative and diverse data that covers a wider range of gender expressions and identities. You can also use data augmentation techniques to generate synthetic data that balances the distribution of sensitive attributes.
- Data preprocessing: You can apply feature engineering techniques to remove or transform features that are highly correlated with the sensitive attributes. You can also use techniques like reweighting or resampling to adjust the weights or frequencies of different subgroups in your data.
- Model training: You can use regularization techniques to penalize your model for learning biased patterns from the data. You can also use adversarial learning techniques to train your model to be invariant to the sensitive attributes.
- Model evaluation: You can use post-processing techniques to adjust the predictions or thresholds of your model based on the fairness metrics. You can also use interactive tools like What-If Tool or Manifold to visualize and explore the effects of different bias mitigation techniques on your model performance and fairness.

You can also use tools like AI Fairness 360 or Fairlearn to implement some of these bias mitigation techniques in your code. For example, using AI Fairness 360, you can use the `aif360.algorithms.preprocessing.Reweighing` class to reweight your data based on demographic parity. Here is some sample code:

```python
# Import the required libraries
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Load your data into a BinaryLabelDataset object
df = pd.read_csv('your-data.csv') # Assuming this is your data file
dataset = BinaryLabelDataset(df=df,
                             label_names=['Label'], # Assuming Label is the column name for the label attribute in your data
                             protected_attribute_names=['Gender'], # Assuming Gender is the column name for the sensitive attribute in your data
                             favorable_label=1, # Assuming 1 is male and 0 is female in your data
                             unfavorable_label=0)

# Compute the original demographic disparity
metric = BinaryLabelDatasetMetric(dataset,
                                  unprivileged_groups=[{'Gender': 0}], # Assuming 0 is female in your data
                                  privileged_groups=[{'Gender': 1}]) # Assuming 1 is male in your data
print('Original demographic disparity:', metric.disparate_impact())

# Apply reweighting based on demographic parity
rw = Reweighing(unprivileged_groups=[{'Gender': 0}],
                privileged_groups=[{'Gender': 1}])
rw.fit(dataset)
dataset_transf = rw.transform(dataset)

# Compute the transformed demographic disparity
metric_transf = BinaryLabelDatasetMetric(dataset_transf,
                                         unprivileged_groups=[{'Gender': 0}],
                                         privileged_groups=[{'Gender': 1}])
print('Transformed demographic disparity:', metric_transf.disparate_impact())
```

I hope this helps you with your project. If you have any more questions, please feel free to ask. 😊

Source: Conversation with Bing, 7/30/2023
(1) arXiv:2103.05841v1 [cs.CL] 10 Mar 2021. https://arxiv.org/pdf/2103.05841.
(2) Gender Classification and Bias Mitigation in Facial Images - arXiv.org. https://arxiv.org/pdf/2007.06141v1.pdf.
(3) Gender Classification and Bias Mitigation in Facial Images. https://arxiv.org/abs/2007.06141.
(4) Gender Classification and Bias Mitigation in Facial Images. https://paperswithcode.com/paper/gender-classification-and-bias-mitigation-in.
(5) Bias in the machine: How can we address gender bias in AI?. https://www.raspberrypi.org/blog/gender-bias-in-ai-machine-learning-biased-data/.
(6) When Good Algorithms Go Sexist: Why and How to Advance AI Gender Equity. https://ssir.org/articles/entry/when_good_algorithms_go_sexist_why_and_how_to_advance_ai_gender_equity.
(7) Removing gender bias from algorithms - The Conversation. https://theconversation.com/removing-gender-bias-from-algorithms-64721.
(8) Bias Detection in Machine Learning Models using Amazon SageMaker .... https://towardsdatascience.com/bias-detection-in-machine-learning-models-using-amazon-sagemaker-clarify-d96482692611.
(9) 5 Tools to Detect and Eliminate Bias in Your Machine Learning Models. https://towardsdatascience.com/5-tools-to-detect-and-eliminate-bias-in-your-machine-learning-models-fb6c7b28b4f1.

-------------------------------------------------------------------------------------------------------

Hello, this is Bing. I can help you with your questions about algorithmic bias in face recognition.

1. To detect algorithmic bias in your code, you need to measure the performance of your gender classifier on different demographic groups, such as race, age, or skin tone. You can use various fairness metrics to quantify the disparity in accuracy or error rates across these groups. For example, you can use the mean difference metric, which is the difference in the mean outcomes between the privileged and unprivileged groups. A positive value indicates favoritism toward the privileged group, while a negative value indicates favoritism toward the unprivileged group. A value close to zero indicates fairness.

2. To reduce algorithmic bias in your code, you need to apply some bias mitigation techniques to your data or model. There are three main types of bias mitigation techniques: pre-processing, in-processing, and post-processing. Pre-processing techniques modify the data before training the model to remove or reduce bias. In-processing techniques modify the learning algorithm or objective function to enforce fairness constraints during training. Post-processing techniques modify the predictions of the model after training to achieve fairness goals.

3. Yes, AIF360 is capable of doing both detection and mitigation of algorithmic bias in face recognition. AIF360 is an open-source toolkit that provides a comprehensive set of algorithms and metrics for fairness and bias correction in machine learning. You can use AIF360 with Python or R to integrate fairness assessment and improvement into your workflow.

Here is an example of how you can use AIF360 with Python to detect and mitigate algorithmic bias in your gender classifier:

```python
# Import AIF360 and other libraries
import aif360
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import OptimPreproc
from sklearn.metrics import accuracy_score

# Load your data as a BinaryLabelDataset object
# Assume your data has a column called 'gender' as the label and a column called 'race' as the protected attribute
# You can also specify other columns as features or metadata
data = BinaryLabelDataset(df = your_data, label_names = ['gender'], protected_attribute_names = ['race'], 
                          favorable_label = 1, unfavorable_label = 0)

# Split your data into train and test sets
train_data, test_data = data.split([0.7], shuffle = True)

# Define the privileged and unprivileged groups based on the protected attribute
# For example, if you want to compare white and black groups
privileged_groups = [{'race': 1}] # white
unprivileged_groups = [{'race': 0}] # black

# Compute the mean difference metric on the train data
metric_train = BinaryLabelDatasetMetric(train_data, privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups)
mean_difference_train = metric_train.mean_difference()
print("Mean difference on train data: ", mean_difference_train)

# Compute the mean difference metric on the test data
metric_test = BinaryLabelDatasetMetric(test_data, privileged_groups = privileged_groups,
                                       unprivileged_groups = unprivileged_groups)
mean_difference_test = metric_test.mean_difference()
print("Mean difference on test data: ", mean_difference_test)

# Apply a pre-processing technique to mitigate bias in the train data
# For example, use Optimized Preprocessing (OP) algorithm
# OP learns a probabilistic transformation that aligns the distributions of the outcomes in the privileged and unprivileged groups
# You can specify the optimizer and the number of iterations
OP = OptimPreproc(optimizer = 'scipy', max_iter = 250)
OP.fit(train_data, privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups)
train_data_transformed = OP.transform(train_data)

# Compute the mean difference metric on the transformed train data
metric_train_transformed = BinaryLabelDatasetMetric(train_data_transformed, privileged_groups = privileged_groups,
                                                    unprivileged_groups = unprivileged_groups)
mean_difference_train_transformed = metric_train_transformed.mean_difference()
print("Mean difference on transformed train data: ", mean_difference_train_transformed)

# Train your gender classifier on the transformed train data
# For example, use a logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X_train = train_data_transformed.features # feature matrix
y_train = train_data_transformed.labels # label vector
model.fit(X_train, y_train)

# Make predictions on the test data using your gender classifier
X_test = test_data.features # feature matrix
y_test = test_data.labels # label vector
y_pred = model.predict(X_test) # predicted labels

# Evaluate the accuracy of your gender classifier on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of gender classifier on test data: ", accuracy)
```

I hope this helps! 😊

Source: Conversation with Bing, 7/30/2023
(1) Federal study of top facial recognition algorithms finds ‘empirical .... https://www.theverge.com/2019/12/20/21031255/facial-recognition-algorithm-bias-gender-race-age-federal-nest-investigation-analysis-amazon.
(2) Bias in facial recognition (article) | Khan Academy. https://www.khanacademy.org/computing/ap-computer-science-principles/data-analysis-101/x2d2f703b37b450a3:machine-learning-and-bias/a/bias-in-facial-recognition.
(3) The Problem of Bias in Facial Recognition - CSIS. https://www.csis.org/blogs/strategic-technologies-blog/problem-bias-facial-recognition.
(4) Algorithmic Bias in Facial Recognition - Keyo. https://www.keyo.com/biometric-news/algorithmic-bias-in-facial-recognition.
(5) Facing Bias in Facial Recognition Technology - The Regulatory Review. https://www.theregreview.org/2021/03/20/saturday-seminar-facing-bias-in-facial-recognition-technology/.
(6) Guide to AI Fairness 360: An Open Source Toolkit for Detection And .... https://analyticsindiamag.com/guide-to-ai-fairness-360-an-open-source-toolkit-for-detection-and-mitigation-of-bias-in-ml-models/.
(7) AI Fairness 360 Examples (Tutorials and Demos) - GitHub. https://github.com/Trusted-AI/AIF360/blob/master/examples/README.md.
(8) Installation Guide for AI Fairness 360 Toolkit - O'Reilly. https://conferences.oreilly.com/oscon/oscon-or-2019/cdn.oreillystatic.com/en/assets/1/event/295/oscon19_InstallationGuideAIFairness360Toolkit.pdf.
(9) AI Fairness 360 – Open Source - IBM. https://www.ibm.com/opensource/open/projects/ai-fairness-360/.
(10) undefined. https://avatars.githubusercontent.com/u/56103733?v=4.
(11) undefined. https://github.com/Trusted-AI/AIF360/blob/master/examples/README.md?raw=true.
(12) undefined. https://desktop.github.com.
(13) undefined. https://docs.github.com/articles/about-issue-and-pull-request-templates.
(14) undefined. http://aif360.mybluemix.net/.
(15) undefined. https://github.com/IBM/AIF360/wiki/OSCON-PORTLAND-2019.
(16) undefined. http://aif360.mybluemix.net.
(17) undefined. https://github.com/ibm/aif360.
(18) undefined. https://aif360.slack.com/.

--------------------------------------------------------------------------------------------------------
This page is my project for gender classification using face recognition trained on a dataset containing two subsets named male and female containing images only and no other labels. 1. Write code using AIF360 to detect (and then show) algorithmic bias in this model and then to mitigate it.
    
Here is a possible code using AIF360 to detect and mitigate algorithmic bias in your gender classifier:

```python
# Import AIF360 and other libraries
import aif360
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import accuracy_score

# Load your data as a BinaryLabelDataset object
# Assume your data has a column called 'gender' as the label and no other protected attributes
# You can also specify other columns as features or metadata
data = BinaryLabelDataset(df = your_data, label_names = ['gender'], 
                          favorable_label = 1, unfavorable_label = 0)

# Split your data into train and test sets
train_data, test_data = data.split([0.7], shuffle = True)

# Define the privileged and unprivileged groups based on the label
# For example, if you want to compare male and female groups
privileged_groups = [{'gender': 1}] # male
unprivileged_groups = [{'gender': 0}] # female

# Compute the mean difference metric on the train data
metric_train = BinaryLabelDatasetMetric(train_data, privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups)
mean_difference_train = metric_train.mean_difference()
print("Mean difference on train data: ", mean_difference_train)

# Compute the mean difference metric on the test data
metric_test = BinaryLabelDatasetMetric(test_data, privileged_groups = privileged_groups,
                                       unprivileged_groups = unprivileged_groups)
mean_difference_test = metric_test.mean_difference()
print("Mean difference on test data: ", mean_difference_test)

# Apply a pre-processing technique to mitigate bias in the train data
# For example, use Reweighing algorithm
# Reweighing assigns weights to each sample in the training set such that the distributions of the outcomes in the privileged and unprivileged groups are similar
RW = Reweighing()
RW.fit(train_data, privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups)
train_data_transformed = RW.transform(train_data)

# Compute the mean difference metric on the transformed train data
metric_train_transformed = BinaryLabelDatasetMetric(train_data_transformed, privileged_groups = privileged_groups,
                                                    unprivileged_groups = unprivileged_groups)
mean_difference_train_transformed = metric_train_transformed.mean_difference()
print("Mean difference on transformed train data: ", mean_difference_train_transformed)

# Train your gender classifier on the transformed train data
# For example, use a logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X_train = train_data_transformed.features # feature matrix
y_train = train_data_transformed.labels # label vector
model.fit(X_train, y_train)

# Make predictions on the test data using your gender classifier
X_test = test_data.features # feature matrix
y_test = test_data.labels # label vector
y_pred = model.predict(X_test) # predicted labels

# Evaluate the accuracy of your gender classifier on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of gender classifier on test data: ", accuracy)
```

I hope this helps! 😊

Source: Conversation with Bing, 7/30/2023
(1) Guide to AI Fairness 360: An Open Source Toolkit for Detection And .... https://analyticsindiamag.com/guide-to-ai-fairness-360-an-open-source-toolkit-for-detection-and-mitigation-of-bias-in-ml-models/.
(2) AI Fairness 360 Examples (Tutorials and Demos) - GitHub. https://github.com/Trusted-AI/AIF360/blob/master/examples/README.md.
(3) Installation Guide for AI Fairness 360 Toolkit - O'Reilly. https://conferences.oreilly.com/oscon/oscon-or-2019/cdn.oreillystatic.com/en/assets/1/event/295/oscon19_InstallationGuideAIFairness360Toolkit.pdf.
(4) AI Fairness 360 – Open Source - IBM. https://www.ibm.com/opensource/open/projects/ai-fairness-360/.
(5) Can Facial Recognition Overcome Its Racial Bias? - YES! Magazine. https://www.yesmagazine.org/social-justice/2020/04/16/privacy-facial-recognition.
(6) Racial Discrimination in Face Recognition Technology. https://sitn.hms.harvard.edu/flash/2020/racial-discrimination-in-face-recognition-technology/.
(7) Are women always better able to recognize faces? The unveiling ... - PLOS. https://journals.plos.org/plosone/article%3Fid%3D10.1371/journal.pone.0257741.
(8) gender bias in face recognition - arXiv.org. https://arxiv.org/pdf/2006.07845v1.pdf.
(9) Eye-tracking the own-gender bias in face recognition: Other-gender .... http://eprints.bournemouth.ac.uk/28614/1/ManHills_2017.pdf.
(10) undefined. https://avatars.githubusercontent.com/u/56103733?v=4.
(11) undefined. https://github.com/Trusted-AI/AIF360/blob/master/examples/README.md?raw=true.
(12) undefined. https://desktop.github.com.
(13) undefined. https://docs.github.com/articles/about-issue-and-pull-request-templates.
(14) undefined. http://aif360.mybluemix.net/.
(15) undefined. https://github.com/IBM/AIF360/wiki/OSCON-PORTLAND-2019.
(16) undefined. http://aif360.mybluemix.net.
(17) undefined. https://github.com/ibm/aif360.
(18) undefined. https://aif360.slack.com/.
(19) undefined. https://doi.org/10.1371/journal.pone.0257741.

--------------------------------------------------------------------------------------------------------
