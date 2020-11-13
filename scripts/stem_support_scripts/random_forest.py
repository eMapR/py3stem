#some of the code borrowed from: https://github.com/geo-yrao/esip-ml-tutorials/blob/master/classification/ESIP_Machine_Learning_Tutorials_Classification-Python.ipynb
#original author was Yuhan (Douglas) Rao 
import json
import sys
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

def clean_train_data(input_data,year_of_interest): 
	df = pd.read_csv(input_data)
	#get rid of the annoying system:index col from GEE
	if df.columns[0] == 'system:index': 
		df=df.drop(columns=['system:index','.geo'],axis=1)
	df=df.dropna()
	print(df.head())
	df = df[[x for x in df.columns if (year_of_interest in x) or ('class' in x)]]
	# unique,counts = np.unique(df['class'],return_counts=True)
	# plt.bar(['not_glacier','glacier'],counts)
	# plt.show()
	# plt.close('all')
	X = df.values[:,1:]  # Data. The six L8 channels in the dataframe.
	y = df.values[:,0]
	# Create a binary classification data set of the data and the target for corn vs. urban pixels.
	indices_glacier = np.where(df['class']==1)
	indices_not_glacier = np.where(df['class']==0)
	x_bin = np.concatenate((X[indices_glacier], X[indices_not_glacier]))
	y_bin = np.concatenate((y[indices_glacier], y[indices_not_glacier]))
	return x_bin,y_bin
	

class RandomForest(): 
	def __init__(self,x_input_data,y_input_data,output_dir): 
		self.x_input_data=x_input_data
		self.y_input_data=y_input_data
		self.output_dir=output_dir


	def make_train_test(self): 
		# Binarize the forest and water sample labels.  We do this step because sklearn's
		# classifiers will assume that we're dealing with a multiclass classification
		# problem if they see any labels that aren't 0 and 1.
		lb = LabelBinarizer()
		y_binary = lb.fit_transform(self.y_input_data.tolist())

		## We are spliting our data set into training and testing sets based on a 80:20 ratio
		X_train, X_test, y_train, y_test = train_test_split(self.x_input_data, y_binary, test_size=0.2, random_state=0)
		return X_train, X_test, y_train, y_test

	def run_random_forest(self): 

		## For consistency with our R notebook, we set the *n_estimator=500* as the 
		## default number of trees.

		## To build our first random forest model, we fix 
		##     max_features = 4; min_sample_leaf = 1
		## to be consistent with R notebook
		train_test = self.make_train_test()
		X_train=train_test[0]
		X_test=train_test[1]
		y_train=train_test[2]
		y_test=train_test[3]
		# Now we create a random forest classifier.
		rf_Classifier = RandomForestClassifier(n_estimators=500, max_depth=None,
		                                      max_features=4, min_samples_leaf=1,
		                                      min_samples_split=2, random_state=0)

		# Using the training set, we can fit our model now
		rf_Classifier.fit(X_train, y_train.ravel())

		# Have the newly trained classifier predict the classes of the withheld testing data.
		rf_predicted = rf_Classifier.predict(X_test)
		print(rf_predicted)

		## Calculate the confusion matrix and normalized confusion matrix
		rfMatrix = confusion_matrix(y_test.ravel(), rf_predicted.ravel())
		rfMatrix_normalized = rfMatrix/rfMatrix.sum()

		# Initialize figure, axes for the two confusion matrices.
		fig, ax = plt.subplots(1, 2, figsize=(14, 5))

		# Plot the raw counts' confusion matrix.
		seaborn.heatmap(
		    rfMatrix, cmap="Greens", annot=rfMatrix, square=True, cbar=True,
		    xticklabels=["Glacier","Non-glacier"], yticklabels=["Glacier","Non-glacier"],
		    ax=ax[0]
		)

		# Add labels to the x-axis and the y-axis.
		ax[0].set_xlabel("Predicted", fontsize=16)
		ax[0].set_ylabel("Reference", fontsize=16)

		# Plot the percentages' confusion matrix.
		seaborn.heatmap(
		    rfMatrix_normalized, cmap="Greens", annot=True, square=True,
		    xticklabels=["Glacier","Non-glacier"], yticklabels=["Glacier","Non-glacier"],
		    ax=ax[1]
		)

		# Add a label to the x-axis.
		ax[1].set_xlabel("Predicted", fontsize=16)

		# Add a title to the figure.
		fig.suptitle("Confusion matrices (Random Forest): raw counts and normalized frequencies", fontsize=16)

		# Display the figure.
		fig.show()

		print(
    	classification_report(y_test.ravel(), rf_predicted.ravel(),
                          target_names=["Glacier", "Non-glacier"])
		)
	def run_k_fold_cross_fold_validation(self): 
		## Import the data spliting function from sklearn.model_selection

		## Initialize a k-fold cross-validation generator to split our data sets for us.
		kfold_generator = KFold(n_splits=3, random_state=7)

		# Show histograms of class distributions across the splits
		labels_list = ["Glacier", "Non-glacier"]
		colors_list = ["darkgreen", "royalblue"]

		# Here, instead of using Pyplot's high-level commands, we're going to produce a 
		# matplotlib figure with multiple subplots. These subplots are represented by 
		# "axes" objects.
		# Creating a panel of plots with three rows and two columns.
		fig, ax = plt.subplots(3, 2, figsize=(15, 14), sharex=True, sharey=True)

		# Iterate through the splits and plot the classes' distributions.
		for axes_row_idx, (splits_train, splits_test) in enumerate(kfold_generator.split(self.x_input_data)):
		    
		    # Get this split's training and testing labels.
		    y_trn = self.y_input_data[splits_train]
		    y_tst = self.y_input_data[splits_test]

		    # Get the counts for each of the unique class labels so we can plot their distribution.
		    _, counts_train = np.unique(y_trn, return_counts=True)
		    _, counts_test  = np.unique(y_tst, return_counts=True)

		    # Plot the training splits' distributions in the left-hand column and the 
		    # testing splits' distributions in the right-hand column. The ";" at the end
		    # of the following two lines is to suppress some returned output.
		    ax[axes_row_idx][0].bar(labels_list, counts_train, color=colors_list);
		    ax[axes_row_idx][1].bar(labels_list, counts_test , color=colors_list);

		    # Add a label to the y axis of the left column's subplot.
		    ax[axes_row_idx][0].set_ylabel("Frequency")

		# Add titles over the left and right columns.
		ax[0][0].set_title("Training split histogram")    
		ax[0][1].set_title("Testing split histogram")

		# Nicely ask the computer to show us the plots.
		fig.show()
		plt.show()
		plt.close('all')

	def run_stratified_cross_fold(self): 
		# Initialize a 3-fold stratified splits generator.
		stratified_kfold_generator = StratifiedKFold(n_splits=3, random_state=7)
		# Show histograms of class distributions across the splits
		labels_list = ["Glacier", "Non-glacier"]
		colors_list = ["darkgreen", "royalblue"]
		train_test = self.make_train_test()
		X_train=train_test[0]
		X_test=train_test[1]
		y_train=train_test[2]
		y_test=train_test[3]
		# Iterate through the splits and plot them.
		fig, ax = plt.subplots(3, 2, figsize=(15, 14), sharex=True, sharey=True)
		for row_idx, (splits_train, splits_test) in\
			enumerate(stratified_kfold_generator.split(self.x_input_data, self.y_input_data)): #previously x_bin,y_bin

			# Get the training and testing labels.
			y_trn = self.y_input_data[splits_train]
			y_tst = self.y_input_data[splits_test] #previously ybin

			# Get the counts for each of the unique class labels so we can plot their distribution.
			_, counts_train = np.unique(y_trn, return_counts=True)
			_, counts_test = np.unique(y_tst, return_counts=True)

			# Plot this split's class distributions.
			ax[row_idx][0].bar(labels_list, counts_train, color=colors_list);
			ax[row_idx][1].bar(labels_list, counts_test, color=colors_list);

			# Add a label to the shared y-axis.  
			ax[row_idx][0].set_ylabel("Frequency")

			# Add separate titles to the two subplots.
			ax[0][0].set_title("(Stratified) Training split distribution")    
			ax[0][1].set_title("(Stratified) Testing split distribution")

		#fig.show()
		#plt.show()
		#plt.close('all')

	#def choose_model_hyper_params(self): 


		# Set up the parameter grid. In this case, it's just a list of all the
		# candidate number of neighbors we'll consider.
		tuned_parameters = {'n_estimators': [1000], 
		                    'max_depth':    [None],
		                    'min_samples_split': [2],
		                    'random_state': [0],
		                    'max_features': [2, 3, 4, 5, 6,7,8,9,10], 
		                    'min_samples_leaf': [1, 3, 5, 7,9,11,13], 
		                    'n_jobs':[10]}

		# Initialize a ra classifier object. 
		rf_classifier = RandomForestClassifier()

		# Create the grid search object. This object will take the kNN classifier
		# and run stratified 10-fold cross-validation for each of the potential
		# candidates for k. It will record the averaged accuracy for each k so
		# that afterwards we can view how the classifier's accuracy improves or
		# worsens with respect to k.
		gridsearch_cv_obj = GridSearchCV(
		    rf_classifier, 
		    tuned_parameters, 
		    scoring=make_scorer(accuracy_score), 
		    cv=stratified_kfold_generator,
		    n_jobs=-1,
		  )

		# Run the grid search for the optimal number of neighbors, k.
		gridsearch_cv_obj.fit(X_train, y_train.ravel());

		## Print best parameter combination
		print("Best parameters set found on development set:")
		print()
		print(gridsearch_cv_obj.best_params_)
		print()

		## Calculating the mean and standard deviation of the accuracy for the model test scores
		mean_test_accuracy = gridsearch_cv_obj.cv_results_["mean_test_score"]
		stds_test_accuracy  = gridsearch_cv_obj.cv_results_["std_test_score"]

		max_features = gridsearch_cv_obj.cv_results_["param_max_features"]
		min_samples_leaf = gridsearch_cv_obj.cv_results_["param_min_samples_leaf"]

		## rearrange the accuracy data to a 2d array so we can have a heatmap
		mean_accuracy_2d = mean_test_accuracy.reshape(9, 7)
		stds_accuracy_2d = stds_test_accuracy.reshape(9, 7)
		# 5 rows for max_features & 4 columns for min_samples_leaf

		## Creating the heatmap for both mean accuracy and standard deviation
		# Initialize figure, axes for the accuracy heatmap.
		fig, ax = plt.subplots(1, 2, figsize=(14, 5))

		# Plot the raw counts' confusion matrix.
		seaborn.heatmap(
		    mean_accuracy_2d, cmap="Blues", annot=True, square=False, cbar=True,
		    xticklabels=[1, 3, 5, 7,9,11,13], yticklabels=[2, 3, 4, 5, 6,7,8,9,10],
		    ax=ax[0]
		)

		# Add labels to the x-axis and the y-axis.
		ax[0].set_xlabel("Minimum # of samples", fontsize=16)
		ax[0].set_ylabel("Maximum # of features", fontsize=16)

		# Plot the percentages' confusion matrix.
		seaborn.heatmap(
		    stds_accuracy_2d, cmap="Reds", annot=True, square=False,
		    xticklabels=[1, 3, 5, 7,9,11,13], yticklabels=[2, 3, 4, 5, 6,7,8,9,10],
		    ax=ax[1]
		)

		# Add labels to the x-axis and the y-axis.
		ax[1].set_xlabel("Minimum # of samples", fontsize=16)

		# Add a title to the figure.
		fig.suptitle("Random forest testing accuracy for 10-fold startified Cross Validation", fontsize=16)

		# Display the figure.
		fig.show()
		plt.show()
		plt.close('all')

def main(): 
	params = sys.argv[1]
	with open(str(params)) as f:
		variables = json.load(f)
		train_set = variables["train_set"]
		output_dir = variables["output_dir"]
		year_of_interest = variables['year_of_interest']
		rf_inputs=clean_train_data(train_set,year_of_interest)
		#run_rf=RandomForest(rf_inputs[0],rf_inputs[1],output_dir).run_random_forest()
		cv_run=RandomForest(rf_inputs[0],rf_inputs[1],output_dir).run_stratified_cross_fold()
if __name__ == '__main__':
	main()