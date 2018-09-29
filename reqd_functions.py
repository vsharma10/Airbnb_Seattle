# Udacity Project: Term 2 (Blog post)
# Analysis of Airbnb, Seattle dataset
################################
# Created by Vivekanand Sharma #
# Dated Sept 27, 2018          #
################################

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns


from collections import Counter
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from wordcloud import WordCloud



def clean_data(df):
	'''
	INPUT
	df - Raw dataframe

	OUTPUT
	Cleaned dataframe
	'''
	data_df = df.copy()

	# Clean columns with price amounts having '$' and ',' characters
	data_df['price'] = data_df[['price']].apply(lambda x : x.str.replace('$', '')).apply(lambda x : x.str.replace(',', '')).astype(float)
	data_df['security_deposit'] = data_df[['security_deposit']].apply(lambda x : x.str.replace('$', '')).apply(lambda x : x.str.replace(',', '')).astype(float)
	data_df['cleaning_fee'] = data_df[['cleaning_fee']].apply(lambda x : x.str.replace('$', '')).apply(lambda x : x.str.replace(',', '')).astype(float)
	data_df['extra_people'] = data_df[['extra_people']].apply(lambda x : x.str.replace('$', '')).apply(lambda x : x.str.replace(',', '')).astype(float)
	
	# Clean percentage columns with '%' character
	data_df['host_response_rate'] = data_df[['host_response_rate']].apply(lambda x : x.str.replace('%', '')).astype(float)
	data_df['host_acceptance_rate'] = data_df[['host_acceptance_rate']].apply(lambda x : x.str.replace('%', '')).astype(float)
	
	# Convert thumnail data as available or not-available
	data_df['thumbnail_available'] = np.where(data_df['thumbnail_url'].isnull(), 0, 1)
	data_df.drop('thumbnail_url', axis = 1, inplace = True)

	# Convert host since to number of days
	data_df['host_since'] = pd.to_datetime(data_df['host_since'], format='%Y-%m-%d')
	data_df['host_since'] = data_df[['host_since']].apply(lambda x : days_between(x,'2016-01-04'))
	
	# Convert amenities column to number of amenities
	amen_count = []
	for str in data_df.amenities.values:
		amenities = list(csv.reader([str[1:-2]]))[0]
		amen_count.append(len(amenities))
	
	data_df['amenities_count'] = amen_count
	data_df = data_df.drop(['amenities'], axis=1)
	
	return data_df


def days_between(d1, d2):
	'''
	INPUT
	d1, d2 - Start and End dates

	OUTPUT
	Difference between dates in number of days
	'''
	#d1 = datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.strptime(d2, "%Y-%m-%d")
	
	return abs((d2-d1).dt.days)



def process_features(feature_df):
	'''
	Perform feature encoding, imputes missing values, and scales features
	INPUT
	feature_df - Feature dataframe

	OUTPUT
	scaled_df - Perform feature encoding, imputes missing values, and 
	returns scaled features
	'''

	# Identify and encode categorical variables
	cat_feat = list(feature_df.select_dtypes(include = ['object']).columns)

	# Encode categorical variables
	feature_df = pd.get_dummies(feature_df, columns = cat_feat)

	# Impute missing values	
	fill_NaN = Imputer(missing_values=np.nan, strategy='most_frequent', axis=1)
	imputed_df = pd.DataFrame(fill_NaN.fit_transform(feature_df))
	imputed_df.columns = feature_df.columns
	imputed_df.index = feature_df.index

	# Feature scaling
	num_feat = list(imputed_df.select_dtypes(include = ['int64','float64']).columns)
	scaler = MinMaxScaler()
	scaled_df = imputed_df.copy()
	scaled_df[num_feat] = scaler.fit_transform(imputed_df[num_feat])
	
	return scaled_df



def create_text_features(text_df):
	'''
	INPUT
	text_df - DataFrame with text data

	OUTPUT
	scaled_df - scaled text features 
	'''


	vectorizer = CountVectorizer(min_df=5).fit(text_df)
	text_features = vectorizer.transform(text_df)
	text_features = pd.SparseDataFrame([ pd.SparseSeries(text_features[i].toarray().ravel()) \
							  for i in np.arange(text_features.shape[0]) ])

	
	text_features.columns = vectorizer.get_feature_names()
	text_features.index = text_features.index

	num_feat = list(text_features.select_dtypes(include = ['int64','float64']).columns)
	scaler = MinMaxScaler()
	scaled_df = text_features.copy()
	scaled_df[num_feat] = scaler.fit_transform(text_features[num_feat])

	return scaled_df



def boost_classifier(clf, parameters, feature_df, labels):
	'''
	Optimize the classifier
	
	INPUTS
	clf        - Model object from sklearn
	feature_df - DataFrame of features
	labels     - Response variable
	
	OUTPUTS
	X_train, X_test, y_train, y_test - output from sklearn train test split
	best_clf - Optimized model
	'''
	
	# Split the 'features' and 'label' data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(feature_df, labels, test_size = 0.3, random_state = 0)
																																							
	# Make an fbeta_score scoring object using make_scorer()
	scorer = make_scorer(fbeta_score,beta=0.5)

	# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
	grid_obj = GridSearchCV(clf, parameters, scorer, n_jobs=-1)

	# Fit the grid search object to the training data and find the optimal parameters using fit()
	grid_fit = grid_obj.fit(X_train,y_train)

	# Get the estimator
	best_clf = grid_fit.best_estimator_
	
	return best_clf, X_train, X_test, y_train, y_test


def prediction_scores(clf, X_train, X_test, y_train, y_test):
	'''
	INPUTS
	clf - Model object from sklearn
	X_train, X_test, y_train, y_test - output from sklearn train test split
	
	OUTPUTS
	test_accuracy  - Accuracy score on test data
	train_accuracy - Accuracy score on train data
	'''
	
	# Make predictions using the model
	test_preds = (clf.fit(X_train, y_train)).predict(X_test)
	train_preds = (clf.fit(X_train, y_train)).predict(X_train)
	
	# Calculate accuracy for the model
	test_accuracy = accuracy_score(y_test, test_preds)
	train_accuracy = accuracy_score(y_train, train_preds)
	
	return test_accuracy, train_accuracy
	
	
def print_scores(test_accuracy, train_accuracy):
	'''
	INPUTS
	test_accuracy  - Accuracy score on test data
	train_accuracy - Accuracy score on train data

	OUTPUTS
	Prints accuracy scores
	'''

	print("Accuracy score on testing data: {:.4f}".format(test_accuracy))
	print("Accuracy score on training data: {:.4f}".format(train_accuracy))


def feature_plot(importances, X_train, y_train):
	'''
	INPUTS
	importances - Feature importances
	X_train, y_train - Training data

	OUTPUT
	Plot of five most important features
	'''
	
	# Displays the five most important features
	indices = np.argsort(importances)[::-1]
	columns = X_train.columns.values[indices[:5]]
	values = importances[indices][:5]

	# Creat the plot
	fig = plt.figure(figsize = (15,5))
	plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
	plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
		  label = "Feature Weight")
	plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
		  label = "Cumulative Feature Weight")
	plt.xticks(np.arange(5), columns)
	plt.xlim((-0.5, 4.5))
	plt.ylabel("Weight", fontsize = 12)
	plt.xlabel("Feature", fontsize = 12)
	
	plt.legend(loc = 'upper center')
	plt.tight_layout()
	plt.show()


def create_wordcloud(importances, X_train, y_train):
	'''
	INPUTS
	importances - Feature importances
	X_train, y_train - Training data

	OUTPUT
	Word colud of top 50 word features
	'''
	indices = np.argsort(importances)[::-1]
	columns = X_train.columns.values[indices[:50]]
	values = importances[indices][:50]
	word_dict = dict(zip(columns, values))

	# Generate wordcloud using top 50 word features
	wordcloud = WordCloud().generate_from_frequencies(word_dict)
	plt.figure(figsize=(25, 15))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()