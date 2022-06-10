import string
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Add any additional imports here
# TODO


np.random.seed(416)

# Load data
products = pd.read_csv('food_products.csv')
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

# Data Processing: Remove punctuation
def remove_punctuation(text):
    if type(text) is str:
        return text.translate(str.maketrans('', '', string.punctuation))
    else:
        return ''
    
products['review_clean'] = products['review'].apply(remove_punctuation)

# Feature Extraction: Count words
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(products['review_clean'])

product_data = pd.DataFrame(count_matrix.toarray(),
        index=products.index,
        columns=vectorizer.get_feature_names())

#  Create a new DataFrame that has all these features as columns plus the sentiment label!
product_data['sentiment'] = products['sentiment']
product_data['review_clean'] = products['review_clean']
product_data['summary'] = products['summary']

# Train/Validation/Test Split
train_data, test_data = train_test_split(product_data, test_size=0.2)
validation_data, test_data = train_test_split(test_data, test_size=0.5)

# Q1: Majority class classifier
# TODO
majority_label = train_data["sentiment"].mode()[0]
number_majority = sum(validation_data["sentiment"] == majority_label)
total = validation_data["sentiment"].count()

majority_classifier_validation_accuracy = number_majority/total


# Train a sentiment model
features = vectorizer.get_feature_names()
sentiment_model = LogisticRegression(penalty='l2', C=1e23)
sentiment_model.fit(train_data[features], train_data['sentiment'])

# Q2: Compute most positive/negative
# TODO
coefficients = sentiment_model.coef_[0] 
max_min_dict = {"max" : {"word" : features[coefficients.argmax()],
 "index" : coefficients.argmax(),
 "value" : coefficients.max()},
 "min" : {"word" : features[coefficients.argmin()],
 "index" : coefficients.argmin(),
 "value" : coefficients.min()}}

most_negative_word = max_min_dict["min"]["word"]
most_positive_word = max_min_dict["max"]["word"]

# Q3: Most positive/negative review
# TODO

sentiment_validation_prob = sentiment_model.predict_proba(validation_data[features])

most_positive_review = validation_data.iloc[sentiment_validation_prob[:,1].argmax()]["review_clean"]
most_negative_review = validation_data.iloc[sentiment_validation_prob[:,0].argmax()]["review_clean"]

# Q4: Sentiment model validation accuracy 
# TODO
from sklearn.metrics import accuracy_score
y_true = validation_data["sentiment"]
y_pred = sentiment_model.predict(validation_data[features])

sentiment_model_validation_accuracy = accuracy_score(y_true, y_pred)

# Q5: Confusion matrix
# TODO

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Q6 and Q7

# Set up the regularization penalities to try
l2_penalties = [0.01, 1, 4, 10, 1e2, 1e3, 1e5]
l2_penalty_names = [f'coefficients [L2={l2_penalty:.0e}]' 
                    for l2_penalty in l2_penalties]

# Q6: Add the coefficients to this coef_table for each model
coef_table = pd.DataFrame(columns=['word'] + l2_penalty_names)
coef_table['word'] = features

# Q7: Set up an empty list to store the accuracies (will convert to DataFrame after loop)
l2_penalty_list1 = []
train_accuracy = []
validation_accuracy = []

for l2_penalty, l2_penalty_column_name in zip(l2_penalties, l2_penalty_names):
    # (Q6 and Q7): Train the model 
    # TODO
    model = LogisticRegression(fit_intercept = False, C = 1/l2_penalty).fit(train_data[features], train_data["sentiment"])
    
    # (Q6): Save the coefficients in coef_table
    # TODO
    coef_table[l2_penalty_column_name] = model.coef_[0]

    # (Q7): Calculate and save the train and validation accuracies
    # TODO
    l2_penalty_list1.append(l2_penalty)
    y_train_true = train_data["sentiment"]
    y_train_pred = model.predict(train_data[features])
    train_accuracy.append(accuracy_score(y_train_true, y_train_pred, normalize=True))
    y_val_true = validation_data["sentiment"]
    y_val_pred = model.predict(validation_data[features])
    validation_accuracy.append(accuracy_score(y_val_true, y_val_pred, normalize=True))

accuracy_data = {'l2_penalty': l2_penalty_list1, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy} 


accuracies_table = pd.DataFrame(accuracy_data)

# Q8 
# TODO


positive_words = coef_table.nlargest(5, "coefficients [L2=1e+00]")["word"]
negative_words = coef_table.nsmallest(5, "coefficients [L2=1e+00]")["word"]
