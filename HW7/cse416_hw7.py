import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

np.random.seed(416)

# Setup
text = pd.read_csv('tweets-2020-4-30.csv')
text = text.fillna('') # some rows are nan so replace with empty string
vectorizer = TfidfVectorizer(max_df=0.95)
tf_idf = vectorizer.fit_transform(text['text'])
feature_names = vectorizer.get_feature_names()

# Q1
# TODO compute num_tweets and num_words
num_tweets = tf_idf.shape[0]
num_words = tf_idf.shape[1]

# Q2
# TODO create and fit the model and transform our data
nmf = NMF(init='nndsvd', n_components=5)
tweets_projected = nmf.fit_transform(tf_idf)

# Q3
# TODO
q3 = 'word'

# Q4
small_words = ['dogs', 'cats', 'axolotl']
small_weights = np.array([1, 4, 2])

# TODO Write code to make sorted_small_words as described above
ind = np.argsort(small_weights)
sorted_small_words_asc = np.take_along_axis(np.array(small_words), ind, axis=0)
sorted_small_words = sorted_small_words_asc[::-1].tolist()

# Q5
def words_from_topic(topic, feature_names):
    """
    Sorts the words by their weight in the given topic from largest to smallest.
    topic and feature_names should have the same number of entries.

    Args:
     - topic (np.array): A numpy array with one entry per word that shows the weight in this topic.
    - feature_names (list): A list of words that each entry in topic corresponds to

    Returns:
    - A list of words in feature_names sorted by weight in topic from largest to smallest. 
    """
    # TODO
    ind = np.argsort(topic)
    sorted_feature_names = np.take_along_axis(np.array(feature_names), ind, axis = 0)
    return sorted_feature_names[::-1].tolist()

# Q6
# TODO look at the output above to identify which topic the tweet above is most associated to
q6 = 2

# Q7
# TODO find index of largest topic
y = np.zeros(len(tweets_projected))
for i in range(len(tweets_projected)):
    y[i] = list(tweets_projected[i]).index(max(tweets_projected[i]))
count = np.unique(y, return_counts = True)
for i in range(np.shape(count)[1]):
    if count[1][i] == max(count[1]):
        largest_topic = int(count[0][i])

# Setup - Q8
nmf_small = NMF(n_components=3, init='nndsvd')
tweets_projected_small = nmf_small.fit_transform(tf_idf)

# Q8
z = y = np.zeros(len(tweets_projected_small[:, 2]))
for i in range(len(tweets_projected_small[:, 2])):
    if tweets_projected_small[:, 2][i] >= 0.15:
        z[i] = i
    else:
        z[i] = -1
rows = np.delete(np.unique(z, return_counts=True)[0],0)
outlier_tweets = np.empty(len(rows), dtype=object)
for i in range(len(rows)):
    outlier_tweets[i] = text.iloc[int(rows[i])]['text']
outlier_tweets = np.unique(outlier_tweets, return_counts=True)[0]
