import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
sales = pd.read_csv('home_data.csv')

# Q1
# TODO
num_rows = sales.shape[0]

# TODO
y = sales.loc[:,"price"]

# TODO
num_inputs = sales.drop(columns = ["price", "date", "id"]).shape[1]

# Q2
# TODO
avg_price_3_bed = sales[(sales["bedrooms"] == 3)].price.mean()

# Q3
# TODO
percent_q3 = sales[(sales["sqft_living"] >= 2000) & (sales["sqft_living"] < 4000)].shape[0] / sales.shape[0]


# Q4
# TODO
from sklearn.model_selection import train_test_split
import numpy as np

# Set seed to create pseudo-randomness
np.random.seed(416)

# Split data into 80% train and 20% test
train_data, val_data = train_test_split(sales, test_size=0.2)

# Set seed to create pseudo-randomness
np.random.seed(416)

# Split data into 80% train and 20% validation
train_data, val_data = train_test_split(sales, test_size=0.2)

basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

advanced_features = basic_features + [
    'condition',      # condition of the house
    'grade',          # measure of qality of construction
    'waterfront',     # waterfront property
    'view',           # type of view
    'sqft_above',     # square feet above ground
    'sqft_basement',  # square feet in basementab
    'yr_built',       # the year built
    'yr_renovated',   # the year renovated
    'lat',            # the longitude of the parcel
    'long',           # the latitide of the parcel
    'sqft_living15',  # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',     # average lot size of 15 nearest neighbors
]

from sklearn.linear_model import LinearRegression

# TODO
basic_model = LinearRegression().fit(train_data[basic_features], train_data["price"])

advanced_model = LinearRegression().fit(train_data[advanced_features], train_data["price"])


# Q5
# TODO
from sklearn.metrics import mean_squared_error
# TODO

#print(list(basic_model.predict(val_data[basic_features])))
#print(list(val_data.loc[:,"price"]))

train_rmse_basic = np.sqrt(mean_squared_error(list(train_data["price"]), list(basic_model.predict(train_data[basic_features]))))
train_rmse_advanced = np.sqrt(mean_squared_error(list(train_data["price"]), list(advanced_model.predict(train_data[advanced_features]))))


# Q6
# TODO

# TODO
val_rmse_basic = np.sqrt(mean_squared_error(list(val_data["price"]), list(basic_model.predict(val_data[basic_features]))))
val_rmse_advanced = np.sqrt(mean_squared_error(list(val_data["price"]), list(advanced_model.predict(val_data[advanced_features]))))