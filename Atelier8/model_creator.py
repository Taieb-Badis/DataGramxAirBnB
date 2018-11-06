import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def create_model(prices_path, listings_path):
	prices = pd.read_csv(prices_path, sep=";")
	listings = pd.read_csv(listings_path, sep=";")
	listings = listings.drop(589)  # Cet individu n'a pas de prix !
	print("Chargement des données effectué.")

	uniques = np.unique(prices["listing_id"]) 
	intersects = np.intersect1d(uniques, np.unique(listings["listing_id"]))
	listings_prices = prices[prices["listing_id"].isin(intersects)]
	dates =  pd.to_datetime(listings_prices["day"], format='%Y-%m-%d')
	week_prices = listings_prices.assign(date=dates).set_index(dates).groupby(['listing_id', pd.Grouper(key='date', freq='7D')]).mean().reset_index().loc[:, ["listing_id", "date", "local_price"]]

	def populate(row):
	    return row.append(listings[listings["listing_id"] == row["listing_id"]].iloc[0,2:])
	    
	X = week_prices.apply(populate, axis=1)
	label_encoder = LabelEncoder()
	X["label_dates"] = label_encoder.fit_transform(X["date"])
	X_train, X_test, Y_train, Y_test = train_test_split(
	    X.loc[:, ["label_dates", "latitude", "longitude", "person_capacity", "bedrooms", "bathrooms"]],
	    X.loc[:, "local_price"],
	    test_size=0.2)

	param = {'max_depth':3, 'eta':1.0, 'silent':1, 'objective':'reg:linear', 'n_estimators': 300 }
	booster = xgb.XGBRegressor(**param)
	booster.fit(X_train, Y_train)
	return booster, X_train, X_test, Y_train, Y_test


