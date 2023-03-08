import pandas as pd
import numpy as np

def get_entropy(arr, eigen):
	if eigen == "star_rating":
		dic = {"good":0, "bad":0}
		for star in arr:
			if star >= 4:
				dic["good"] += 1
			else:
				dic["bad"] += 1
		s = sum(dic.values())
		H_X = 0
		for key, val in dic.items():
			H_X -= (val/s) * np.log(val/s)
		return H_X
	elif eigen == "helpful_votes":
		dic = {"helpful":0, "helpless":0}
		for vote in arr:
			if vote > 2:
				dic["helpful"] += 1
			else:
				dic["helpless"] += 1
		s = sum(dic.values())
		H_X = 0
		for key, val in dic.items():
			H_X -= (val/s) * np.log(val/s)
		return H_X
	elif eigen == "review_score":
		dic= {"good":0, "bad":0}
		for review in arr:
			if review > 0:
				dic["good"] += 1
			else:
				dic["bad"] += 1
		s = sum(dic.values())
		H_X = 0
		for key, val in dic.items():
			H_X -= (val/s) * np.log(val/s)
		return H_X
	else:
		print("error")
dataarr=pd.read_csv("entropy_data.csv")
print("star_rating:")
print(get_entropy(dataarr["star_rating"], "star_rating"))
print("helpful_votes")
print(get_entropy(dataarr["vote_values"], "helpful_votes"))
print("review_scores")
print(get_entropy(dataarr["review_scores"], "review_score"))
