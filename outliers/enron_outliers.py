#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../my_final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

#get the maximum salary and bonus

result = max(data, key=lambda d: d[0]+d[1])
print result[0], result[1]

outlier = [k for k, v in data_dict.iteritems() if v['salary'] == int(result[0]) and v['bonus'] == int(result[1])]
print outlier

for k, v in data_dict.iteritems():
    if v['salary'] == int(result[0]) and v['bonus'] == int(result[1]):
        print k

data_dict.pop(outlier[0], 0)

#check if there are any more outliers left
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()