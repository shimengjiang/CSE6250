import utils
import etl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	train_path = '../data/train/'
	events, mortality, feature_map = etl.read_csv(train_path)
	patient_features, mortality = etl.create_features(events, mortality, feature_map)
	etl.save_svmlight(patient_features, mortality, '../deliverables/myfeatures_svmlight.train',
				  '../deliverables/myfeatures.train')
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/myfeatures_svmlight.train")

	patient_featuresT, mortalityT = createTestFeature()
	etl.save_svmlight(patient_featuresT, mortalityT, "../deliverables/testfeatures_svmlight.train",
				  "../deliverables/test_features.txt")
	X_test, _ = utils.get_data_from_svmlight("../deliverables/testfeatures_svmlight.train")
	return X_train, Y_train, X_test

def createTestFeature():
	events = pd.read_csv('../data/test/events.csv')
	featureMap = pd.read_csv('../data/test/event_feature_map.csv')
	aggregated_events = etl.aggregate_events(events, 1, featureMap, "../data/test/")

	patient_features = {}
	mortality1 = {}

	for row in aggregated_events.iterrows():
		patientId = int(round(row[1]["patient_id"]))
		featureId = int(round(row[1]["feature_id"]))
		featureValue = row[1]["feature_value"]
		if patientId in patient_features.keys():
			patient_features[patientId].append((featureId, featureValue))
		else:
			patient_features[patientId] = []
			patient_features[patientId].append((featureId, featureValue))

	for key in patient_features.keys():
		mortality1[key] = 0

	return patient_features, mortality1
'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	parameters = {"n_estimators" : [10, 50, 100, 200, 400], "max_depth" : [2, 4, 8, None]}
	rfc = RandomForestClassifier()
	clf = GridSearchCV(rfc, parameters, scoring='roc_auc', cv=5)
	clf.fit(X_train, Y_train)
	#Y_pred = clf.predict_proba(X_test)[:, 1]
	Y_pred = clf.predict(X_test)
	return Y_pred


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	