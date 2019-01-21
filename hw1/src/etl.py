import utils
import pandas as pd
import numpy as np
import time
from dateutil import parser
import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    '''
    eventPerPatient = events.groupby("patient_id").count()
    mortPerPatient = mortality.groupby("patient_id").count()
    
    #set of distinct patient id
    patientDead = set()
    patientAlive = set()
    for mort in mortPerPatient.iterrows():
        patientDead.add(mort[0])
    for event in eventPerPatient.iterrows():
        if event[0] not in patientDead:
            patientAlive.add(event[0])
    '''

    indx_date = mortality[["patient_id", "timestamp"]].copy()
    indx_date.rename(columns={ indx_date.columns[1]: "indx_date" }, inplace=True)
    indx_date["indx_date"] = indx_date["indx_date"].map(lambda x : parser.parse(x) + datetime.timedelta(days=-30))

    indx_date1 = events[["patient_id", "timestamp"]].copy()
    indx_date1.columns = ["patient_id", "indx_date"]
    indx_date1["indx_date"] = indx_date1["indx_date"].map(lambda x: parser.parse(x))
    indx_date1 = indx_date1.groupby("patient_id").max().reset_index()

    indx_date = pd.concat([indx_date, indx_date1])
    indx_date = indx_date.drop_duplicates(["patient_id"])
    indx_date["indx_date"] = indx_date["indx_date"].map(lambda x: x.strftime("%Y-%m-%d"))

    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    filtered_events = pd.merge(events, indx_date)
    filtered_events["indx_date"] = filtered_events["indx_date"].map(lambda x: parser.parse(x))
    filtered_events["timestamp"] = filtered_events["timestamp"].map(lambda x: parser.parse(x))
    filtered_events = filtered_events[filtered_events["indx_date"] - filtered_events["timestamp"] <= datetime.timedelta(days=2000)]
    filtered_events = filtered_events[filtered_events["indx_date"] >= filtered_events["timestamp"]]
    filtered_events["indx_date"] = filtered_events["indx_date"].map(lambda x: x.strftime("%Y-%m-%d"))
    filtered_events["timestamp"] = filtered_events["timestamp"].map(lambda x: x.strftime("%Y-%m-%d"))
    filtered_events = filtered_events.filter(items=['patient_id', 'event_id', "value"])

    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'],
                           index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath.
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    aggregated_events = pd.merge(filtered_events_df, feature_map_df)[["patient_id", "idx", "value"]].copy()
    aggregated_events = aggregated_events.dropna(axis=0, how='any')
    aggregated_eventsSum = aggregated_events[aggregated_events["idx"] < 2680]
    aggregated_eventsSum = aggregated_eventsSum.groupby(["patient_id", "idx"]).sum().reset_index()
    aggregated_eventsCnt = aggregated_events[aggregated_events["idx"] >= 2680]
    aggregated_eventsCnt = aggregated_eventsCnt.groupby(["patient_id", "idx"]).count().reset_index()
    aggregated_eventsCnt[["value"]] = aggregated_eventsCnt[["value"]].astype(float)
    aggregated_events = pd.concat([aggregated_eventsSum, aggregated_eventsCnt])

    aggregated_events_1 = aggregated_events.groupby("idx")["value"].max().reset_index()
    aggregated_events_1.rename(columns={ aggregated_events_1.columns[1]: "maxVal" }, inplace=True)
    aggregated_events = pd.merge(aggregated_events, aggregated_events_1)
    aggregated_events["value"] = aggregated_events["value"] / aggregated_events["maxVal"]
    aggregated_events = aggregated_events[["patient_id", "idx", "value"]].copy()

    aggregated_events.columns = ['patient_id', 'feature_id', 'feature_value']
    aggregated_events = aggregated_events.sort_values(by=['patient_id', 'feature_id'])
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',
                             columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
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

    for row in mortality.iterrows():
        if row[1]["patient_id"] in mortality1.keys():
            mortality1[row[1]["patient_id"]] = 1


    return patient_features, mortality1

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')


    for key in patient_features.keys():
        label = mortality[key]

        deliverable1.write(bytes((str(label)), 'UTF-8'))  # Use 'UTF-8'
        deliverable2.write(bytes((str(key)), 'UTF-8'))

        for t in patient_features[key]:
            deliverable1.write(bytes((" " + str(t[0]) + ":" + str(t[1])), 'UTF-8'))
            deliverable2.write(bytes((" " + str(t[0]) + ":" + str(t[1])), 'UTF-8'))
        deliverable1.write(bytes(("\n"), 'UTF-8'))
        deliverable2.write(bytes(("\n"), 'UTF-8'))

    deliverable1.close()
    deliverable2.close()

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()