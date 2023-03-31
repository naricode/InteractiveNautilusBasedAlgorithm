import numpy as np

#####  Standard deviation method #######

anomalies = []

# multiply and add by random numbers to get some real values
data1 = np.random.randn(50000)  * 20 + 20

# Function to Detection Outlier on one-dimentional datasets.
def find_anomalies(random_data):
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    print(lower_limit)
    # Generate outliers
    for outlier in random_data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

# find_anomalies(data1)

data2=np.random.randn(40000)
def find_outliers(rand_data):
    stde=np.std(rand_data)
    mean=np.mean(rand_data)
    limit=3*stde
    lower_bound=mean - limit
    print(lower_bound)
    upper_bound=mean + limit
    print(upper_bound)
    for outlier in rand_data:
        if outlier > upper_bound or outlier < lower_bound:
            anomalies.append(outlier)
    return anomalies

find_outliers(data2)
cs=0
