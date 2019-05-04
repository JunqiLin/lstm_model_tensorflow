
## LSTM for Time Series Prediction by Tensorflow

##### version
- python 2.7
- tensorflow 1.12.0 cpu
- plotly
- pymongo 3.7.2

##### database
- mongodb

##### file
- data_process.py 

	store data into mongodb, you need to run this file first because our model read data from database

- data_generator.py

	read data from database and process the data into the type needed by lstm model.
	
- model.py

	this is our core lstm model which includes two hidden layers and middle lstm layer. When you get the data, then run this file and see the result.
	
##### how to run
- install mongodb and run data_process.py(Actually we can read data directly from csv file but we want our code can cope with different data sources without modifying code and can read data more quickly so we use the database )
- run model.py






