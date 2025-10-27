# mood-detection
The code is inside the /code folder. 
The results are in the /results folder. 
The data are in the /data folder. 

To run the code, you need fastparquet, scikit-learn, pandas, numpy installed in your environment.
For the country specific experiment, run the code python.exe experiment_country_specific.py. For the second experiment, use python experiment_transfer_country_agnostic_I.py to train on another country and test on Mak. The last experiment_transfer_country_agnostic_II.py trains on all other countries and tests on Mak.
