- generate several new datasets (1000 row each)
    - standard data - no drift
    - data with low drift
    - data with medium drift
    - data with high drift

- save these batches in best suitable place 
(file system / postgres: it may be one table with a column of values(standard/low_drift/mid_drift/high_drift))

- work on frontend to enhance it and make it better without bugs. 
and add dropdownlist with new dataset flag (standard/low_drift/mid_drift/high_drift) and button that perform the following:
    - import data batches (ex:200 row) from the specified category
    - check for drift and trigger webhook if there is drift

- presentation with script