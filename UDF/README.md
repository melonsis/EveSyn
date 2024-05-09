# UDF guide
Here we instruct how to use our provided UDFs.
## Preparation
1. Set database connect information in ```/UDF_dependencies/mbi/dataset.py```, ```EveSyn_APBM.sql```, ```EveSyn_Initialize.sql``` and  ```EveSyn_Update.sql```.
2. Copy all folders in  ```UDF_dependencies``` to your Python site-package folder, like
```
/usr/local/python3/lib/python3.8/site-packages/
```
or
```
~/.local/python3/lib/python3.8/site-packages/
```

3. Create tables with the structures shown below.

|Name|Column name|Data Source|Note
|----|----|----|----|
|```dataset```|Attributes of ```dataset```|```dataset```.csv| Partial dataset for initialize, not full dataset
|```dataset```_domain|DOMAIN, SIZE|```dataset```-domain.json|
|```dataset```_synth_domain|DOMAIN, SIZE|```dataset```-domain.json|Same as ```dataset```_domain

4. Create ```plpython3u``` language/extension in PostgreSQL by using
```
CREATE EXTENSION plpython3u;
```
## Usage
1. Import ```EveSyn_Initialize.sql```, ```EveSyn_APBM.sql``` and ```EveSyn_Update.sql``` to PostgreSQL with ```psql``` or simply paste the content of files which mentioned.
2. Set a scheduled task with database engine.
3. At each timestamp, run ```EveSyn_APBM()``` with 
```
SELECT EveSyn_apbm(tablename, wsize, budget);
```
4. At the first timestamp, run ```EveSyn_Initialize()``` with
```
SELECT EveSyn_init(dataset, budget);
```
Otherwise, run ```EveSyn_Update()``` with
```
SELECT EveSyn_update(dataset, budget);
```
5. Run ```EveSyn_APBM()``` with 
```
SELECT EveSyn_apbm(tablename, wsize, budget);
```
to determine the budgets consumed.

## Note
Since tracking the update logs needs to be implemented with a specific database engine, we do not implement APBM's "increment-only detection" feature in the example UDF. It could be implemented when setting a scheduled task for periodic updates using Event Scheduler (for MySQL) or pg_cron (for PostgreSQL).
