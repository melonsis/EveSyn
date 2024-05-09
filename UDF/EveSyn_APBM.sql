-- We strongly recommend you to delete all annotations before use.
-- For details of implemention, see files in ev-mechanisms and mechanisms.
DROP FUNCTION IF EXISTS public.evesyn_apbm(text, real, real);
CREATE OR REPLACE FUNCTION public.evesyn_apbm(
	tablename text,
    wsize real,
	epsilon real)
    RETURNS text
    LANGUAGE 'plpython3u'
AS $BODY$

from mbi import Dataset, Domain
import pandas as pd
import numpy as np
import itertools
import os
import json
from mechanisms.cdp2adp import cdp_rho
from photools.cliques import clique_read
import psycopg2
from sqlalchemy import create_engine, MetaData
def on_load():
    con = psycopg2.connect(database="fill", user="with", password="your own")
    sql_rho_cmd = "select * from \"rho_used\""
    rho_df = pd.read_sql(sql=sql_rho_cmd, con=con)
    return rho_df

def on_start(wsize, rho_remain):
    return rho_remain/wsize



def have_intersection(df1, df2):
    intersection = pd.merge(df1, df2, how='inner')
    return not intersection.empty

def error_cal(data,synth,workload):
    errors=[]
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    return np.mean(errors)

def on_end(engine,tablename):
    previous_synth_name = tablename+'_synth'
    current_synth_name = tablename+'_synth_temp'
    
    previous_synth =  Dataset.load(previous_synth)
    current_synth = Dataset.load(current_synth)
    data = Dataset.load(tablename)
    workload = list(itertools.combinations(data.domain, 2))
    error_p = error_cal(data,previous_synth,workload)
    error_c = error_cal(data,current_synth,workload)
    if error_c <= error_p:
        current_synth.df.to_sql(name=previous_synth_name, con=engine, index=False, if_exists = 'replace')
    else:
        budget = 0
    return budget


if __name__ == "__main__":
    connection = 'postgresql+psycopg2://Fill:With@localhost:5432/yourown'
    engine= create_engine(connection)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    if 'rho_used' not in metadata.tables:
        budget_init = {'Timestamp': [1],
        'Used': [epsilon/2],
        'Status':["end"]}
        budget_df = pd.DataFrame(budget_init)
        budget_df.to_sql('rho_used', engine, if_exists='replace', index=False)
    else:
        budget_df = on_load()
        last_row_status = df.iloc[-1]['Status']
        last_time_stamp = df.iloc[-1]['Timestamp']
        
        if last_row_status = 'end':
            if last_time_stamp >= wsize:
                rho_used = df.iloc[-(wsize):]['Budget'].sum()
            else:
                rho_used = df.iloc['Budget'].sum()
            
            rho_remain = epsilon-rho_used
            budget = on_start(wsize, rho_remain)
            new_row = {'Timestamp': [last_time_stamp+1],
                    'Used': [budget],
                    'Status':["start"]}
            budget_df = budget_df.append(new_row, ignore_index=True)
            budget_df.to_sql('rho_used', engine, if_exists='replace', index=False)
        else:
            budget = on_end(engine,tablename)
            budget_df.loc[df.index[-1], 'Status'] = 'end'
            if budget = 0:
                budget_df.loc[df.index[-1], 'Used'] = '0'
            budget_df.to_sql('rho_used', engine, if_exists='replace', index=False)

        $BODY$;

ALTER FUNCTION public.evesyn_apbm(text, real, real)
    OWNER TO test;
