# 折叠-读取数据库
from impala.dbapi import connect
from kaitoupao2 import *

def data_decode(data):
    if isinstance(data,bytes):
        return data.decode('utf-8')
    else :
        return data

def run_sql(query_str):
    host='10.10.24.77'
    port = 10045
    db_name= 'snbmod'
    passwd = 'snbmod@2021'
    conn = connect(host,
                  port=port,
                  database=db_name,
                  user=db_name,
                  password=passwd,
                  auth_mechanism='plain')

    cursor = conn.cursor()
    with TimerContext():
        cursor.execute(query_str)
    d = cursor.description
    columns = [x[0].split('.')[1] if(len(x[0].split('.'))>1) else x[0] for x in d]
    res = cursor.fetchall()
    df = pd.DataFrame(res, columns= columns)
    for each in df.columns:
        df[each]=df[each].map(data_decode)
    return df

def pure_run_sql(query_str):
    host='10.10.24.77'
    port = 10045
    db_name= 'snbmod'
    passwd = 'snbmod@2021'
    conn = connect(host,
                  port=port,
                  database=db_name,
                  user=db_name,
                  password=passwd,
                  auth_mechanism='plain')
    cursor = conn.cursor()
    with TimerContext():
        cursor.execute(query_str)
        
def download_from_hive(table_name):
    print(table_name)
    sql_here = f'''hive -e "set hive.resultset.use.unique.column.names=false;set hive.cli.print.header=true;select * from {table_name}" > {create_originalData_path(table_name.split('.')[-1] + ".csv")}'''
    os.system(sql_here)

pure_run_sql("set hive.security.authorization.enabled = false")