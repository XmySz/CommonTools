import pyodbc
import pandas as pd

# 定义连接参数
server = '2.0.6.33'
username = 'sa'
password = 'HZxxk8399'

# 建立与SQL Server的连接
connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};UID={username};PWD={password}'
conn = pyodbc.connect(connection_string, autocommit=True)

# 获取所有数据库的名称，排除系统数据库
databases = []
cursor = conn.cursor()
cursor.execute("""
    SELECT name FROM sys.databases 
    WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');
""")
for row in cursor.fetchall():
    databases.append(row[0])

# 存储结果的列表
results = []

# 遍历每个数据库，获取schema和table信息
for db in databases:
    conn.execute(f'USE [{db}]')
    cursor.execute("""
        SELECT 
            TABLE_CATALOG AS 数据库名称, 
            TABLE_SCHEMA AS 架构名, 
            TABLE_NAME AS 表名
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE';
    """)
    for row in cursor.fetchall():
        results.append(tuple(row))

# 将结果转换为DataFrame
df = pd.DataFrame(results, columns=['数据库名称', '架构名', '表名'])

# 输出结果到CSV文件
df.to_excel(r"C:\Users\Zyn__\Desktop\HIS住院数据库结构映射表.xlsx", index=False)

