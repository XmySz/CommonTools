<hr style="height: 4px; background-color: black; border: none;">

## 一、Data Definition Language

<hr style="height: 4px; background-color: black; border: none;">

### CREATE

```sql
CREATE TABLE table_name (
column1 data_type(size),
column2 data_type(size),
...
);
```

### DROP

```sql
DROP TABLE table_name;
```

### ALTER

```sql
ALTER TABLE table_name ADD column_name datatype;
ALTER TABLE table_name DROP COLUMN column_name;
ALTER TABLE table_name MODIFY COLUMN column_name datatype(size);

ALTER TABLE tableName ADD CONSTRAINT constraintName PRIMARY KEY (column1, column2, ... column_n);
ALTER TABLE tableName DROP CONSTRAINT constraintName;
ALTER TABLE table_name DROP PRIMARY KEY;
```

### TRUNCATE

```sql
TRUNCATE TABLE table_name;
```

### RENAME

```sql
ALTER TABLE table_name RENAME TO new_table_name;
```

```sql
ALTER TABLE table_name RENAME COLUMN old_column_name TO new_column_name;
```

<hr>

<hr>

<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 二、Data Manipulation Language

<hr style="height: 4px; background-color: black; border: none;">

### SELECT

```sql
SELECT column1, column2, ... FROM table_name;
```

```sql
SELECT DISTINCT column1, column2, ... FROM table_name;
```

```sql
SELECT column1, column2, ... FROM table_name WHERE condition;
```

```sql
SELECT column1, column2, ... FROM table_name ORDER BY column ASC|DESC;
```

```sql
SELECT column1, column2 FROM table_name GROUP BY column1, column2;
```

```sql
SELECT column_name, function(column_name) FROM table_name WHERE condition GROUP BY column_name HAVING function(column_name) condition value;
```

```sql
SELECT table1.column1, table2.column2... FROM table1 INNER JOIN table2 ON table1.matching_column = table2.matching_column;
```

```sql
SELECT table1.column1, table2.column2... FROM table1 LEFT JOIN table2 ON table1.matching_column = table2.matching_column;
```

```sql
SELECT table1.column1, table2.column2... FROM table1 RIGHT JOIN table2 ON table1.matching_column = table2.matching_column;
```

```sql
SELECT table1.column1, table2.column2... FROM table1 FULL JOIN table2 ON table1.matching_column = table2.matching_column;
```

```sql
SELECT a.column_name, b.column_name... FROM table_name AS a, table_name AS b WHERE condition;
```

```sql
SELECT table1.column1, table2.column2... FROM table1, table2;
```

### INSERT

```sql
INSERT INTO table_name cVALUES (value1, value2, ..., valueN);
```

```sql
INSERT INTO table_name VALUES (value1, value2, ..., valueN);
```

```sql
INSERT INTO table1 (column1, column2, ... , columnN) SELECT column1, column2, ... , columnN FROM table2 
WHERE condition;
```

### UPDATE

```sql
UPDATE table_name SET column1 = value1, column2 = value2, ... WHERE condition;
```

### DELETE

```sql
DELETE FROM table_name [WHERE condition]
```

 <hr>

<hr>

<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 三、Aggregate Queries

<hr style="height: 4px; background-color: black; border: none;">

### ⭐Numeric

- #### FLOOR() CEILING() ABS() MOD() ROUND() AVG() MIN() MAX() VAR() STDEV() SUM() COUNT() SQRT() PI() ...  

### ⭐String

- #### CONCAT() LENGTH() SUBSTRING() REPLACE() UPPER() LOWER() REVERSE() ⭐TRANSLATE()...

- #### TRIM(): TRIM 函数删除字符串的前导和尾随空格, 还可以删除其他指定的字符。

  ```sql
  TRIM([LEADING|TRAILING|BOTH] [removal_string] FROM original_string)
  ```

### ⭐Date And Time

- #### YEAR() 

- #### MONTH() 

- #### DAY()

- #### CURRENT_DATE:返回当前的日期。 sql server中无法使用

  ```sql
  SELECT CURRENT_DATE;
  ```

- #### GETDATE():以 DateTime数据类型返回当前日期和时间, 它不需要任何参数。

  ```sql
  SELECT GETDATE() AS CurrentDateTime;
  ```

- #### DATEDIFF():根据要使用的时间单位返回两个日期值之间的差异。

  ```sql
  SELECT DATEDIFF(day, '2022-01-01', '2022-01-15') AS DiffInDays;
  ```

- #### DATEADD():从日期中添加或减去指定的时间间隔。

  ```sql
  SELECT DATEADD(year, 1, '2022-01-01') AS NewDate;
  ```

- #### DATEPART():提取日期或时间字段的特定部分。

  ```sql
  DATEPART(datepart, date)
  ```

  ```
  SELECT DATEPART(year, '2021-07-14') AS 'Year';
  SELECT DATEPART(month, '2021-07-14') AS 'Month';
  SELECT DATEPART(day, '2021-07-14') AS 'Day';
  SELECT DATEPART(hour, '2021-07-14T13:30:15') AS 'Hour',
         DATEPART(minute, '2021-07-14T13:30:15') AS 'Minute',
         DATEPART(second, '2021-07-14T13:30:15') AS 'Second';
  SELECT DATEPART(weekday,'2021-07-14') AS WeekdayNumber;
  SELECT DATEPART(iso_week, '2021-07-14') AS ISOWeekNumber;	
  SELECT DATEPART(dayofyear, '2021-07-14') AS DayOfYear;
  SELECT DATEPART(dy, '2021-07-14) AS DayOfYear;
   
  ```

- #### TIMESTAMP: 存储日期和时间的数据类型, 通常用于跟踪对记录所做的更新和更改,提供发生的时间顺序。

  ```sql
  CREATE TABLE table_name (    
     column1 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,		--每次更新行时自动更新时间戳
     column2 VARCHAR(100),
     ...
  );
  ```

- #### CONVERT():用于从一种数据类型转换为另一种数据类型，通常用于格式化 DateTime 值。

  ```sql
  SELECT CONVERT(VARCHAR(19), GETDATE()) AS FormattedDateTime;
  ```

<hr>

<hr>

<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 四、Data Constraints

<hr style="height: 4px; background-color: black; border: none;">

### Primary Key

唯一标识数据库表中的每条记录。主键必须包含唯一值。与 UNIQUE 约束完全相同，但一张表中可以有多个唯一约束，但每个表只能有一个 PRIMARY KEY 约束。

```sql
CREATE TABLE Students (
    ID int NOT NULL,
    Name varchar(255) NOT NULL,
    Age int,
    PRIMARYc KEY (ID)
);
```

```sql
ALTER TABLE Employees ADD PRIMARY KEY (ID);
```

### Foreign Key

防止会破坏表之间链接的操作。外键是一个表中的字段（或字段集合），它引用另一表中的主键。

```sql
CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    ID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (ID) REFERENCES Students(ID)
);
```

```sql
ALTER TABLE Employees ADD PRIMARY KEY (ID);
```

### Unique

确保列中所有值都不同。

```sql
CREATE TABLE Students (
    ID int NOT NULL UNIQUE,
    Name varchar(255) NOT NULL,
    Age int
);
```

```sql
ALTER TABLE Employees ADD PRIMARY KEY (ID);
```

```sql
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

### NOT NULL

确保这一列没有NULL值。

```sql
CREATE TABLE Students (
    ID int NOT NULL,
    Name varchar(255) NOT NULL,
    Age int
);
```

```sql
ALTER TABLE table_name DROP CONSTRAINT constraint_name;
```

### CHECK

CHECK 约束确保列中的所有值都满足特定条件。

```sql
CREATE TABLE Students (
    ID int NOT NULL,
    Name varchar(255) NOT NULL,
    Age int,
    CHECK (Age>=18)
);
```

```sql
ALTER TABLE Employees ADD CONSTRAINT CHK_EmployeeAge CHECK (Age >= 21 AND Age <= 60);
```

### DEFAULT

当未指定列时，为列提供默认值。

```sql
CREATE TABLE Students (
    ID int NOT NULL,
    Name varchar(255) NOT NULL,
    Age int,
    City varchar(255) DEFAULT 'Unknown'
);
```

<hr>

<hr>

<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 五、Sub Queries/Nested Queries/Outer Queries

<hr style="height: 4px; background-color: black; border: none;">

```sql
SELECT column_name [, column_name] FROM table1 [, table2 ]
WHERE  column_name OPERATOR
   (SELECT column_name [, column_name]
   FROM table1 [, table2 ]
   [WHERE])
```

### Scalar Subquery

```sql
SELECT column_name [, column_name]
FROM table1 [, table2 ]
WHERE column_name OPERATOR
   (SELECT column_name [, column_nacme]
   FROM table1 [, table2 ]
   [WHERE])
```

### Row Subquery

```sql
SELECT * FROM student 
WHERE (roll_id, age)=(SELECT MIN(roll_id),MIN(age) FROM student);
```

### Column Subquery

```sql
SELECT column_name [, column_name ]
FROM table1 [, table2 ]
WHERE (SELECT column_name [, column_name ]
FROM table_name 
WHERE condition);
```

### Table Subquery

```sql
SELECT column_name [, column_name ]
FROM
	(SELECT column_name [, column_name ]
		FROM table1 [, table2 ])
WHERE condition;
```

<hr>

<hr>

<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 六、Conditional Expression

<hr style="height: 4px; background-color: black; border: none;">
### CASE

`CASE`表达式是一个流程控制语句，允许向查询添加 if-else 逻辑。

```sql
SELECT FirstName, City,
    CASE
        WHEN City = 'Berlin' THEN 'Germany'
        WHEN City = 'Madrid' THEN 'Spain'
        ELSE 'Unknown'
    END AS Country
FROM Customers;
```

### NULLIF

如果两个给定表达式相等，则`NULLIF`返回 null。

```sql
SELECT NULLIF(5,5) AS Same,
       NULLIF(5,7) AS Different;
```

### COALESCE

`COALESCE`函数返回列表中的第一个非空值。它采用逗号分隔的值列表并返回第一个不为空的值。

```sql
SELECT ProductName,
    COALESCE(UnitsOnOrder, 0) As UnitsOnOrder,
    COALESCE(UnitsInStock, 0) As UnitsInStock,
FROM Products;
```

### IIF

如果条件为 TRUE，则`IIF`函数返回 value_true；如果条件为 FALSE，则返回 value_false

```sql
SELECT IIF (1>0, 'One is greater than zero', 'One is not greater than zero');
```

<hr>
<hr>
<hr>
<hr style="height: 4px; background-color: black; border: none;">

## 七、VIEWS

<hr style="height: 4px; background-color: black; border: none;">

​		SQL views are virtual tables that do not store data directly. They are essentially a saved SQL query and can pull data from multiple tables or just present the data from one table in a different way.

### Creating Views

```sql
CREATE VIEW [OR ALTER] schema_name.view_name [(column_list)]
AS
    select_statement;
```

### Modifying Views

```sql
CREATE OR REPLACE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

### Dropping Views

```sql
DROP VIEW [IF EXISTS] 
    schema_name.view_name1, 
    schema_name.view_name2,
    ...;
```

<hr>

<hr>

<hr>

<hr style="height: 4px; background-color: black; border: none;">

## 八、INDEXES

<hr style="height: 4px; background-color: black; border: none;">

​		Indexes can drastically **speed up data retrieval** in SQL databases by allowing the database to immediately locate the data needed without having to scan the entire database. However, these additional data structures also consume storage, and maintaining them can slow down any create, update, or delete operations, hence the need to manage them appropriately.

### Creating Indexes

```sql
CREATE INDEX index_name ON table_name(column_name);
```

```sql
CREATE CLUSTERED INDEX index_name ON table_name (column_name);
```

```sql
CREATE NONCLUSTERED INDEX index_name ON table_name (column_name);
```

```sql
CREATE INDEX index_name ON table_name (column1, column2);
```

```sql
CREATE UNIQUE INDEX index_name ON table_name (column_name);
```

```sql
CREATE FULLTEXT INDEX ON table_name(column_name) KEY INDEX index_name;
```

### Removing Indexes

```sql
DROP INDEX index_name;
```

### Listing Indexes

```sql
SHOW INDEXES IN table_name;
```

### Modifying Indexes

```sql
REINDEX INDEX index_name;
```

