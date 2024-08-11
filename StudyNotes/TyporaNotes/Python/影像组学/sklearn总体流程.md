## 使用[scikit-learn](https://scikit-learn.org/stable/)框架进行影像组学的流程

### 一、导入要用到的包

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

```

### 二、导入数据

```python
data = pd.read_excel(r"E:\PycharmProjects\StudyPython\BreastHer2\data\Features.xlsx")
```

### 三、划分X和Y

```python
X = data.drop(['Label', 'Patient_ID'], axis=1)
Y = data['Label']
```

### 四、划分训练集和测试集

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### 五、定义并训练模型

```python
model = LinearRegression()
model.fit(X_train, Y_train)
```

### 六、根据模型计算测试集的预测

```python
Y_pred = model.predict(X_test)
```

### 七、计算指标

```python
MSE = mean_squared_error(Y_test, Y_pred)
R2 = r2_score(Y_test, Y_pred)
AUC = roc_auc_score(Y_test, np.array([1 if i > 0.5 else 0 for i in Y_pred]))
```