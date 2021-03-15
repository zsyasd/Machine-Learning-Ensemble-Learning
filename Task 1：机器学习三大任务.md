# 什么是机器学习

机器学习是一门致力于研究如何通过计算的手段，利用经验来改善系统自身性能的学科。在计算机系统中，「经验」通常以「数据」形式存储机器学习的一个重要的目标就是利用数学模型来理解数据，发现数据中的规律，用作数据的分析和预测。在计算机中，数据通常由一组向量组成，这组向量中的每个向量都是一个样本，我们用$x_i$来表示一个样本，其中$i=1,2,3,...,N$,共N个样本，每个样本$x_i=(x_{i1},x_{i2},...,x_{ip},y_i)$共p+1个维度，前p个维度的每个维度我们称为一个特征，最后一个维度$y_i$我们称为因变量(响应变量)。特征用来描述影响因变量的因素，如：我们要探寻身高是否会影响体重的关系的时候，身高就是一个特征，体重就是一个因变量。通常在一个数据表dataframe里面，一行表示一个样本$x_i$，一列表示一个特征。      
根据数据是否有因变量，机器学习的任务可分为：**有监督学习**和**无监督学习**。

   - 有监督学习：给定某些特征去估计因变量，即因变量存在的时候，我们称这个机器学习任务为有监督学习。如：我们使用房间面积，房屋所在地区，环境等级等因素去预测某个地区的房价。          
   - 无监督学习：给定某些特征但不给定因变量，建模的目的是学习数据本身的结构和关系。如：我们给定某电商用户的基本信息和消费记录，通过观察数据中的哪些类型的用户彼此间的行为和属性类似，形成一个客群。注意，我们本身并不知道哪个用户属于哪个客群，即没有给定因变量。

![1 1](https://user-images.githubusercontent.com/55370336/111173762-defa1800-85e1-11eb-93f6-abd310451c5a.png)

根据因变量的是否连续，有监督学习又分为**回归**和**分类**：
   - 回归：因变量是连续型变量，如：房价，体重等。
   - 分类：因变量是离散型变量，如：是否患癌症，西瓜是好瓜还是坏瓜等。

为了更好地叙述后面的内容，我们对数据的形式作出如下约定：      
第i个样本：$x_i=(x_{i1},x_{i2},...,x_{ip},y_i)^T,i=1,2,...,N$     
因变量$y=(y_1,y_2,...,y_N)^T$        
第k个特征:$x^{(k)}=(x_{1k},x_{2k},...,x_{Nk})^T$     
特征矩阵$X=(x_1,x_2,...,x_N)^T$

在机器学习中，我们通常会使用 `sklearn`工具库来探索机器学习项目。

```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use("ggplot")      
import seaborn as sns
```

## 回归

寻找数据中规律的问题是一个基本的问题，有着很长的很成功的历史。机器学习领域关注的是利用计算机算法自动发现数据中的规律，以及使用这些规律采取将数据分类等行动。我们以波士顿房价为例：

```python
from sklearn import datasets
boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
boston_data["Price"] = y
```

sklearn中所有内置数据集都封装在datasets对象内：
返回的对象有：
   - data:特征X的矩阵(ndarray)
   - target:因变量的向量(ndarray)
   - feature_names:特征名称(ndarray)

让我们来看一看数据

```python
boston_data.head()
```



![image](https://user-images.githubusercontent.com/55370336/111174179-30a2a280-85e2-11eb-88a3-a4af39847c0a.png)

```python
boston_data
```



![image](https://user-images.githubusercontent.com/55370336/111174205-37311a00-85e2-11eb-828f-f6c39a4c8dd2.png)

各个特征的相关解释：

* CRIM：各城镇的人均犯罪率
* ZN：规划地段超过25,000平方英尺的住宅用地比例
* INDUS：城镇非零售商业用地比例
* CHAS：是否在查尔斯河边(=1是)
* NOX：一氧化氮浓度(/千万分之一)
* RM：每个住宅的平均房间数
* AGE：1940年以前建造的自住房屋的比例
* DIS：到波士顿五个就业中心的加权距离
* RAD：放射状公路的可达性指数
* TAX：全部价值的房产税率(每1万美元)
* PTRATIO：按城镇分配的学生与教师比例
* B：1000(Bk - 0.63)^2其中Bk是每个城镇的黑人比例
* LSTAT：较低地位人口
* Price：房价

既然是回归任务，那让我们看看给定自变量（feature）的条件下因变量(target)的变化情况

```python
sns.scatterplot(boston_data['NOX'],boston_data['Price'],color="r",alpha=0.6)
plt.title("Price~NOX")
plt.show()
```

![image](https://user-images.githubusercontent.com/55370336/111174299-4a43ea00-85e2-11eb-8bca-31329042dca1.png)

## 分类

给每个输入向量分配到有限数量离散标签中的一个，被称为分类(classification)问题。如果要求的输出由一个或者多个连续变量组成，那么这个任务被称为回归(regression)。

我们可以看一个鸢尾花类别分类的例子，来自大名鼎鼎的iris数据集：

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
iris_data = pd.DataFrame(X,columns=features)
iris_data['target'] = y
iris_data.head()
```

![image](https://user-images.githubusercontent.com/55370336/111174507-72cbe400-85e2-11eb-9b53-957e546f926d.png)



```python
# 可视化特征
marker = ['s','x','o']
for index,c in enumerate(np.unique(y)):
    plt.scatter(x=iris_data.loc[y==c,"sepal length (cm)"],y=iris_data.loc[y==c,"sepal width (cm)"],alpha=0.8,label=c,marker=marker[c])
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()
```

![img](https://user-images.githubusercontent.com/55370336/111174532-76f80180-85e2-11eb-910d-56a029e213b0.png)

我们可以看到：每种不同的颜色和点的样式为一种类型的鸢尾花，数据集有三种不同类型的鸢尾花。因此因变量是一个类别变量，因此通过特征预测鸢尾花类别的问题是一个分类问题。

各个特征的相关解释：

* sepal length (cm)：花萼长度(厘米)
* sepal width (cm)：花萼宽度(厘米)
* petal length (cm)：花瓣长度(厘米)
* petal width (cm)：花瓣宽度(厘米)

## 无监督学习

在其他的机器学习问题中，训练数据由一组输入向量x组成，没有任何对应的目标值。 在这样的`无监督学习(unsupervised learning)`问题中，目标可能是发现数据中相似样本的 分组，这被称为`聚类(clustering)`，或者决定输入空间中数据的分布，这被称为`密度估计 (density estimation)`，或者把数据从高维空间投影到二维或者三维空间，为了`数据可视化 (visualization)`。

我们可以使用sklearn生成符合自身需求的数据集，下面我们用其中几个函数例子来生成无因变量的数据集：
https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets

![1.2](https://user-images.githubusercontent.com/55370336/111173774-e0c3db80-85e1-11eb-96c8-cca4de4e669e.png)



```python
# 生成月牙型非凸集
from sklearn import datasets
x, y = datasets.make_moons(n_samples=2000, shuffle=True,
                  noise=0.05, random_state=None)
for index,c in enumerate(np.unique(y)):
    plt.scatter(x[y==c,0],x[y==c,1],s=7)
plt.show()
```

![image](https://user-images.githubusercontent.com/55370336/111174787-b0c90800-85e2-11eb-9201-4ee7a2d8ae32.png)



```python
# 生成符合正态分布的聚类数据
from sklearn import datasets
x, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=3)
for index,c in enumerate(np.unique(y)):
    plt.scatter(x[y==c, 0], x[y==c, 1],s=7)
plt.show()
```

![image](https://user-images.githubusercontent.com/55370336/111174803-b58dbc00-85e2-11eb-990c-fe5721a94147.png)



