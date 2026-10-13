import numpy as np
import pandas as pd
print(np.__version__)  # 版本2.0.2
print(pd.__version__)  # 版本2.3.3


#一、数组基础#


#创建整型/浮点型数组.astype()方法

#arr1 = np.array([1, 2, 3])
#print(arr1)
# arr2 = np.array([1.0, 2, 3])
# print(arr2)
# #
# #整数数组转为浮点数数组
# arr3 = arr1.astype(float)  #arr1此时作为array类的实例
# print(arr3)
# #浮点数组转整数数组
# arr4 = arr2.astype(int)
# print(arr4)
#
# #整数数组遇到除法，转为浮点数结果
# print(arr1 / 1)

#数组的形状参数，三维数组：3层中括号，如[[[1,2,3]]]，形状为(x,y,z)
#(x,y,z,w)分别是每个维度的大小
#np.ones()函数，传入形状参数
#在处理批量数据时，不同的维度通常表示：
#第1维：批量大小
#第2维：通道数或特征图数
#第3、4维：空间维度（高度、宽度）

#所谓的形状，就是先数最外层的方括号，看它里面包含了几个次级方括号，然后就得到最外层的维数，内层维度也这样推知
#例如：
#[[[1]
#  [2]
#  [3]
#  [4]]]
#去掉最外层括号，还有一层括号，所以x=1，该层括号内含4个括号，所以y=4，在4个括号内，每个都不再内含括号，所以z=1，形状（1，4，1）

# arr1 = np.ones(3)
# print(arr1)
# arr2 = np.ones((1, 3)) #一个元素，每个元素包含三个维度
# print(arr2)
# arr3 = np.ones((2, 3, 4))  #2个元素，每个元素3个维度，每个维度下有4个元素
# print(arr3)
#
# print(arr3.shape)
#
# #数组维度的转换
# #print(np.arange.__doc__) #查看函数说明文档
#
# arr1 = np.arange(12)  #创建一维数组
# print(arr1)
# arr0 = np.arange(12).reshape(2, 6) #方法的链式调用
# print(arr0)
# #升级为二维数组，-1用于自动推定参数
# arr2 = arr1.reshape(1, -1)
# print(arr2)
# arr3 = arr2.reshape(-1, 6)
# print(arr3)
# arr4 = arr3.reshape(4, -1)
# print(arr4)
# #升级为2*2*3的三维数组
# arr5 = arr1.reshape(2, 2, -1)
# print(arr5)
# #降为一维数组
# arr6 = arr5.reshape(-1)
# print(arr6)



##二、数组的创建#
#

# #创建向量
# arr1 = np.array([1, 2, 3])
# print(arr1)
# #创建列矩阵
# arr2 = np.array([[1], [2], [3]])
# print(arr2)
# #创建行矩阵
# arr3 = np.array([[1, 2, 3]])
# print(arr3)
# print(arr3.shape)
#
# #递增数组
# arr1 = np.arange(10.0)  #浮点数组
# print(arr1)
# arr2 = np.arange(10, 20.0)
# print(arr2)
# arr3 = np.arange(1, 21, 2.0)
# print(arr3)
#
# #同值数组
# arr1 = 3.14 * np.ones((2, 3))
# print(arr1)
# arr2 = np.ones((1, 3))
# print(arr2)

#创建随机数组

# 
#0-1均匀分布的浮点型随机数组
#arr1 = np.random.random((2, 5))
#print(arr1)#
# #创建60-100范围内均匀分布的随机数组
#arr1 = (100 - 60) * np.random.random((3, 3)) + 60
#print(arr1)
#
# #整数型随机数组
# arr2 = np.random.randint(10, 100, (1, 15))
# print(arr2)
# #  或
# arr2 = ((100 - 10) * np.random.random((1, 15))).astype(int) + 10
# print(arr2)
#
# #正态数组: 参数（均值， 标准差， 形状）
# arr3 = np.random.normal(0, 1, (2, 3))
# print(arr3)
# #标准正态：
# arr3 = np.random.randn(2, 3)
# print(arr3)
# arr4 = np.random.randn() #默认输出一个元素
# print(arr4)
#
# #print(np.random.randn.__doc__)
#


#三、数组的索引#


#创建矩阵
#print(arr2[0, 2])
#print(arr2[1, -2])
# #修改元素
#arr2[1, 1] = 100.9
#print(arr2)
#
# #花式索引(两个中括号）：用向量代替普通索引的行列，并且输出结果也是向量
# arr1 = np.arange(0, 90, 10)
# print(arr1)
# #[1，1, 3，4]为索引列表，返回指定位置的元素
# print(arr1[[1, 1, 3, 4]])
#
# arr2 = np.arange(1, 17).reshape(4, 4)
# print(arr2)
#输出0行0列和1行1列
# print(arr2[[0, 1], [0, 1]])
# #输出位置为[0,2],[1,1],[2,0]的元素
# print(arr2[[0, 1, 2], [2, 1, 0]])
#
# #修改元素
# arr2[[0, 1, 2, 3], [3, 2, 1, 0]] = 100
# print(arr2)


##向量切片
#arr1=np.arange(10)
#print(arr1)
#print(arr1[1:4])
#print(arr1[:4])
#print(arr1[1:])
#print(arr1[::2]) #每两个元素采样一次
##矩阵的切片
# arr2 = np.arange(1, 21).reshape(4, 5)
# print(arr2)
# #获取1-2行、1-3列
# print(arr2[1: 3, 1: -1])
# #跳跃采样
# print(arr2[:: 3, :: 2])
#
# #提取矩阵的行
# arr3 = np.arange(1, 21).reshape(4, 5)
# print(arr3)
# #提取第2行
# print(arr3[2, :])
# #提取1-2行
# print(arr3[1: 3, :])
#
# #提取矩阵的列
#arr4 = np.arange(1, 21).reshape(4, 5)
#print(arr4)
# #提取第2列
#print(arr4[:, 2])
#提取后以向量表示
#print(arr4[:, 2].shape)
#cut=arr4.reshape((1,-1)) #升级为矩阵
#print(cut)
#print(cut.T) #转置操作，只对矩阵有用
# #在前面/外部添加维度，变为形状（1，4）的二维数组
#print(arr4[np.newaxis, :, 2])
#print(arr4[np.newaxis, :, 2].shape)
#
#
# # #提取第2列，以列向量表示
# print(arr4[:, 2: 3])
# print(arr4[:, 2: 3].shape) #形状为（4，1）
# #在前面添加维度，变为形状（1，4，1）的三维数组
# print(arr4[np.newaxis, :, 2: 3])
# print(arr4[np.newaxis, :, 2: 3].shape)
#
# #将一维数组/向量转为列向量/二维列矩阵，-1为自动推定行数
# print(arr4[:, 2, np.newaxis])
# print(arr4[:, 2].reshape(-1, 1))
# #提取1-2列
# print(arr4[:, 1: 3])
#
#
# #视图共享相同的数据内存,
# #数组切片仅仅是视图（或引用），不是副本，NumPy切片不会创建新的变量
# arr = np.arange(10)
# cut = arr[: 3]
# print(cut)
# cut[0] = 100
# print(arr)
# #若要备份切片为新变量：
# copy = arr[: 3].copy()
# copy[0] = 100
# print(arr)  #原数组不受影响
#
# #数组之间的赋值只是绑定，不创建新变量
# arr1 = np.arange(10)
# arr2 = arr1
# print(arr2)
# arr2[0] = 100
# print(arr1)



#四、数组的变形#

#数组的转置, 只对矩阵有效

# #向量的转置
# arr1 = np.arange(1, 4)
# arr2 = arr1.reshape(1, -1) #升级为矩阵
# print(arr2)
# arr3 = arr2.T #转置
# print(arr3)
#
# #列矩阵的转置
# arr1 = np.arange(3).reshape(3, 1)
# arr2 = arr1.T
# print(arr2)
#
# #矩阵的转置
# arr1 = np.arange(4).reshape(2, 2)
# print(arr1)
# print(arr1.T)
#
#
# #数组的翻转
# arr1 = np.arange(10)
# print(arr1)
# #向量只能使用up-down, 因为数学中默认向量是列向量
# arr_ud = np.flipud(arr1)
# print(arr_ud)
# #矩阵翻转
# arr2 = np.arange(1, 21).reshape(4, 5)
# print(arr2)
# arr_lr = np.fliplr(arr2)#左右
# print(arr_lr)
# arr_ud = np.flipud(arr2)#上下
# print(arr_ud)
#
# #矩阵的变形
#
# #嵌套列表转矩阵
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr1)
# #变形为向量
# arr2 = arr1.reshape(6)
# print(arr2)
# #继续变形为矩阵
# arr3 = arr2.reshape(1, -1)
# print(arr3)
#
# #数组的拼接
#
# #向量拼接
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# arr3 = np.concatenate([arr1, arr2])
# print(arr3)
#
# #同型矩阵的矩阵拼接
# arr1 = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
# arr2 = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
# #按第一个维度（行）拼接，拼后形状（6，3）
# arr3 = np.concatenate([arr1, arr2])  #默认axis=0
# print(arr3)
# #按第二个维度（列）拼接，拼后形状（2，6）
# arr4 = np.concatenate([arr1, arr2], axis=1)
# print(arr4)
#
#
# #数组的分裂
#
# #ny.split(ary, indices_or_sections, axis=0)
# #ary: 要分割的数组.ndices_or_sections: 分割点，可以是：
# #整数：将数组等分成指定数量的子数组；整数列表：在指定索引位置进行分割
# #axis: 沿哪个轴进行分割（默认为0，即第一个轴
#
# arr = np.arange(10, 100, 10)
# print(arr)
# arr1, arr2, arr3 = np.split(arr, [2, 8])
# print(arr1)
# print(arr2)
# print(arr3)
#
# #矩阵的分裂
#arr = np.arange(1, 17).reshape(4, 4)
#print(arr)
# #按第一个维度分裂
#arr1, arr2, arr3 = np.split(arr, [1, 2]) #以第一行第二行为分割线，分成三部分
#print(arr1, '\n\n', arr2, '\n\n', arr3)
# #按第二个维度分裂
#arr1, arr2, arr3 = np.split(arr, [1, 3], axis=1) #以第一列、第三列为分割线，分成三部分
#print(arr1, '\n\n', arr2, '\n\n', arr3)




#五、数组的运算#



#数组与系数之间的运算

# arr = np.arange(1, 9).reshape(2, 4)
# print(arr)
# print(arr + 10)
# print(arr - 10)
# print(arr * 10)
# print(arr / 10)
# print(arr ** 2)
# print(arr // 6)
# print(arr % 6)  #这些运算未改变数组的形状
#
# #数组之间的运算
#
# arr1 = np.arange(-1, -9, -1).reshape(2, 4) #若设置步长为1，则从起始值永远无法递增到达结束值，此时规定生成一个空数组
# arr2 = -arr1
# print(arr1)
# print(arr2)
# print(arr1 + arr2)
# print(arr1 - arr2)
# print(arr1 * arr2)
# print(arr1 / arr2)
# print(arr1 ** arr2)


#广播

#向量与矩阵做运算，向量自动升级为矩阵
#行或列矩阵，经过广播，适配另一个矩阵形状

# #向量广播
# arr1 = np.array([-100, 0, 100])
# print(arr1)
# arr2 = np.random.random((10, 3))
# print(arr2)
# print(arr1 * arr2)
# #行矩阵广播
# arr3 = arr1.reshape(1, 3)
# print(arr3)
# print(arr3 * arr2)

# #列矩阵广播,沿水平方向广播
# arr1 = np.arange(3).reshape(3, 1)
# print(arr1)
# arr2 = np.ones((3, 5))
# print(arr2)
# print(arr1 * arr2)
#
# #行列矩阵同时广播（少见用法）
# arr1 = np.arange(3)
# print(arr1)
# arr2 = np.arange(3).reshape(3, 1)
# print(arr2)
# print(arr1 * arr2)
#



# 数组的函数 #



# 矩阵乘积（线性代数版）

# # 向量内积
# arr1 = np.arange(5)
# arr2 = np.arange(5)
# print(np.dot(arr1, arr2))
#
# # 向量与矩阵
# arr1 = np.arange(4)
# arr2 = np.arange(12).reshape(4, 3)
# print(arr1)
# print(arr2)
# # 当乘法中存在向量时，输出仍为向量
# print(np.dot(arr1, arr2))
# print(np.dot(arr1, arr2).shape)
# # 向量转为行矩阵再相乘，输出矩阵
# arr3 = np.arange(4).reshape(1, 4)
# # 必须按照线性代数中矩阵乘法前后顺序来写，写成np.dot(arr2, arr3)则报错
# print(np.dot(arr3, arr2))
# print(np.dot(arr3, arr2).shape)
#
# # 矩阵乘向量
# arr1 = np.arange(12).reshape(3, 4)
# arr2 = np.arange(4)
# print(arr1)
# print(arr2)
# print(np.dot(arr1, arr2))

# # 矩阵与矩阵
# arr1 = np.arange(10).reshape(5, 2)
# print(arr1)
# arr2 = np.arange(16).reshape(2, 8)
# print(arr2)
# print(np.dot(arr1, arr2))


# # 重要数学函数

# # 绝对值函数
# arr_v = np.array([-10, 0, 10])
# abs_v = np.abs(arr_v)
# print('原数组是：', arr_v)
# print('绝对值是：', abs_v)
#
# # 三角函数
# theta = np.arange(3) * np.pi / 2
# sin_v = np.sin(theta)
# cos_v = np.cos(theta)
# tan_v = np.tan(theta)
# print('原数组：', theta)
# print('正弦值：', sin_v)
# print('余弦值：', cos_v)
# print('正切值：', tan_v) #e++意思是e*10的几次方

# # 指数函数
# x = np.arange(1, 4)
# print('x   =', x)
# print('e^x =', np.exp(x))
# print('2^x =', 2 ** x)
#
# # 对数函数
#x = np.array([1, 10, 100, 1000])
#print('x    =', x)
#print('ln(x) =', np.log(x))
#print('log2(x) =', np.log2(x))
#print('log2(x) =', np.log(x) / np.log(2))  # 换底公式写法
#print('log10(x) =', np.log10(x))


# # 聚合函数
#
# # 最大值/最小值函数
# arr = np.random.random((2, 3))
# print(arr)
# # axis=0，即按行比较，每行的第i个元素互相比较，所以输出结果也是i个
# print('按维度一求最大值：', np.max(arr, axis=0))
# # axis=1， 按列比较，每列的第j个元素互相比较，输出结果也是j个
# print('按维度二求最大值，', np.max(arr, axis=1))
# print('整体求最大值，', np.max(arr))
#
# # 求和函数np.sum(), 求积函数np.prod()
# arr = np.arange(10).reshape(2, 5)
# print(arr)
# # axis=0，每行第i个元素都相加；axis=1，每列第j个元素都相加
# print('按维度一求和：', np.sum(arr, axis=0))
# print('按维度二求和：', np.sum(arr, axis=1))
# print('整体求和：', np.sum(arr))
#
# # 均值函数np.mean(), 标准差函数np.std()
# arr = np.arange(10).reshape(2, 5)
# print(arr)
# print('按维度一求平均：', np.mean(arr, axis=0))
# print('按维度二求平均：', np.mean(arr, axis=1))
# print('整体求平均：', np.mean(arr))
#
# # 聚合函数碰到缺失值会报错，因此出现了聚合函数的安全版本，计算时忽略缺失值：
# # 如：np.nansum(), np.nanmean(), np.min()，nan意味“非数”




# 七、布尔数组 #


# # 创建布尔数组
#
# # 数组与数字比较
# arr = np.arange(1, 7).reshape(2, 3)
# print(arr)
# print(arr <= 4)

# # 同维数组比较
# arr1 = np.arange(1, 6)
# arr2 = np.flipud(arr1)
# print(arr1)
# print(arr2)
# print(arr1 > arr2)
#
# # 多个条件比较: &, |, ~(非)
# arr = np.arange(1, 10)
# print(arr)
# print((arr < 4) | (arr > 6))
# print(arr)
#
# # 布尔数组中True的数量
# arr = np.random.normal(0, 1, 10000)
# # 统计该分布中绝对值小于1的元素个数
# num = np.sum(np.abs(arr) < 1)
# print(num)  #结果约为6800，因为标准正态曲线中(μ-σ，μ+σ)面积约占68%


# # np.any()函数，只要布尔型数组里含有True，就返回True
# arr1 = np.arange(1, 10)
# arr2 = np.flipud(arr1)
# print(arr1)
# print(arr2)
# print(arr1 == arr2)
# print((np.any(arr1 == arr2)))
#
# # np.all()函数，全是True才返回True
# # 模拟六级成绩分布
# arr = np.random.normal(500, 70, 100000)
# # 判断是否所有考生成绩都高于250
# print(arr > 250)
# # 根据3σ准则，99.73%考生成绩高于290分（290=500-3*70），但仍有低于250分者
# print(np.all(arr > 250))


# # 布尔数组作为掩码
# # 若一个普通数组和一个布尔数组维度相同，可以用布尔数组筛选元素
#
# arr = np.arange(1, 13).reshape(3, 4)
# print(arr)
# print(arr > 4)
# # 筛选出arr>4的元素
# print(arr[arr > 4])  # 输出的是向量
#
# # 筛选出数组逐元素比较的结果
# arr1 = np.arange(1, 10)
# arr2 = np.flipud(arr1)
# print(arr1)
# print(arr2)
# print(arr1 > arr2)
# print(arr1[arr1 > arr2])
# print(arr2[arr1 > arr2])


# # 满足条件的元素所在位置,np.where()
#
# # 模拟六级成绩，找出最高分所在位置
#arr = np.random.normal(500, 70, 10)
#print(arr)
#print(arr > 600)
# # np.where函数默认返回值是元组
#print(np.where(arr > 600))
#print(np.where(arr == np.max(arr)))
# # 不再输出元组，只提取索引位置：[0]用于提取元组的第一个元素
#print(np.where(arr > 600)[0])
#print(np.where(arr == np.max(arr))[0])




# 八、从数组到张量 #


# 数组与张量
# numpy和pytorch基础语法几乎一致：
# np对应torch
# 数组array对应张量tensor
# numpy的n维数组对应pytorch的n阶张量
# 数组与张量可以相互转换：
# 数组arr转为张量ts: ts = torch.tensor(arr)
# 张量ts转为数组arr: arr = np.array(ts)

# 语法不同点：
#      左numpy，              右pytorch          区别
# .astype()                  .type()
# np.random.random()          torch.rand()
# np.random.randint()        torch.randint()   不接纳向量（一维数组）
# np.random.normal()         torch.normal()    不接纳向量
# np.random.randn()          torch.randn()
# .copy()                    .clone()
# np.concatenate()           torch.cat()
# np.split()                 torch.split()     参数的用法更灵活
# np.dot()                   torch.matmul()    matmul是万能的通用的
# np.dot(v, v)  # v是向量     torch.dot()
# np.dot(m, v)  # m是矩阵     torch.mv()
# np.dot(m, m)               torch.mm()
# np.exp()                   torch.exp()       必须传入张量
# np.log()                   torch.log()       必须传入张量
# np.mean(）                 torch.mean()      必须传入浮点数组
