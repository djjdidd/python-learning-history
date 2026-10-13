import matplotlib.pyplot as plt

#from matplotlib_inline import backend_inline
#backend_inline.set_matplotlib_formats('svg')

#matlab方式
fig1=plt.figure()#创建新图窗
x=[1,2,3,4,5]#数据x值
y=[1,8,27,64,125]#数据y值
plt.plot(x,y)
plt.show()

#plt.plot(x,y)#plot先描点在连线

#fig1.savefig(r'D:\我的天.svg')#保存矢量图
#面向对象方式
ax1=plt.axes()
x=[1,2,3,4,5]#数据x值
y=[1,8,27,64,125]#数据y值
ax1.plot(x,y)
plt.show()

#绘制多线条
fig2=plt.figure()
x=[1,2,3,4,5]
y1=[2,3,5,7,8]
y2=[3,5,6,8,0]
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()

#绘制多子图
fig3=plt.figure()
plt.subplot(2,1,1),plt.plot(x,y1)#二行一列第一个
plt.subplot(2,1,2),plt.plot(x,y2)
plt.show()

#图表类型：二维图plot()，网格图imshow()，统计图hist()
#二维图颜色color
fig2=plt.figure()
x=[1,2,3,4,5]
y1=[2,3,5,7,8]
y2=[3,5,6,8,0]
plt.plot(x,y1,color='#7CB5EC')
plt.plot(x,y2,color='#A2A2D0')
plt.show()
#设置风格linestyle-,--,-.,:
fig2=plt.figure()
x=[1,2,3,4,5]
y1=[2,3,5,7,8]
y2=[3,5,6,8,0]
plt.plot(x,y1,linestyle='--')
plt.plot(x,y2,linestyle=':')
plt.show()
#设置粗细linewidth=xx  0.5-3
#设置标记marker . o ^ s d,尺寸用markersize
fig2=plt.figure()
x=[1,2,3,4,5]
y1=[2,3,5,7,8]
y2=[3,5,6,8,0]
y3=[8,35,55,37,38]
y4=[3,54,6,84,40]
y5=[23,34,54,7,48]
plt.plot(x,y1,marker='.')
plt.plot(x,y2,marker='o')
plt.plot(x,y3)
plt.plot(x,y4)
plt.plot(x,y5)
plt.show()

#网格图
import numpy as np
x=np.linspace(0,10,1000)
l=np.sin(x)*np.cos(x).reshape(-1,1)
fig3=plt.figure
plt.imshow(l)
plt.colorbar()#配置颜色条
plt.show()

#统计图
#bins为区间划分数量 alpha为透明度默认为一
#histtype表示类型 color表示颜色 edgecolor表示边缘颜色
data=np.random.randn(10000)
plt.hist(data,bins=20,alpha=0.5,histtype='stepfilled')
plt.show()

#坐标轴上下限lim
fig6=plt.figure()
plt.plot(x,y)
plt.xlim(1,5)
plt.ylim(1,125)
#坐标轴上下限axis
fig6=plt.figure()
plt.plot(x,y)
plt.axis([1,5,1,125])

#标题与坐标轴名称
fig6=plt.figure()
plt.plot(x,y)
plt.title('this is the title')
plt.xlabel('this is xlabel')
plt.ylabel('this is ylabel')

#图例legend
#loc图例位置 frameon图例边框 ncol图例列数
plt.plot(x,y1,label='y=x')
plt.plot(x,y2,label='y=0')
plt.legend()
#加网格
plt.grid()

