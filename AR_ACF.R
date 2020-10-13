#自定义时间序列数据
ts(1:8,start=c(2015,2),frequency=4)

#导入火鸡数据
library(nutshell)
data(turkey.price.ts)

turkey.price.ts

#时间序列数据简单描述
start(turkey.price.ts)
end(turkey.price.ts)
frequency(turkey.price.ts)
deltat(turkey.price.ts

#绘制时间序列图
library(nutshell)
data(turkey.price.ts)
plot(turkey.price.ts)

#绘制自相关函数图
acf(turkey.price.ts)
#绘制偏相关函数图
pacf(turkey.price.ts)

#输出自相关系数
library(nutshell)
data(turkey.price.ts)
acf(turkey.price.ts,plot=FALSE)
#输出偏相关系数
pacf(turkey.price.ts,plot=FALSE)
#输出互相关系数
library(nutshell)
data(ham.price.ts)
ccf(turkey.price.ts, ham.price.ts, plot=FALSE)

#训练ar时间序列模型
library(nutshell)
data(turkey.price.ts)
turkey.price.ts.ar <- ar(turkey.price.ts)
turkey.price.ts.ar

#预测未来一年价格
predict(turkey.price.ts.ar,n.ahead=12)

#绘制预测结果
ts.plot(turkey.price.ts,predict(turkey.price.ts.ar,n.ahead=24)$pred,lty=c(1:2))
