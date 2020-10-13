# 读入包
library(fpp2)
# 读入数据
cq_dmj <- read.csv(file="合计.csv", header=TRUE, sep=",")

# 线性插值处理缺失值
cq_dmj$Total[cq_dmj$Total==0]<-NA
cq_dmj$Total <- na.interp(cq_dmj$Total)

# 处理离群值
# cq_dmj$Total <- tsclean(cq_dmj_Ts)

# 转换为时间序列对象
cq_dmj_Ts <- ts(cq_dmj$Total, start=c(2017, 1), frequency=52)

# 读入利群销售数据
cq_dmj_lqx <- read.csv(file="利群新.csv", header=TRUE, sep=",")
# 线性插值处理缺失值
cq_dmj_lqx$LiqunNew[cq_dmj_lqx$LiqunNew==0]<-NA
cq_dmj_lqx$LiqunNew <- na.interp(cq_dmj_lqx$LiqunNew)
# 转换为时间序列对象
cq_dmj_LiqunNew_Ts <- ts(cq_dmj_lqx$LiqunNew, start=c(2017, 1), frequency=52)

# =====================================================
# 销售数据可视化

# 时间序列折线图
autoplot(cq_dmj_Ts) +
  ggtitle("城区-小明") +
  xlab("年份") +
  ylab("件")+
  theme(plot.title = element_text(hjust = 0.5))
  
# 时间序列季节图
ggseasonplot(cq_dmj_Ts, year.labels=TRUE, year.labels.left=TRUE) +
  xlab("每周")+
  ylab("件") +
  ggtitle("季节图：小明销量")+
  theme(plot.title = element_text(hjust = 0.5))

# 时间序列极坐标图
ggseasonplot(cq_dmj_Ts, polar=TRUE) +
  xlab("每周")+
  ylab("件") +
  ggtitle("极坐标季节图：小明销量")+
  theme(plot.title = element_text(hjust = 0.5))

# 时间序列季节子图
ggsubseriesplot(cq_dmj_Ts) +
  xlab("每周")+
  ylab("件") +
  ggtitle("子序列季节图：小明销量")+
  theme(plot.title = element_text(hjust = 0.5))

# 绘制散点图
qplot(cq_dmj_Ts,cq_dmj_LiqunNew_Ts) +
  ylab("总销量") + xlab("利群新销量")+
  theme(plot.title = element_text(hjust = 0.5))
# 绘制散点图矩阵
as.data.frame(cq_dmj$Total,cq_dmj_lqx$LiqunNew)%>% GGally::ggpairs()

# 绘制滞后图
gglagplot(cq_dmj_Ts)

# 绘制自相关图
ggAcf(cq_dmj_Ts, lag=48) +
  ggtitle('自相关图') 
  
# =====================================================
# 简单预测方法

# 均值预测 + 随机游动预测(Naive) + 季节性Naïve方法 + 漂移法
autoplot(cq_dmj_Ts) +
  autolayer(meanf(cq_dmj_Ts, h=26)$mean, series="均值") +
  autolayer(naive(cq_dmj_Ts, h=26)$mean, series="Naïve") +
  autolayer(snaive(cq_dmj_Ts, h=26)$mean, series="季节性 naïve") +
  autolayer(rwf(cq_dmj_Ts, drift=TRUE, h=26)$mean, series="漂移") +
  ggtitle("小明销量的预测") +
  xlab("年份") + ylab("件") +
  guides(colour=guide_legend(title="预测")) +
  theme(plot.title = element_text(hjust = 0.5))

# =====================================================
# 数据变换后预测
fc <- rwf(cq_dmj_Ts, drift=TRUE, lambda=0, h=26, level=80)
fc2 <- rwf(cq_dmj_Ts, drift=TRUE, lambda=0, h=26, level=80, biasadj=TRUE)
autoplot(cq_dmj_Ts) +
  autolayer(fc, series="简单逆变换") +
  autolayer(fc2$mean, series="偏差调整") +
  guides(colour=guide_legend(title="预测")) +
  theme(plot.title = element_text(hjust = 0.5))
  
# =====================================================
# 预测结果残差诊断

# naive结果残差诊断
res <- residuals(naive(cq_dmj_Ts))

# 残差折线图
autoplot(res) + xlab("周") + ylab("") +
  ggtitle("Naïve方法的残差")+
  theme(plot.title = element_text(hjust = 0.5))
  
# 残差直方图
gghistogram(res) + ggtitle("残差直方图")+
  theme(plot.title = element_text(hjust = 0.5))
  
# Box-Pierce检验
Box.test(res, lag=10, fitdf=0)
# Ljung-Box检验
Box.test(res,lag=10, fitdf=0, type="Lj")
# 殘差復合檢驗
checkresiduals(naive(cq_dmj_Ts))

# =====================================================
# 准确率计算

# 取子集
# 窗口抓取
window(cq_dmj_Ts, start=2018)
# 子集函數
subset(cq_dmj_Ts, start=length(cq_dmj_Ts)-4*5)
head(cq_dmj_Ts, 4*5)
tail(cq_dmj_Ts, 4*5)

# 将前两年的销售量划分为测试集
cq_dmj_Ts2 <- window(cq_dmj_Ts,end=c(2018,52))
# 简单预测方法
cq_dmjfit1 <- meanf(cq_dmj_Ts2,h=27)
cq_dmjfit2 <- rwf(cq_dmj_Ts2,h=27)
cq_dmjfit3 <- snaive(cq_dmj_Ts2,h=27)
autoplot(cq_dmj_Ts2) +
  autolayer(cq_dmjfit1$mean, series="均值") +
  autolayer(cq_dmjfit2$mean, series="Naïve") +
  autolayer(cq_dmjfit3$mean, series="季节性naïve") +
  xlab("年份") + ylab("件") +
  ggtitle("小明销量的预测") +
  guides(colour=guide_legend(title="预测")) +
  theme(plot.title = element_text(hjust = 0.5))
  
# 计算预测模型拟合误差
cq_dmj_Ts3 <- window(cq_dmj_Ts, start=2019)
accuracy(cq_dmjfit1, cq_dmj_Ts3)
accuracy(cq_dmjfit2, cq_dmj_Ts3)
accuracy(cq_dmjfit3, cq_dmj_Ts3)

# 交叉验证法
e <- tsCV(cq_dmj_Ts2, rwf, drift=TRUE, h=27)
sqrt(mean(e^2, na.rm=TRUE))
sqrt(mean(residuals(rwf(cq_dmj_Ts2, drift=TRUE))^2, na.rm=TRUE))

# =====================================================
# 绘制置信区间

# 直接绘制置信区间
autoplot(naive(cq_dmj_Ts2,h=27))

# Bootstrap法计算置信区间
naive(cq_dmj_Ts2,h=27,bootstrap=TRUE)

# =====================================================
# 绘制置信区间

# 乘法分解
cq_dmj_Ts %>% decompose(type="multiplicative") %>% 
  autoplot() + xlab("年份") +
  ggtitle("小明销量的经典乘法分解")

# STL分解
fit <- cq_dmj_Ts %>% stl(t.window=13, s.window="periodic", robust=TRUE) 
autoplot(fit)+
  ggtitle("小明销量的STL分解")+
  theme(plot.title = element_text(hjust = 0.5))
  
# 提取分解后数据
autoplot(cq_dmj_Ts, series="Data") +
  autolayer(trendcycle(fit), series="Trend") +
  autolayer(seasadj(fit), series="Seasonally Adjusted") +
  xlab("年份") + ylab("销量") +
  ggtitle("小明的销量") +
  scale_colour_manual(values=c("gray","blue","red"),
                     breaks=c("数据","季节调整","取数"))+
  theme(plot.title = element_text(hjust = 0.5))

# STl+naive预测
fit <- stl(cq_dmj_Ts, t.window=13, s.window="periodic", robust=TRUE)
fit %>% seasadj() %>% naive() %>% autoplot() + ylab("预测销量") +
  ggtitle("季节调整数据的朴素预测")+
  theme(plot.title = element_text(hjust = 0.5)) 
  
# 再季节化
fit %>% forecast(method="naive", h=27) %>% autoplot() + ylab("预测销量")+
  theme(plot.title = element_text(hjust = 0.5))
  
# STLF
fcast <- stlf(cq_dmj_Ts, method='naive', h=27)

# =====================================================
# 指数平滑模型

# 简单指数平滑SES
# 参数估计
fc <- ses(cq_dmj_Ts)
# 在1-12时期的向前一步训练误差的精度
round(accuracy(fc),2)
# 绘图
autoplot(fc) +
  autolayer(fitted(fc), series="拟合值") +
  ylab("销量") + xlab("年份") +
  ggtitle('简单指数平滑预测') + 
  theme(plot.title = element_text(hjust = 0.5))
  
# Holt的线性趋势法和衰减趋势法 
fc <- holt(cq_dmj_Ts2, h=27)
fc2 <- holt(cq_dmj_Ts2, damped=TRUE, phi = 0.9, h=27)
# 绘图
autoplot(cq_dmj_Ts) +
  autolayer(fc$mean, series="Holt的线性趋势法") +
  autolayer(fc2$mean, series="Holt的衰减趋势法 ") +
  ggtitle("Holt方法预测") +
  xlab("年份") + ylab("销量（件）") +
  guides(colour=guide_legend(title="预测"))+
  theme(plot.title = element_text(hjust = 0.5))
  
# ETS模型
fit <- ets(cq_dmj_Ts2)
summary(fit)
# 绘制结果
autoplot(fit)
# 残差分析
cbind('Residuals' = residuals(fit),
      'Forecast errors' = residuals(fit, type='response')) %>%
  autoplot(facet=TRUE) + xlab("年份") + ylab("")+
  theme(plot.title = element_text(hjust = 0.5))

# ETS预测
fit %>% forecast(h=27) %>%
  autoplot() +
  xlab("时间") +
  ylab("小明的销量（件）")+
  ggtitle('基于ETS模型的预测') +
  theme(plot.title = element_text(hjust = 0.5))  
  
# =====================================================
# ARIMA模型

# 差分检验
Box.test(diff(cq_dmj_Ts),lag=10,type="Ljung-Box")
# 季节差分
cbind("销售量 (件)" = cq_dmj_Ts,
      "每周销量对数" = log(cq_dmj_Ts),
      "每年销量变化对数" = diff(log(cq_dmj_Ts),52)) %>%
  autoplot(facets=TRUE) +
    xlab("年份") + ylab("") +
    ggtitle("小明销量")+
  theme(plot.title = element_text(hjust = 0.5))
# 单位根检验
library(urca)
cq_dmj_Ts %>% ur.kpss() %>% summary() 
cq_dmj_Ts %>% diff() %>% ur.kpss() %>% summary()
# 确定一次差分次数
ndiffs(cq_dmj_Ts)
# 确定季节差分次数
usmelec %>% log() %>% nsdiffs()
usmelec %>% log() %>% diff(lag=52) %>% ndiffs()

# 非季节性ARIMA
fit <- auto.arima(cq_dmj_Ts2, seasonal = FALSE)
fit %>% forecast(h=27) %>% autoplot(include=80)
# ACF
ggAcf(cq_dmj_Ts)
# PACF
ggPacf(cq_dmj_Ts)
# 展示时间序列及相关图
ggtsdisplay(cq_dmj_Ts)
# 精准非季节AutoArima
(fit2 <- auto.arima(cq_dmj_Ts2,seasonal=FALSE,
  stepwise=FALSE, approximation=FALSE))

# 季节调整
cq_dmj_Ts %>% stl(s.window='periodic') %>% seasadj() -> eeadj
autoplot(eeadj)
# 一阶差分及检查
eeadj %>% diff() %>% ggtsdisplay(main="")
# ARIMA(0,0,1)
fit <- Arima(eeadj,order=c(0,0,1))
summary(fit)
# 检查残差
checkresiduals(fit)
# 预测
autoplot(forecast(fit, h=27))
# 绘制特征根
autoplot(fit) 

# 季节性ARIMA模型
(fit4 <- auto.arima(cq_dmj_Ts, stepwise=FALSE, approximation=FALSE))
fit4_fore <- fit4 %>% forecast(h=27)
fit4 %>% forecast(h=27)%>% autoplot(include=80)
# 绘制预测结果
autoplot(cq_dmj_Ts) +
  autolayer(fit4_fore, series="ARIMA预测结果") +
  xlab("年份") + ylab("件") +
  ggtitle("小明销量的预测") +
  guides(colour=guide_legend(title="预测")) +
  theme(plot.title = element_text(hjust = 0.5))
  
# 交叉验证+ETS/ARIMA
fets <- function(x,h) {
  forecast(ets(x),h = h)
}
farima <- function(x,h) {
  forecast(auto.arima(x),h=h)
}
# 计算指数平滑法的交叉验证误差,设为e1
e1 <- tsCV(cq_dmj_Ts,fets,h=1)
# 计算 ARIMA 的交叉验证误差,设为e2
e2 <- tsCV(cq_dmj_Ts,farima,h=1)
# 计算各个模型的均方误差
mean(e1^2,na.rm=TRUE)
mean(e2^2,na.rm=TRUE)

# =====================================================
# 动态ARIMA模型

# ARIMA 误差回归
cq_dmj_LiqunNew2 <- window(cq_dmj_LiqunNew_Ts,end=c(2018,52))
# ARIMA 误差回归，以利群新销量为自变量
(fit <- auto.arima(cq_dmj_Ts, xreg=cq_dmj_LiqunNew_Ts)
# Forecasting
fcast <- forecast(fit, h=27, xreg=rep(mean(cq_dmj_LiqunNew_Ts),27))
autoplot(fcast) + xlab("年份") +
  ylab("销量预测")+
  theme(plot.title = element_text(hjust = 0.5))
  
# 动态谐波回归
# 测试不同傅里叶级数的效果
plots <- list()
for (i in seq(6)) {
  fit <- auto.arima(cq_dmj_Ts, xreg = fourier(cq_dmj_Ts, K = i), 
    seasonal = FALSE, lambda = 0)
  plots[[i]] <- autoplot(forecast(fit, h=27, xreg=fourier(cq_dmj_Ts, K=i, h=27))) +
    xlab(paste("K=",i,"   AICC=",round(fit$aicc,2))) 
}
# 绘制不同级数图像
gridExtra::grid.arrange(plots[[1]],plots[[2]],plots[[3]],
                        plots[[4]],plots[[5]],plots[[6]], nrow=3)
						
# =====================================================
# 多季节性数据

# 多季节性时间序列STL分解
cq_dmj_Ts %>% mstl() %>% 
  autoplot() + xlab("周") 
  
# mstl预测
cq_dmj_Ts %>%  stlf(h=27) %>% 
  autoplot() + xlab("周") 

# 多季节动态谐波回归
# 训练多季节动态谐波回归模型
fit <- auto.arima(cq_dmj_Ts, seasonal=FALSE, lambda=0,
         xreg=fourier(cq_dmj_Ts, K=10))
# 预测未来销量
fit %>%
  forecast(h=27, xreg=fourier(cq_dmj_Ts,K=10, h=27)) %>%
  autoplot() +
   ylab("来电次数") + xlab("周")  
   
# TBATS模型
# 训练模型
tbats(cq_dmj_Ts2) -> fit2
# 进行预测
fc2 <- forecast(fit2, h=27)
# autoplot
autoplot(fc2) +
  ylab("销量") + xlab("周") 
  
# =====================================================
# VARS
# 查看滞后长度
VARselect(cbind(cq_dmj_Ts, cq_dmj_LiqunNew_Ts), lag.max=8, type="const")[["selection"]]

# 拟合并检验VAR(2)
var1 <- VAR(cbind(cq_dmj_Ts, cq_dmj_LiqunNew_Ts), p=1, type="const")
serial.test(var1, lags.pt=10, type="PT.asymptotic")
# 拟合并检验VAR(2)
var2 <- VAR(cbind(cq_dmj_Ts, cq_dmj_LiqunNew_Ts), p=2, type="const")
serial.test(var2, lags.pt=10, type="PT.asymptotic")

# 使用VAR(1)进行预测
forecast(var1) %>%
  autoplot() + xlab("年")
# 使用VAR(2)进行预测
forecast(var2) %>%
  autoplot() + xlab("年")

# =====================================================
# 神经网络自回归
fit <- nnetar(cq_dmj_Ts, lambda=0)
autoplot(forecast(fit,PI=TRUE,h=27)) +
  xlab("年份") + ylab("销量（件）")
  
# Bootstrap生成预测区间
sim <- ts(matrix(0, nrow=2L, ncol=10L),
          start=end(cq_dmj_Ts)[1L])
for(i in seq(9))
  sim[,i] <- simulate(fit, nsim=2L)
autoplot(cq_dmj_Ts) + autolayer(sim) +
  xlab("年份") + ylab("销量（件）")

# =====================================================
# baggedETS

# 普通ETS预测结果
etsfc <- cq_dmj_Ts2 %>% ets() %>% forecast(h=27)
# baggedETS预测结果
baggedfc <- cq_dmj_Ts2 %>% baggedETS() %>% forecast(h=27)

autoplot(cq_dmj_Ts) +
  autolayer(baggedfc$mean, series="BaggedETS") +
  autolayer(etsfc$mean, series="ETS") +
  guides(colour=guide_legend(title="Forecasts")) +
  xlab("年份") + ylab("销量（件）")
  
# =====================================================
# 周数据预测

# STLF预测周数据
cq_dmj_Ts %>% stlf() %>% autoplot() + 
  xlab("年份")
  
# 动态谐波回归预测周数据
# 找出最优傅里叶项个数
bestfit <- list(aicc=Inf)
for(K in seq(25)) {
  fit <- auto.arima(cq_dmj_Ts, xreg=fourier(cq_dmj_Ts, K=K),
    seasonal=FALSE)
  if(fit[["aicc"]] < bestfit[["aicc"]]) {
    bestfit <- fit
    bestK <- K
  }
}
# 建立模型并预测
fc <- forecast(bestfit, xreg=fourier(cq_dmj_Ts, K=bestK, h=104))
autoplot(fc) +
  ylab("销量（件）") + xlab("年份")
  
# =====================================================
# 组合预测

# 提取训练集
train <- window(cq_dmj_Ts, end=c(2019,2))
# 待预测规模
h <- length(cq_dmj_Ts) - length(train)
# 五种子模型：ETS、ARIMA、STLF、NNAR、TBATS
ETS <- forecast(ets(train), h=h)
ARIMA <- forecast(auto.arima(train, lambda=0, biasadj=TRUE), h=h)
STL <- stlf(train, lambda=0, h=h, biasadj=TRUE)
NNAR <- forecast(nnetar(train), h=h)
TBATS <- forecast(tbats(train, biasadj=TRUE), h=h)
# 组合五种的均值作为最后结果
Combination <- (ETS[["mean"]] + ARIMA[["mean"]] +
  STL[["mean"]] + NNAR[["mean"]] + TBATS[["mean"]])/5
  
# 绘图比较组合模型与子模型效果
autoplot(cq_dmj_Ts) +
  autolayer(ETS, series="ETS", PI=FALSE) +
  autolayer(ARIMA, series="ARIMA", PI=FALSE) +
  autolayer(STL, series="STL", PI=FALSE) +
  autolayer(NNAR, series="NNAR", PI=FALSE) +
  autolayer(TBATS, series="TBATS", PI=FALSE) +
  autolayer(Combination, series="Combination") +
  xlab("年份") + ylab("销量（件）") +
  ggtitle("小明销量的预测") 

# 输出各模型准确性
c(ETS=accuracy(ETS, cq_dmj_Ts)["Test set","RMSE"],
  ARIMA=accuracy(ARIMA, cq_dmj_Ts)["Test set","RMSE"],
  `STL-ETS`=accuracy(STL, cq_dmj_Ts)["Test set","RMSE"],
  NNAR=accuracy(NNAR, cq_dmj_Ts)["Test set","RMSE"],
  TBATS=accuracy(TBATS, cq_dmj_Ts)["Test set","RMSE"],
  Combination=accuracy(Combination, cq_dmj_Ts)["Test set","RMSE"])
