# !可用模型函数：1-均值法:meanF(p,d);2-季节朴素法:snaiveF(p,d);3-时间序列分解:stlDeco(p);
# +4-ETS指数平滑法:etsF(p,d);5-季节ARIMA法:sarimaF(p,d);6-多季节STL分解预测:mstlF(p,d);
# +7-多季节动态谐波回归:mdhArimaF(p,d);8-TBATS模型:tbatsF(p,d);9-神经网络自回归:nnarF(p,d);
# +10-组合预测模型:combineF(p,d).
# ! 模型7-mdhArimaF/8-tbatsF/9-nnarF/10-combineF 的计算时间较长.
# ！10-组合预测模型无法计算置信区间.

# 读入需要的包
library(fpp2)
library(urca)

# 1-均值法(数据路径/path，预测天数/duration)
meanF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cols_name <- colnames(cq_dmj)
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	cq_dmjfit1 <- meanf(cq_dmj_Ts,h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_meanf_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================

# 2-季节性朴素预测方法(数据路径/path，预测天数/duration)
snaiveF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	cq_dmjfit1 <- snaive(cq_dmj_Ts,h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_snaive_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}
# =======================================================================================

# 3-时间序列STL分解(数据路径/path)
stlDeco<-function(path){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- stl(cq_dmj_Ts, t.window=13, s.window="periodic", robust=TRUE)
	cq_df <- as.data.frame(fit$time.series)
	row.names(cq_df) <- cq_dmj[[1]]
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_STL_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================
# 4-ETS指数平滑预测法(数据路径/path，预测天数/duration)
etsF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- ets(cq_dmj_Ts)
	cq_dmjfit1 <- forecast(fit, h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_ETS_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================
# 5-季节性ARIMA预测法(数据路径/path，预测天数/duration)
sarimaF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- auto.arima(cq_dmj_Ts, stepwise=FALSE, approximation=FALSE)
	cq_dmjfit1 <- forecast(fit, h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_sARIMA_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}


# =======================================================================================
# 6-多季节STL分解预测(数据路径/path，预测天数/duration)
mstlF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	cq_dmjfit1 <- stlf(cq_dmj_Ts, h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_mSTL_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}


# =======================================================================================
# 7-多季节动态谐波回归预测(数据路径/path，预测天数/duration)
mdhArimaF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- auto.arima(cq_dmj_Ts, seasonal=FALSE, lambda=0,
         xreg=fourier(cq_dmj_Ts, K=10))
	cq_dmjfit1 <- forecast(fit, h=duration,xreg=fourier(cq_dmj_Ts,K=10, h=duration))
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_mdhARIMA_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================
# 8-TBATS模型(数据路径/path，预测天数/duration)
tbatsF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- tbats(cq_dmj_Ts)
	cq_dmjfit1 <- forecast(fit, h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_TBATS_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================
# 9-神经网络自回归模型(数据路径/path，预测天数/duration)
nnarF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 预测并处理预测结果
	fit <- nnetar(cq_dmj_Ts, lambda=0)
	cq_dmjfit1 <- forecast(fit, PI=TRUE, h=duration)
	cq_df <- as.data.frame(cq_dmjfit1)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_NNAR_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}

# =======================================================================================
# 10-组合预测模型(数据路径/path，预测天数/duration)
combineF<-function(path, duration){
	# 读入并处理数据
	cq_dmj <- read.csv(file=path, header=TRUE, sep=",")
    cq_dmj[2][cq_dmj[2]==0]<-NA
	cq_dmj[2] <- na.interp(cq_dmj[2])
	cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)
	# 五种子模型：mdhArimaF、ARIMA、STLF、NNAR、TBATS
	MDH <- forecast(auto.arima(cq_dmj_Ts, seasonal=FALSE, lambda=0,
         xreg=fourier(cq_dmj_Ts, K=10)), h=duration, xreg=fourier(cq_dmj_Ts,K=10, h=duration))
	ARIMA <- forecast(auto.arima(cq_dmj_Ts, lambda=0, biasadj=TRUE, stepwise=FALSE,
	approximation=FALSE), h=duration)
	STL <- stlf(cq_dmj_Ts, lambda=0, h=duration, biasadj=TRUE)
	NNAR <- forecast(nnetar(cq_dmj_Ts), h=duration)
	TBATS <- forecast(tbats(cq_dmj_Ts, biasadj=TRUE), h=duration)
	# 组合五种的均值作为最后结果
	Combination <- (MDH[["mean"]] + ARIMA[["mean"]] +
		STL[["mean"]] + NNAR[["mean"]] + TBATS[["mean"]])/5
	# 预测并处理预测结果	
	cq_df <- as.data.frame(Combination)
	foreDate <- ''
	LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
	for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
	row.names(cq_df) <- foreDate
	# 保存并传出路径
	Result_path <- paste(substr(path,1,nchar(path)-4),"_combined_Result.csv")
	write.csv(cq_df, file = Result_path)
	return(Result_path)
}
# =======================================================================================
