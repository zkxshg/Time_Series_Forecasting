library(vars)
library(fpp2)

VARF<-function(sale, poli, duration){

    # 读入并处理数据
    cq_dmj <- read.csv(file=sale, header=TRUE, sep=",")

    cq_policy <- read.csv(file=poli, header=TRUE, sep=",")

    # 保证策略量和销售量相同长度
    mm = merge(cq_dmj, cq_policy, by.x = colnames(cq_dmj)[1], by.y = colnames(cq_policy)[1])
    # 切分合并结果
    cq_dmj = data.frame(mm[,1], mm[,2])
    cq_policy = data.frame(mm[,1], mm[,3])

    # 首先处理策略量
    cq_policy[2][cq_policy[2]==0]<-NA
    cq_policy[2] <- na.interp(cq_policy[2])
    # 处理销售量
    cq_dmj[2][cq_dmj[2]==0]<-NA
    cq_dmj[2] <- na.interp(cq_dmj[2])
    # 转为 ts
    cq_pol_Ts <- ts(cq_policy[[2]],start=c(cq_policy[1,1]%/%100, cq_policy[1,1]%%100),frequency=52)
    cq_dmj_Ts <- ts(cq_dmj[[2]],start=c(cq_dmj[1,1]%/%100, cq_dmj[1,1]%%100),frequency=52)

    # 预测并处理预测结果
    delay = VARselect(data.frame(cq_dmj_Ts, cq_pol_Ts), lag.max=8, type="const")[["selection"]]
    fit <- VAR(as.ts(data.frame(cq_dmj_Ts, cq_pol_Ts)), p=delay[1], type="const")
    cq_dmjfit1 <- forecast(fit, h=duration)

    cq_df <- as.data.frame(cq_dmjfit1$forecast$cq_dmj_Ts)
    foreDate <- ''
    LastDate <- as.integer(cq_dmj[[1]][length(cq_dmj[[1]])]) 
    for (i in 1:duration) {foreDate[i] <- ((LastDate)%/%100 + (LastDate%%100 + i)%/%52)*100 + (LastDate%%100 + i)%%52}
    row.names(cq_df) <- foreDate

    # 保存并传出路径
    Result_path <- paste(substr(sale,1,nchar(sale)-4),"_策略量_Result.csv")
    write.csv(cq_df, file = Result_path)
    return(Result_path)
}
