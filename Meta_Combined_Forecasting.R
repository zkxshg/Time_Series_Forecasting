  
library(fpp2)
library(tabuSearch)

# ================ fitness function ===============
evaluate <- function(para){
  
  # =========== ARIMA ===========
  # ARIMA-Trend
  trend = c()
  # p
  p = para[1] * 4 + para[2] * 2 + para[3] * 1
  trend <- append(trend, p)
  # d
  d = para[4] * 2 + para[5] * 1
  trend <- append(trend, d)
  # q
  q = para[6] * 4 + para[7] * 2 + para[8] * 1
  trend <- append(trend, q)
  
  # ARIMA-Season
  season = c()
  # P
  P = para[9] * 2 + para[10] * 1
  season <- append(season, P)
  # D
  D = para[11] * 2 + para[12] * 1
  season <- append(season, D)
  # Q
  Q = para[13] * 2 + para[14] * 1
  season <- append(season, Q)
  
  print("ARIMA = ")
  print(trend)
  print(season)
  
  # Train s-ARIMA model
  fit1 <- tryCatch(
    { Arima(Train_data,order=trend,seasonal=season) },
    warning = function(w) { message('Waring @ ',trend) ; return(Arima(Train_data))},
    error = function(e) { message('Error @ ',trend) ; return(Arima(Train_data)) },
    finally = { message('next...') }
  )
  
  # =========== ETS ===========
  # state space and damped setting
  stateSpace = ""
  ifDamp = NULL
  
  # error
  if (para[15] == 0) {
    stateSpace <- paste(stateSpace, 'A', sep = "")
  } else {
    stateSpace <- paste(stateSpace, 'M', sep = "")
  }
  
  # trend
  if (para[16] == 0) { 
    
    if (para[17] == 0) {
      stateSpace <- paste(stateSpace, 'N', sep = "") 
    } else {
      stateSpace <- paste(stateSpace, 'A', sep = "") 
    }
    
  } else {
    
    if (para[17] == 0) {
      stateSpace <- paste(stateSpace, 'A', sep = "") 
      ifDamp = TRUE
    } else {
      stateSpace <- paste(stateSpace, 'Z', sep = "") 
    }
    
  }
  
  # season
  if (para[18] == 0) { 
    
    if (para[19] == 0) {
      stateSpace <- paste(stateSpace, 'N', sep = "") 
    } else {
      # stateSpace <- paste(stateSpace, 'A', sep = "") 
      stateSpace <- paste(stateSpace, 'Z', sep = "")
    }
    
  } else {
    
    if (para[19] == 0) {
      # stateSpace <- paste(stateSpace, 'M', sep = "") 
      stateSpace <- paste(stateSpace, 'Z', sep = "")
    } else {
      stateSpace <- paste(stateSpace, 'Z', sep = "") 
    }
    
  }
  
  # ====== deal with invalid setting ============
  if (stateSpace == "AAM") stateSpace = "AAN"
  if (stateSpace == "ANM") stateSpace = "ANN"
  if (stateSpace == "AZM") stateSpace = "AZN"
  
  print("ETS = ")
  print(stateSpace)
  print(ifDamp)
 
  fit2 = ets(Train_data, model=stateSpace, damped=ifDamp)
  
  aver_AIC = 0.5 * AIC(fit1) + 0.5 * AIC(fit2)
  
  # update the best parameter
  if (aver_AIC < mini_AIC) {
    mini_AIC <<- aver_AIC;
    best_trend <<- trend
    best_season <<- season
    best_stateSpace <<- stateSpace
    best_ifDamp <<- ifDamp
  }
      
  return(aver_AIC)
}

# ================== compare the RMSE =================================
RMSE_SEQ<-function(pre){
  RMSE1 <- c()
  for (i in 1:Test_size){
    RMSE1 <- append(RMSE1, accuracy(pre[1:i], Test_data[1:i])[2])
  }
  RMSE1 <- ts(RMSE1, start=start(Test_data), frequency=12)
  return(RMSE1)
}

# ================ Plot comparison with SARIMA / STLF / TBATS / ETS
ARIMA_plot<-function(trend, season, stateSpace, ifDamp){
  # Train
  fit1 <- auto.arima(Train_data)
  fit3 <- tbats(Train_data, biasadj=TRUE)
  fit4 <- ets(Train_data)
  
  fit51 <- Arima(Train_data,order=trend,seasonal=season)
  fit52 <- ets(Train_data, model=stateSpace, damped=ifDamp)
  
  
  # Forecasting
  pre1 <- forecast(fit1, h=Test_size)$mean
  pre2 <- stlf(Train_data, lambda=0, h=Test_size, biasadj=TRUE)$mean
  pre3 <- forecast(fit3, h=Test_size)$mean
  pre4 <- forecast(fit4, h=Test_size)$mean
  pre5 <- 0.5 * forecast(fit51, h=Test_size)$mean + 0.5 * forecast(fit52, h=Test_size)$mean
  
  # cal RMSE
  rm1 <- RMSE_SEQ(pre1)
  rm2 <- RMSE_SEQ(pre2)
  rm3 <- RMSE_SEQ(pre3)
  rm4<- RMSE_SEQ(pre4)
  rm5<- RMSE_SEQ(pre5) / 2
  
  # plot the error
  autoplot(rm1) +
    autolayer(rm1, series="RMSE of SARIMA") +
    autolayer(rm2, series="RMSE of STLF") +
    autolayer(rm3, series="RMSE of TBATS") +
    autolayer(rm4, series="RMSE of ETS") +
    autolayer(rm5, series="RMSE of PTS-CF") +
    ggtitle("Forecasting of data") +
    xlab("Year") + ylab("") +
    guides(colour=guide_legend(title="Forecasting")) +
    theme(plot.title = element_text(hjust = 0.5))
  
}

# ======== Initialize ===============
InitialConfig = c(1,0,1, 0,1, 1,0,1, 1,0, 0,1, 1,0, 0, 1,0, 1,0)

# ================== Data Split ========================
# timeseries = goog
# timeseries =  euretail
timeseries = sunspotarea

data <- timeseries
Test_size <- length(data) / 3
Train_size <- length(data) - Test_size
Test_data <- subset(data, start=Train_size+1)
Train_data <- subset(data, end=Train_size)

# =============== Record ====================
mini_AIC = 9999999
best_trend = c(0,0,0)
best_season = c(0,0,0)
best_stateSpace = "NNN"
best_ifDamp = NULL

# ============= Optimal parameters search policy ==================
res <- tabuSearch(size = 19, iters = 10, objFunc = evaluate, config = InitialConfig, listSize = 10, nRestarts = 1)

# ============= Update the optimal parameters ======================
trend = best_trend
season = best_season
stateSpace = best_stateSpace
ifDamp = best_ifDamp

# =========== Comparison =========================
ARIMA_plot(trend, season, stateSpace, ifDamp)
