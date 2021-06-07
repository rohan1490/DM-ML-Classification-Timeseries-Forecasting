library(fpp2)
library(tseries)
library(astsa)
data <- read.csv("D:/Study/Data Mining and Machine Learning/Project/Datasets/Weather/weather.csv",header = TRUE, sep = ",")
head(data)
nrow(data)

#Getting max temp
train <- data[1:11323,c(1,2)]
head(train)
tail(train)

test <- data[11324:nrow(data),c(1,2)]
head(test)
tail(test)

data <- data[,2]
train <- train[,2]
test <- test[,2]

str(test)

test <- data[(nrow(data) - 18627):nrow(data),c(1,2)]
str(data)
head(data)
tail(data)
summary(data)

data <- data[,2]

#Convert to timeseries
timeseries <- ts(data,frequency=365,start=c(1980,1,1),end =c(2020,12,31) )
train <- ts(data,frequency=365,start=c(1980,1,1),end =c(2010,12,31) )
test <- ts(data,frequency=365,start=c(2011,1,1),end=c(2020,12,31))


par(mfrow = c(1,1))
par(pty="m")
tsplot(timeseries, ylab="Dublin temperature", lwd=2, col=rgb(.9,  0, .7, .5))			  
tsplot(train, ylab="Dublin temperature", lwd=2, col=rgb(.9,  0, .7, .5))			  
tsplot(test, ylab="Dublin temperature", lwd=2, col=rgb(.9,  0, .7, .5))			  

#Decompose timeseries
plot(decompose(train, type = "additive"))


#Seasonal Plot
library(ggplot2)
ggseasonplot(timeseries, year.labels = TRUE, year.labels.left = TRUE)+ylab("Dublin weather")+ggtitle("Seasonal Plot")

#Mean of plots each month
ggsubseriesplot(timeseries)+ylab("Denmark - Supply of Electricity (Gigawatt-hour)")+ggtitle("Seasonal subseries Plot")

#ARIMA modelling

ndiffs(train)
nsdiffs(train)
adf.test(train)

diff <- diff(train,365)
mean(diff)
autoplot(diff)
ndiffs(diff)
nsdiffs(diff)

par(mfrow = c(1,2))
Acf(diff)
Pacf(diff)

adf.test(diff)

automodel <- auto.arima(train)
automodel
str(automodel)
checkresiduals(automodel)
qqnorm(automodel$residuals)
qqline(automodel$residuals)
Box.test(automodel$residuals, type="Ljung-Box")
summary(automodel)
mean(automodel$residuals)

fcast1 <- forecast(automodel, h=3297)
a1 <- accuracy(fcast1,test)
a1
fcast1
str(a1)

ar2 <-Arima(train, order = c(0,1,1), seasonal = list(order=c(0,1,0), period = 365))
ar2


fcast2 <- forecast(ar2, h=3297)
fcast2
a2 <- accuracy(fcast2,test)
a2

ar3 <-Arima(train, order = c(1,0,1), seasonal = list(order=c(0,1,0), period = 365))
ar3

fcast3 <- forecast(ar3, h=3297)
fcast3
a3 <- accuracy(fcast3,test)
a3

ar4 <-Arima(train, order = c(2,0,1), seasonal = list(order=c(0,1,0), period = 365))
ar4

fcast4 <- forecast(ar4, h=3297)
fcast4
a4 <- accuracy(fcast4,test)
a4
Box.test(ar4$residuals, type="Ljung-Box")


autoplot(timeseries, series = 'Original Timeseries') + 
  autolayer(fcast4$fitted, series = 'ARIMA (2,0,1)(0,1,0)[365]')+
  autolayer(test, series = 'test')+
  xlab("Year")+
  ylab("Temperature Dublin")+
  theme_minimal()+
  theme(legend.position = c(.75, .9))+
  scale_color_manual(values = c("turquoise4","chartreuse3","coral3"))


