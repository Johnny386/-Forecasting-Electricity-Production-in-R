library(fpp2)
setwd("C:/Users/jchreim/Desktop/Forecasting/Project")


##########################
# Electricity Production #
#########################

# read the data
data <- read.csv("Electric_Production.csv")
data$DATE <- as.Date(data$DATE, format = "%m/%d/%Y")
ts_data <- ts(data$IPG2211A2N, start = c(1985, 1), frequency = 12)

#Data Description
min(data$DATE)
max(data$DATE)


# Plot the data
plot(ts_data, col = "red")

# Seasonal plot
seasonplot(ts_data, year.labels = TRUE, year.labels.left = TRUE,
           main = "Electricity Consumption",
           ylab = "Electricity index", col = rainbow(20), pch = 19)

# Seasonal subseries plot
monthplot(ts_data,ylab="Electricity index",xlab="Month", type="l",
          main="Monthplot: Electricity Consumption")

tsdisplay(ts_data)


# Train & Test split
train <- window(tsclean(ts_data), end=c(2009,12)) #training
test <- window(ts_data, start=c(2010,01)) #test

length(train)/length(tsclean(ts_data))
length(test)/length(tsclean(ts_data))




# Transformation needed?
lambda <- BoxCox.lambda(train)
lambda
# Transformation is needed 
transformed_train <- BoxCox(train, lambda)
# Check the difference after transformation
hist(train, main="Before Transformation")
hist(transformed_train, main="After Transformation")
plot(transformed_train, main="After Transformation")
plot(train, main="Before Transformation")




# ETS Models
f1 <- ets(transformed_train, model = "AAA")
f2 <- ets(transformed_train, model = "MAA")
f3 <- ets(transformed_train, model = "ZZZ")

# Comparing the ETS models
summary (f1)
summary (f2)
summary (f3)





# Plot the best ETS model compared to test set
best_ETS_fc <- forecast(f3, h = length(test))

# Back-transform forecast and prediction intervals
best_ETS_fc$mean <- InvBoxCox(best_ETS_fc$mean, lambda)
best_ETS_fc$lower <- InvBoxCox(best_ETS_fc$lower, lambda)
best_ETS_fc$upper <- InvBoxCox(best_ETS_fc$upper, lambda)

# Visualize the forecast vs the Test set
autoplot(best_ETS_fc) +
  autolayer(test, series = "Test Data", color = "red") +
  ggtitle("Forecast vs. Test Set (Original Scale)") +
  ylab("Electricity Production") +
  xlab("Time") +
  theme_minimal() +
  xlim(c(start(best_ETS_fc$mean)[1], end(best_ETS_fc$mean)[1])) +  
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Forecast", "Test Data")) +  
  guides(color = guide_legend(title = "Legend")) 

# Get train and test metrics
accuracy(best_ETS_fc, test)
accuracy(f3)




# ARIMA models
acf(transformed_train)
pacf(transformed_train)

# Fit the models
arima1 <- Arima(transformed_train, order = c(1,1,1), seasonal = c(0,1,1))
arima2 <- Arima(transformed_train, order = c(2,1,0), seasonal = c(0,1,1))
arima3 <- auto.arima(transformed_train)

# Get summary of the models
summary(arima1)
summary(arima2)
summary(arima3)


# Fit on the test set
farima2 <- forecast(arima2, h=length(test))

# Back transformation
best_ARIMA<-InvBoxCox(farima2$mean, lambda)


# Plot the best ARIMA vs test set
autoplot(best_ARIMA) +
  autolayer(test, series = "Test Data", color = "red") +
  ggtitle("Forecast vs. Test Set (Original Scale)") +
  ylab("Electricity Production") +
  xlab("Time") +
  theme_minimal() +
  xlim(c(start(farima2$mean)[1], end(farima2$mean)[1])) +  
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Forecast", "Test Data")) +  
  guides(color = guide_legend(title = "Legend"))

accuracy(best_ARIMA, test)
accuracy(arima2)


# Auto ARIMA Metrics
# Fit on the test set
farima3 <- forecast(arima3, h=length(test))

# Back transformation
best_ARIMA<-InvBoxCox(farima3$mean, lambda)

accuracy(best_ARIMA, test)
accuracy(arima3)




# Comparing ETS & ARIMA
# Check residuals
checkresiduals(best_ETS_fc)
checkresiduals(farima3)

#Ljung Box test
Box.test(residuals(best_ETS_fc), lag = 10, type = "Ljung-Box", fitdf = 3)
Box.test(residuals(farima3), lag = 10, type = "Ljung-Box", fitdf = 3)




# Forecast with the best model
# Refit the ARIMA model to the full dataset
final_arima <- Arima(ts_data, order = c(2,1,0), seasonal = c(0,1,1))
fcast_final <- forecast(final_arima, h = 36)
plot(fcast_final)




# TBATS & NNAR

# TBATS model
fit_tbats <- tbats(transformed_train)

# NNAR model (Neural Network AutoRegressive)
fit_nnar <- nnetar(transformed_train)
# Forecast for same horizon as test set
f_tbats <- forecast(fit_tbats, h = length(test))
f_nnar <- forecast(fit_nnar, h = length(test))


# Back transformation

#TBATS

f_tbats$mean <- InvBoxCox(f_tbats$mean, lambda)
f_tbats$lower <- InvBoxCox(f_tbats$lower, lambda)
f_tbats$upper <- InvBoxCox(f_tbats$upper, lambda)

#NNAR
f_nnar$mean <- InvBoxCox(f_nnar$mean, lambda)
f_nnar$lower <- InvBoxCox(f_nnar$lower, lambda)
f_nnar$upper <- InvBoxCox(f_nnar$upper, lambda)

# Compare accuracy with your best ARIMA model (e.g., f_best_arima)
accuracy_arima <- accuracy(farima3, test)
accuracy_tbats <- accuracy(f_tbats, test)
accuracy_nnar <- accuracy(f_nnar, test)

# View results
print(accuracy_tbats)
print(accuracy_nnar)
