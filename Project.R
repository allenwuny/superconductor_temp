rm(list = ls())
cat("\014")

library(readr)
library(glmnet)
library(tidyverse)
library(randomForest)
library(reshape2)
library(gridExtra)

sc = read_csv("train.csv")

n = dim(sc)[1]
p = dim(sc)[2] - 1

set.seed(1)


loop.size = 100

# Create an empty dataframe to store R squared values
rsq.df = data.frame(matrix(0, nrow = loop.size * 4, ncol = 4))
names(rsq.df) = c("loop", "model", "train.rsq", "test.rsq")

################################################################
##            Training Various Models on Partial Dataset      ##
################################################################


for (i in c(1:loop.size)) {
  cat("Running loop", i, "\n")
  
  # Randomly shuffle the dataset and separate into training and testing
  shuffle = sample(n)
  train = shuffle[1:floor(n*0.8)]
  
  y.train = unlist(subset(sc[train,], select = c(critical_temp)))
  x.train = as.matrix(subset(sc[train,], select = -c(critical_temp)))
  
  y.test = unlist(subset(sc[-train,], select = c(critical_temp)))
  x.test = as.matrix(subset(sc[-train,], select = -c(critical_temp)))
  
  # Cross validation using ridge 
  cv.ridge.start = Sys.time()  
  cv.ridge.fit = cv.glmnet(x = x.train, 
                           y = y.train, 
                           alpha = 0)
  cv.ridge.time = Sys.time() - cv.ridge.start
  

  # Cross validation using lasso  
  cv.lasso.start = Sys.time() 
  cv.lasso.fit = cv.glmnet(x = x.train, 
                           y = y.train, 
                           alpha = 1)
  cv.lasso.time = Sys.time() - cv.lasso.start
  
  # Cross validation using elastic net
  cv.elnet.start = Sys.time()  
  cv.elnet.fit = cv.glmnet(x = x.train, 
                           y = y.train, 
                           alpha = 0.5)
  cv.elnet.time = Sys.time() - cv.elnet.start
  
  # Identify the minimum lambda from cross validation
  ridge.lam = cv.ridge.fit$lambda.min
  lasso.lam = cv.lasso.fit$lambda.min
  elnet.lam = cv.elnet.fit$lambda.min
  
  
  # Train a ridge regression using optimal lambda using training dataset
  ridge.fit = glmnet(x = x.train, 
                     y = y.train, 
                     alpha = 0, 
                     lambda = ridge.lam)
  
  # Train a lasso regression using optimal lambda using training dataset
  lasso.fit = glmnet(x = x.train, 
                     y = y.train, 
                     alpha = 1, 
                     lambda = lasso.lam)
  
  # Train a elastic net regression using optimal lambda using training dataset
  elnet.fit = glmnet(x = x.train, 
                     y = y.train, 
                     alpha = 0.5, 
                     lambda = elnet.lam)
  
  # Train a random forest using training dataset
  rf.fit = randomForest(x = x.train,
                        y = y.train, 
                        mtry=p/3, 
                        importance=TRUE)
  
  # Calcaulte total sum of squares
  tss = mean((y.test - mean(y.test))^2)
  
  # Calculate R squared value for ridge using training data
  ridge.train.y.hat = predict(ridge.fit, x.train)
  ridge.train.rss = mean((y.train - ridge.train.y.hat)^2)
  ridge.train.rsq = 1 - (ridge.train.rss/tss)
  
  # Calculate R squared value for lasso using training data
  lasso.train.y.hat = predict(lasso.fit, x.train)
  lasso.train.rss = mean((y.train - lasso.train.y.hat)^2)
  lasso.train.rsq = 1 - (lasso.train.rss/tss)
  
  # Calculate R squared value for elastic net using training data
  elnet.train.y.hat = predict(elnet.fit, x.train)
  elnet.train.rss = mean((y.train - elnet.train.y.hat)^2)
  elnet.train.rsq = 1 - (elnet.train.rss/tss)
  
  # Calculate R squared value for random forest using training data
  rf.train.y.hat = predict(rf.fit, x.train)
  rf.train.rss = mean((y.train - rf.train.y.hat)^2)
  rf.train.rsq = 1 - (rf.train.rss/tss)
  
  # Calculate R squared value for ridge using testing data
  ridge.test.y.hat = predict(ridge.fit, x.test)
  ridge.test.rss = mean((y.test - ridge.test.y.hat)^2)
  ridge.test.rsq = 1 - (ridge.test.rss/tss)
  
  # Calculate R squared value for lasso using testing data
  lasso.test.y.hat = predict(lasso.fit, x.test)
  lasso.test.rss = mean((y.test - lasso.test.y.hat)^2)
  lasso.test.rsq = 1 - (lasso.test.rss/tss)
  
  # Calculate R squared value for elastic net using testing data
  elnet.test.y.hat = predict(elnet.fit, x.test)
  elnet.test.rss = mean((y.test - elnet.test.y.hat)^2)
  elnet.test.rsq = 1 - (elnet.test.rss/tss)
  
  # Calculate R squared value for random forest using testing data
  rf.test.y.hat = predict(rf.fit, x.test)
  rf.test.rss = mean((y.test - rf.test.y.hat)^2)
  rf.test.rsq = 1 - (rf.test.rss/tss)

  # Update the R squared dataframe from the different models
  rsq.df[(i*4-3):(i*4),] = rbind(c(i, "Ridge", ridge.train.rsq, ridge.test.rsq), 
                                 c(i, "Lasso", lasso.train.rsq, lasso.test.rsq), 
                                 c(i, "Elastic Net", elnet.train.rsq, elnet.test.rsq),
                                 c(i, "Random Forest", rf.train.rsq, rf.test.rsq))
}


################################################################
##            Residual Boxplots                               ##
################################################################


# Calculate the residuals between y and y-bar for from training and testing data
resid.train = data.frame(rbind(cbind("Ridge", y.train - ridge.train.y.hat),
                               cbind("Lasso", y.train - lasso.train.y.hat),
                               cbind("Elastic Net", y.train - elnet.train.y.hat),
                               cbind("Random Forest", y.train - rf.train.y.hat)))

resid.test = data.frame(rbind(cbind("Ridge", y.test - ridge.test.y.hat),
                              cbind("Lasso", y.test - lasso.test.y.hat),
                              cbind("Elastic Net", y.test - elnet.test.y.hat),
                              cbind("Random Forest", y.test - rf.test.y.hat)))


# Update the columns names in the above dataframes
names(resid.train) = c("model", "residual")
names(resid.test) = c("model", "residual")


# Combine the train and test dataframe with an identifying column
resid.train = resid.train %>% mutate(type = "Training")
resid.test = resid.test %>% mutate(type = "Testing")
resid.df = rbind(resid.train, resid.test)


resid.df$model = factor(resid.df$model, levels = c("Ridge", "Lasso", "Elastic Net", "Random Forest"))
resid.df$type = factor(resid.df$type, levels = c("Training", "Testing"))
resid.df$residual = as.numeric(resid.df$residual)

ggplot(resid.df) + 
  aes(x=model, y=as.numeric(residual)) + 
  geom_boxplot() + 
  theme_bw() + 
  facet_grid(.~type, scales="fixed") + 
  theme(axis.text.x = element_text(size = 10)) + 
  theme(axis.text.y = element_text(size = 10)) +
  ylab("Residual") + 
  xlab("Model")


################################################################
##            R-Squared Boxplots                              ##
################################################################


# Update the column types for plotting preparation
rsq.df[,c("train.rsq", "test.rsq")] = apply(rsq.df[,c("train.rsq", "test.rsq")], 2, function(x) as.numeric(x))
names(rsq.df) = c("loop", "model", "Training", "Testing")
rsq.df$model = factor(rsq.df$model, levels = c("Ridge", "Lasso", "Elastic Net", "Random Forest"))


# Reshape the dataframe so the R squared values are in 1 column
rsq.df.melted = melt(rsq.df, id.vars = c("loop", "model"))


ggplot(rsq.df.melted) + 
  aes(x=model, y=as.numeric(value)) + 
  geom_boxplot() + 
  facet_grid(.~variable, scales="fixed") + 
  theme_bw() +
  theme(axis.text.x = element_text(size = 10)) + 
  theme(axis.text.y = element_text(size = 10)) + 
  ylab("R-Squared") +
  xlab("Model")


################################################################
##            10-Fold CV Curves                               ##
################################################################

par(mfrow = c(3,1))

plot(cv.ridge.fit)
title(paste("Ridge 10-Fold CV,", "Time:", round(cv.ridge.time[1],1), units(cv.ridge.time)), line = 2.5)
plot(cv.lasso.fit)
title(paste("Lasso 10-Fold CV,", "Time:", round(cv.lasso.time[1],1), units(cv.lasso.time)), line = 2.5)
plot(cv.elnet.fit)
title(paste("Elastic Net 10-Fold CV,", "Time:", round(cv.elnet.time[1],1), units(cv.elnet.time)), line = 2.5)


################################################################
##            Various Models on Full Dataset                  ##
################################################################

# Subset the original data to create an X matrix and y vector
X = as.matrix(subset(sc[,], select = -c(critical_temp)))
y = unlist(subset(sc[,], select = c(critical_temp)))


# Ridge regression on entire dataset
full.ridge.start = Sys.time()  
full.cv.ridge.fit = cv.glmnet(x = X, 
                              y = y, 
                              alpha = 0)
full.ridge.lam = full.cv.ridge.fit$lambda.min
full.ridge.fit = glmnet(x = X, 
                        y = y, 
                        alpha = 0, 
                        lambda = full.ridge.lam)
full.ridge.time = round(Sys.time() - full.ridge.start,1)

# Lasso regression on entire dataset
full.lasso.start = Sys.time() 
full.cv.lasso.fit = cv.glmnet(x = X, 
                              y = y, 
                              alpha = 1)
full.lasso.lam = full.cv.lasso.fit$lambda.min
full.lasso.fit = glmnet(x = X, 
                        y = y, 
                        alpha = 1, 
                        lambda = full.lasso.lam)
full.lasso.time = round(Sys.time() - full.lasso.start,1)


# Elastic net regression on entire dataset
full.elnet.start = Sys.time()  
full.cv.elnet.fit = cv.glmnet(x = X, 
                              y = y, 
                              alpha = 0.5)
full.elnet.lam = full.cv.elnet.fit$lambda.min
full.elnet.fit = glmnet(x = X, 
                        y= y, 
                        alpha = 0.5, 
                        lambda = full.elnet.lam)
full.elnet.time = round(Sys.time() - full.elnet.start,1)


# Random forest on entire dataset
full.rf.start <- Sys.time()
full.rf.fit = randomForest(x = X, 
                           y = y, 
                           mtry=p/3, 
                           importance=TRUE)
full.rf.time <- round(Sys.time() - full.rf.start,1)



################################################################
##            Preparing and Cleaning Results                  ##
################################################################


# Merge coefficients and importance into 1 dataframe
beta.df = data.frame(Variable = seq(1,p,by=1),
                     Ridge = as.numeric(full.ridge.fit$beta),
                     Lasso = as.numeric(full.lasso.fit$beta), 
                     Elastic.Net = as.numeric(full.elnet.fit$beta),
                     Random.Forest = full.rf.fit$importance[,1])

# Set the order using the elastic net coefficients as reference
beta.df = beta.df %>% 
  arrange(Elastic.Net) %>% 
  mutate(Order = seq(p,1,by=-1))

# Change the variable names to X + variable index number
beta.df$Variable = paste("X",beta.df$Variable,sep="")
beta.df$Variable = factor(beta.df$Variable)

# Sort the data from high to low
beta.df$Variable = reorder(beta.df$Variable, beta.df$Order)

# Reshape the dataframe so the beta coefficients are in 1 column
beta.df = melt(beta.df, id.vars = c("Variable", "Order"))

# Rename the columns name from the melt function
# Add a column to indicate if the coefficients are positive or negative
beta.df = beta.df %>% 
  rename(Coefficient = value, Model = variable) %>% 
  mutate(pos = Coefficient >= 0)

# Plot the coefficients and importance from the regression on full dataset
ggplot(beta.df) + 
  aes(x=Variable, y=Coefficient, fill = pos) + 
  geom_col(show.legend = FALSE) + 
  facet_grid(Model ~., scales="free") + 
  theme_bw() +
  theme(axis.text.x = element_text(size = 7, angle = 30, hjust = 1)) + 
  theme(axis.text.y = element_text(size = 8)) + 
  ylab("Coefficients/Importance") 


# Find the 90% test interval of the R-squared values
ridge.range = quantile(rsq.df[rsq.df[,"model"]=="Ridge", "Testing"], probs=c(0.05, 0.95))
ridge.range = paste(round(ridge.range[1],2), round(ridge.range[2],2), sep="-")
lasso.range = quantile(rsq.df[rsq.df[,"model"]=="Lasso", "Testing"], probs=c(0.05, 0.95))
lasso.range = paste(round(lasso.range[1],2), round(lasso.range[2],2), sep="-")
elnet.range = quantile(rsq.df[rsq.df[,"model"]=="Elastic Net", "Testing"], probs=c(0.05, 0.95))
elnet.range = paste(round(elnet.range[1],2), round(elnet.range[2],2), sep="-")
rf.range = quantile(rsq.df[rsq.df[,"model"]=="Random Forest", "Testing"], probs=c(0.05, 0.95))
rf.range = paste(round(rf.range[1],2), round(rf.range[2],2), sep="-")
interval.df = data.frame(inv = c(ridge.range[1], lasso.range, elnet.range, rf.range), 
                         time = c(full.ridge.time, full.lasso.time, full.elnet.time, full.rf.time))

# Plot a histogram of the response variables
ggplot(sc) + aes(x = critical_temp) + geom_histogram() + theme_bw() + xlab("Critical Temperature") + ylab("Count")

