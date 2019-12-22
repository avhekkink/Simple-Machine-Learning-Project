# My first machine-leaarning project. Analyses data on different iris flowers and fits models
# to predict which species it is, based on their Sepal and Petal lengths and widths.


library(caret)
#downloading the dataset from a csv file
filename <- "iris.csv"
dataset <- read.csv(filename, header=FALSE)
colnames(dataset) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

#creating a train and a test dataset
validation_index <- createDataPartition(dataset$Species, p=0.8, list=FALSE)
#test dataset = validation
validation <- dataset[-validation_index,]
#train dataset = dataset
dataset <- dataset[validation_index,]

#dimensions of dataset
dim(dataset)

#list types for each attribute
sapply(dataset, class)

#snippet of the dataset
head(dataset)

#list of the levels for the class variable: species
levels(dataset$Species)

#summarize the class distribution i.e. how many of each species
percentage <-prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

#summarize the attribute distributions
summary(dataset)

#split the input attributes and the output attribute
x <- dataset[,1:4]
y <- dataset[,5]

#creating 4 boxplots on one image
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

#barplot for class breakdown
plot(y)

#scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#box plot for each attribute
featurePlot(x=x, y=y, plot="box")

#density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#building our models
# a) linear model - Linear Discriminant Analysis (LDA)
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms - Classification and Regression Trees (CART)
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# c) nonlinear algoriths - K-Nearest Neighbours (kNN)
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# d) complex nonlinear algorithms - Support Vector Machines (SVM)
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# e)  complex nonlinear algorithms - Random Forest (RF)
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


#Summary of each model's accuracy
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#compare accuracy of models
dotplot(results)
#We see that lda is the best model

#summarize the best model - lda
print(fit.lda)
#we see this model has 97.5% accuracy +/- 4%

#estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
#model was 100% accurate on the test dataset, woop woop
