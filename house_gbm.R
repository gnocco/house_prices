library(h2o)
h2o.init()

train<-h2o.importFile("C:/Users/asu/Downloads/train.csv")
test<-h2o.importFile("C:/Users/asu/Downloads/test.csv")

#### Exploratory Data Analysis ######

space<-train[,-81]

h2o.nrow(test)
#h2o.describe(space)  ## Returns the first rows of an H2OFrame object. Useful to inspect the data
together<-h2o.rbind(space,test)

h2o.nrow(together)
### Convert Numeric to Categorical ###

to_factors <- c(3,6:17,22:26,28:34,36,40:43,54,56,58,59,61,64:66,73:75,79,80)

#for(i in to_factors) data[,i] <- data[,i] - 1
for(i in to_factors) together[,i] <- h2o.asfactor(together[,i])

col_orig<-setdiff(h2o.colnames(together), c("Id"))

pca<-h2o.prcomp(together[,col_orig], k = 5, impute_missing = TRUE, transform = "NORMALIZE")

train_f<-h2o.cbind(together[1:1460,col_orig],h2o.predict(pca,together[1:1460,col_orig]),train[,81])
test_f<-h2o.cbind(together[1461:2919,"Id"],together[1461:2919,col_orig],h2o.predict(pca,together[1461:2919,col_orig]))
test_f

h2o.describe(train_f)    # Describe again to validate the column information

### Summarize the dara ##
### Displayes the minimum, 1st quartile, median, mean, 3rd quartile and maximum for each numeric column, and the levels and
### category counts of the levels in each categorical column.


### Display the structure of an H2OFrame object ##
h2o.str(train_f) 

### Performs a group by and apply similar to ddply. ##

h2o.hist(h2o.log(train_f[,"SalePrice"]))

### Define  Creditability ( Good/Bad credit) as the Target for modeling.
target <- "SalePrice"  

### Everything other than the target are the predictors
features <- setdiff(h2o.colnames(train_f), c(target))

print(target)
print(features)

### Partition the data into training(60%) and test set(40%).
### setting a seed will guarantee reproducibility
#splitted<-h2o.splitFrame(train_f, seed = 1)
#te<-splitted[[2]]
#tr<-splitted[[1]]


### Now that we have prepared our data, let us train some models.
### We will start by training a h2o.glm model

gbm_model1 <- h2o.gbm(x = features,
                      y = target,
                      training_frame = train_f,
                      #validation_frame = te,
                      model_id = "gbm_model1",
                      nfolds = 10,
                      ntrees = 300,
                      max_depth = 10,
                      keep_cross_validation_predictions = TRUE,
                      distribution = "poisson")

###Evaluate the model summary   

print(summary(gbm_model1))

###Evaluate model performance on test data

#perf_obj <- h2o.performance(gbm_model1, newdata = house_test)
#h2o.accuracy(perf_obj, 0.949411607730009)

pred_creditability <- h2o.cbind(test_f[,"Id"],h2o.predict(gbm_model1,test_f[,features]))
h2o.exportFile(pred_creditability,"C:/Users/asu/Downloads/prediction.csv")
pred_creditability
