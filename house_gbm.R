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

## Construct a large Cartesian hyper-parameter space
ntrees_opts <- c(100,1000,10000) ## early stopping will stop earlier
max_depth_opts <- seq(1,20)
min_rows_opts <- c(1,5,10,20,50,100)
learn_rate_opts <- seq(0.001,0.01,0.001)
sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_opts <- seq(0.3,1,0.05)
col_sample_rate_per_tree_opts = seq(0.3,1,0.05)
#nbins_cats_opts = seq(100,10000,100) ## no categorical features in this dataset

hyper_params = list( ntrees = ntrees_opts,
                     max_depth = max_depth_opts,
                     min_rows = min_rows_opts,
                     learn_rate = learn_rate_opts,
                     sample_rate = sample_rate_opts,
                     col_sample_rate = col_sample_rate_opts,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opts
                     #,nbins_cats = nbins_cats_opts
)


## Search a random subset of these hyper-parmameters (max runtime and max models are enforced, and the search will stop after we don't improve much over the best 5 random models)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 600, max_models = 100, stopping_metric = "AUTO", stopping_tolerance = 0.00001, stopping_rounds = 5, seed = 123456)

gbm.grid <- h2o.grid("gbm",
                     grid_id = "mygrid",
                     x = features,
                     y = target,
                     
                     # faster to use a 80/20 split
                     #training_frame = trainSplit,
                     #validation_frame = validSplit,
                     #nfolds = 0,
                     
                     # alternatively, use N-fold cross-validation
                     training_frame = train_f,
                     nfolds = 5,
                     
                     distribution="gaussian", ## best for MSE loss, but can try other distributions ("laplace", "quantile")
                     
                     ## stop as soon as mse doesn't improve by more than 0.1% on the validation set,
                     ## for 2 consecutive scoring events
                     stopping_rounds = 2,
                     stopping_tolerance = 1e-3,
                     stopping_metric = "MSE",
                     
                     score_tree_interval = 100, ## how often to score (affects early stopping)
                     seed = 123456, ## seed to control the sampling of the Cartesian hyper-parameter space
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

gbm.sorted.grid <- h2o.getGrid(grid_id = "mygrid", sort_by = "mse")
print(gbm.sorted.grid)

best_model <- h2o.getModel(gbm.sorted.grid@model_ids[[1]])
summary(best_model)




pred_creditability <- h2o.cbind(test_f[,"Id"],h2o.predict(best_model,test_f[,features]))
h2o.exportFile(pred_creditability,"C:/Users/asu/Downloads/prediction.csv")
pred_creditability
