library(rsparkling)
library(sparklyr)
library(h2o)
library(tidyverse)

set.seed(1234)
theme_set(theme_minimal())


# connect to spark
sc <- spark_connect(master = "local")

# load sample data

library(titanic)
titanic_tbl <- copy_to(sc, titanic::titanic_train, "titanic", overwrite = TRUE)

# tidy the data

titanic2_tbl <- titanic_tbl %>% 
  mutate(Family_Size = SibSp + Parch + 1L) %>% 
  mutate(Pclass = as.character(Pclass)) %>%
  filter(!is.na(Embarked), Embarked != "") %>%
  mutate(Age = if_else(is.na(Age), mean(Age), Age)) %>%
  sdf_register("titanic2")


# Spark-specific dplyr transformation

titanic_final_tbl <- titanic2_tbl %>%
  mutate(Family_Size = as.numeric(Family_size)) %>%
  sdf_mutate(
    Family_Sizes = ft_bucketizer(Family_Size, splits = c(1,2,5,12))
  ) %>%
  mutate(Family_Sizes = as.character(as.integer(Family_Sizes))) %>%
  sdf_register("titanic_final")


# train- validation split

# Partition the data
partition <- titanic_final_tbl %>% 
  mutate(Survived = as.numeric(Survived),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch)) %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Family_Sizes) %>%
  sdf_partition(train = 0.75, test = 0.25, seed = 1234)

# Create table references
train_tbl <- partition$train
test_tbl <- partition$test


## logistc regression

ml_formula <- formula(Survived ~ Pclass + Sex + Age + SibSp +
                        Parch + Fare + Embarked + Family_Sizes)

# Train a logistic regression model

ml_log <- ml_logistic_regression(train_tbl, ml_formula)

# Decision Tree
ml_dt <- ml_decision_tree(train_tbl, ml_formula)

# Random Forest
ml_rf <- ml_random_forest(train_tbl, ml_formula)

# Gradient Boosted Tree
ml_gbt <- ml_gradient_boosted_trees(train_tbl, ml_formula)

# Naive Bayes
ml_nb <- ml_naive_bayes(train_tbl, ml_formula)

# Neural Network
ml_nn <- ml_multilayer_perceptron(train_tbl, ml_formula, layers = c(11, 15, 2))


## Validation

# Bundle the models into a single list object
ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Naive Bayes" = ml_nb,
  "Neural Net" = ml_nn
)

# Create a function for scoring
score_test_data <- function(model, data = test_tbl){
  pred <- sdf_predict(data, model)
  select(pred, Survived, prediction)
}

# Score all the models
ml_score <- map(ml_models, score_test_data)



# Lift function
calculate_lift <- function(scored_data) {
  scored_data %>%
    mutate(bin = ntile(desc(prediction), 10)) %>% 
    group_by(bin) %>% 
    summarize(count = sum(Survived)) %>% 
    mutate(prop = count / sum(count)) %>% 
    arrange(bin) %>% 
    mutate(prop = cumsum(prop)) %>% 
    select(-count) %>% 
    collect() %>% 
    as.data.frame()
}

# Initialize results
ml_gains <- data_frame(
  bin = seq(from = 1, to = 10),
  prop = seq(0, 1, len = 10),
  model = "Base"
)

# Calculate lift
for(i in names(ml_score)){
  ml_gains <- ml_score[[i]] %>%
    calculate_lift %>%
    mutate(model = i) %>%
    bind_rows(ml_gains, .)
}

# Plot results
ggplot(ml_gains, aes(x = bin, y = prop, color = model)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(type = "qual") +
  labs(title = "Lift Chart for Predicting Survival",
       subtitle = "Test Data Set",
       x = NULL,
       y = NULL)



# Function for calculating accuracy
calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_classification_eval("prediction", "Survived", "accuracy")
}

# Calculate AUC and accuracy
perf_metrics <- data_frame(
  model = names(ml_score),
  AUC = 100 * map_dbl(ml_score, ml_binary_classification_eval, "Survived", "prediction"),
  Accuracy = 100 * map_dbl(ml_score, calc_accuracy)
)
perf_metrics


# Plot results
gather(perf_metrics, metric, value, AUC, Accuracy) %>%
  ggplot(aes(reorder(model, value), value, fill = metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() +
  labs(title = "Performance metrics",
       x = NULL,
       y = "Percent")


# Initialize results
feature_importance <- data_frame()

# Calculate feature importance
for(i in c("Decision Tree", "Random Forest", "Gradient Boosted Trees")){
  feature_importance <- ml_tree_feature_importance(ml_models[[i]]) %>%
    mutate(Model = i) %>%
    rbind(feature_importance, .)
}

# Plot results
feature_importance %>%
  ggplot(aes(reorder(feature, importance), importance, fill = Model)) + 
  facet_wrap(~Model) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  labs(title = "Feature importance",
       x = NULL) +
  theme(legend.position = "none")



## compare model times

# Number of reps per model
n <- 10

# Format model formula as character
format_as_character <- function(x){
  x <- paste(deparse(x), collapse = "")
  x <- gsub("\\s+", " ", paste(x, collapse = ""))
  x
}

# Create model statements with timers
format_statements <- function(y){
  y <- format_as_character(y[[".call"]])
  y <- gsub('ml_formula', ml_formula_char, y)
  y <- paste0("system.time(", y, ")")
  y
}

# Convert model formula to character
ml_formula_char <- format_as_character(ml_formula)

# Create n replicates of each model statements with timers
all_statements <- map_chr(ml_models, format_statements) %>%
  rep(., n) %>%
  parse(text = .)

# Evaluate all model statements
res <- map(all_statements, eval)

# Compile results
result <- data_frame(model = rep(names(ml_models), n),
                     time = map_dbl(res, function(x){as.numeric(x["elapsed"])})) 

# Plot
result %>%
  ggplot(aes(time, reorder(model, time))) + 
  geom_boxplot() + 
  geom_jitter(width = 0.4, aes(color = model)) +
  scale_color_discrete(guide = FALSE) +
  labs(title = "Model training times",
       x = "Seconds",
       y = NULL)


# h2o and logistic regression
library(h2o)
library(rsparkling)

titanic_h2o <- titanic_final_tbl %>% 
  mutate(Survived = as.numeric(Survived),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch)) %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Family_Sizes) %>%
  as_h2o_frame(sc, ., strict_version_check = FALSE)
