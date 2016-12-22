# Enron-Fraud-Identification
By Rafael Velasquez

## Question 1

The goal of this project is to identify persons of interest in the Enron fraud investigation. Machine learning is a useful tool because we have a significant amount of data available to help us with this identification. Specifically, the data is high-dimensional with no obvious relationship between many of the variables and thus perfect for the application of machine learning algorithms that do most of the model identification for us.

The dataset mostly consists of financial data of employees at Enron. There are 146 observations across 20 variables. Of the 146 observations, 18 are identified as persons of interest. We would expect persons of interest to have unusual financial observations (larger bonuses, stock grants, etc.) and so this data is appropriate to our objective. There were a couple of outliers in the data. One was a total row, which was easy to spot because its values where significantly greater than anything else, the second was called "The travel agency in the park", and almost all its variables where NA and thus was easy to spot as well. Outliers were handled in the analysis by simply deleting the entries from the dictionary with the data.



## Question 2

For our analysis, feature selection was performed for the final selection as a somewhat iterative manual process. We began by including all the features and testing the algorithm. This unfortunately did not give great results with precision of 0.39 , and recall of 0.305. Mediocres results and too close to our 30% cutoff for comfort.

Next we tried to use the SelectKBest function from sklearn to create a parsimonious model with k=5.  The top 5 features from the algorithm were (with their respective scores): excercised_stock_options (24.8), total_stock_value (24.2), bonus (20.8), salary (18.3), and a custom feature to be explained later, from_poi_prop (16.4). Unfortunately, this also gave underwhelming results. When we tested this algorithm our precision was 0.25281 and our recall 0.225, well below our desired scores.

Given these issues we decided to take a step back and tried to be parsimonious. We decided to test the idea that persons of interest would have significant contact among each other. In order to do this we generated two proprietary features. They are, the proportion of emails sent or received by an individual from or to a person of interest. These feature seemed intuitive to us as the scaled data was more useful than the raw results which could just be related to the number of emails sent by the person (i.e. a secretary might send out a lot of emails increasing the number of emails sent to a person of interest but as a percentage would be small). In addition to this, the other feature we used was shared_receipt_with_poi as this meant that there had been live interaction with a poi, such as a business trip, meeting, or meal. 

We also scaled all the features using a minmax scaler. Since the features where very disparate ranging from amounts in the millions to percentages, it was helpful to get everything on the same scale for the identification of interesting data.

The feature importance for our algorithm was as follows:

from_poi_prop = 0.18
to_poi_prop = 0.22
shared_receipt_with_poi = 0.6

Lending credence to our theory that live interaction between POIs is a good identifier.
 

## Question 3

The algorithm we ended up using was adaboost as it showed the best results. Some other algorithms tested in the process were random forests and logistic regression.

The performance for the random forest classifier showed a Precision of 0.428 and a recall of 0.324. The adaboost was much improved with a performance featuring Precision of 0.44240 and recall of 0.38400.

As we can see the random forest was slightly worse at avoiding false positives while the Adaboost was significantly better at choosing true positives.

## Question 4

Most algorithms have a certain set of parameters which can be tuned in order to obtain the best results from them. Incorrectly setting the parameters can lead to significant model selection issues such as overfitting. For example if enough nodes are used in a decision tree an analyst can almost perfectly fit the data in sample but to the detriment of its out of sample results. For a K Nearest Neighbours algorithm, select more distinct groups than there actually are in the data can lead to spurious and unreliable results.

I tuned the parameters using the GridSearchCV function in sklearn and stratified shuffle split as the cross validation. Grid search tries different combinations if user inputted parameters and picks the one that performs the best in out of sample cross validation. For our adaboost algorithm the parameters we tuned where n_estimators and the learning rate. N_estimators represents the number of estimators to be used for the boosting while the learning rate determines the contribution of each classifier. Stratified shuffle split was used since it is a useful technique for small n, and unbalanced datasets similar to ours. It creates a series of random test and training sets, but preserves the proportion of the class to be identified. Although this violates the rule that we must not look at labels until after training the algorithm it does not cause significant overfitting issues.

The grid searched over for the adaboost algorithm was:
                     
                     params = {'n_estimators': [50,100,200,500], 'learning_rate': [0.2,0.4,0.6,1]}
                     


## Question 5 

Cross-Validation, is a method for testing the value of a predictive model. It seeks to answer a simple question, how good is this model at predicting observations. It is performed by splitting the data into a training and test set, fitting the model on the training set and then calculating its accuracy on the test set. There are different approaches to this. A popular one is k-folds cross-validation which splits the data into differents folds and trains it on k-1 folds at any one time and tests on the left out fold and does this k times so that all the data is used both for testing and training in an intelligent way.

This is the way we validated our analysis, achieving satisfactory results of a Precision of 0.44240 and recall of 0.38400.

One mistake to avoid in cross-validation, is to put undue weight on the value of the accuracy score instead of the precision and recall scores. This happens in situation similar to ours where there is an imbalance between the classes to be classified into. In our case, there are a small amount of persons of interest, so always guessing that a person is not a person of interest would give us a high accuracy but unusable results, that's why our emphasis has been placed on the Precision and Recall which measures how accurate the algorithm has been at identifying Persons of Interest, not identifying people who are not POIs.

## Question 6

Our two chosen evaluation metrics are Precision and Recall. Precision measures out of the proportion of all results that were identified as persons of interest, how many were actually persons of interest. Recall measures, out of all the true persons of interest how many where identified. So as you can see there is a tradeoff between them that is important for the analyst to consider, especially given what is more costly falsely identifying someone as a POI or missing a POI.

For our algorithm, we achieved Precision of 0.44240 and recall of 0.38400. So, out of the people it identified as POIs 44.2% of them where actual POIs, and out of all the POIs, it identified 38.4% of them correctly.
