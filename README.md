# HW2 Kaggle Competition: Stroke Prediction: End-to-End ML Pipeline

## Summary of modeling pipeline
1. Data was loaded with all variables except for ever_married and work_type to focus on only the variables that were hypothesized to be related to the outcome, preventing overfitting.
2. The features and outcome were first evaluated descriptively to check for imbalance, non-linearity, and potential interactions.
- The outcome was heavily imbalanced with very few cases of stroke (1).
- There were a few moderate correlations between variables, indicating potential for interactions (in line with research).
- No evidence of non-linearity with the logit for the continuous variables, so no polynomial terms were added.
3. The training data was split for validation purposes. Metric features were centered, and interactions were added to the dataset. Different models were tested with the goal of maximizing the F1-score to catch more cases of stroke, keeping in mind the imbalance in the outcome. Parameters were tuned for each model. 
4. Evaluation metric results were compared, specifically focusing on the F1-score.
5. Finally, the best model was chosen, and the competition test data was loaded in, cleaned in the same way as the training data, with final predictions for stroke. 

## Algorithms used
1. Logistic Regression (baseline, with class weights)
2. KNN (k optimized using cross-validation)
3. L2-Regularized Logistic Regression

## Evaluation metrics on training/validation split 
All three models performed similarly on the validation set. The binary logistic regression with class weights achieved the best hold-out F1-score of 0.33, and when submitted to the Kaggle leaderboard, the final public-score F1 was 0.28

| Model | F1   | Precision (1)| Recall(1) | PR-AUC |
|------:|:----:|-------------:|----------:|-------:|
| LogReg| 0.33 | 0.24         | 0.55      | 0.25   |
| KNN   | 0.31 | 0.25         | 0.41      | 0.20   |
| L2Reg | 0.24 | 0.14         | 0.85      | 0.25   |

## Reflections
This was some of the most fun I have had working on an assignment, and great practice for building an ML pipeline. I learned a lot about Python coding and how to actually tune parameters, and it was exciting to see my final score. I made some mistakes along the way, such as almost using the test set mean to center my predictors for the final submission and accidentally including both my centered predictors and regular predictors in the final model. I wondered why my model was having trouble converging and finally realized my mistake!

There were also a few things I would do differently next time, such as exploring different types of added terms, trying out different regularization methods, and comparing with SMOTE to account for the imbalance. I am excited to re-try this submission with different methods!

References:

Gan, Y., Wu, J., Li, L., Zhang, S., Yang, T., Tan, S., Mkandawire, N., Zhong, Y., Jiang, J., Wang,Z., & Lu, Z. (2018). Association of smoking with risk of stroke in middle-aged and older Chinese: Evidence from the China National Stroke Prevention Project. Medicine, 97(47), e13260. https://doi.org/10.1097/MD.0000000000013260

Zhou, L., Chen, G., Liu, C., Liu, L., Zhang, S., & Zhao, X. (2008). Body mass index, blood pressure, and mortality from stroke: a prospective cohort study. Stroke, 39(7), 2065–2071. https://doi.org/10.1161/STROKEAHA.107.495374
