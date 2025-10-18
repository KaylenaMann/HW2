# HW2 Kaggle Competition: Stroke Prediction: End-to-End ML Pipeline

## Summary of modeling pipeline
1. The data was loaded with all variables except for ever_married and work_type because these features were hypothesized not to have strong theoretical associations with stroke risk, while also helping prevent unnecessary model complexity. 
2. The features and outcome descriptives were first evaluated to check for imbalance, non-linearity, and potential interactions.
- The outcome was heavily imbalanced with very few cases (4.13%) of stroke (1).
- There were a few moderate correlations between variables, indicating potential for interactions.
- No evidence of non-linearity with the logit for the continuous variables, so no polynomial terms were added.
3. The training data was split for validation purposes. Metric features were centered, and interactions were added to the dataset. Different models were evaluated using multiple hyperparameter configurations, with the goal of maximizing the F1-score to catch more cases of stroke, while also accounting for class imbalance. 
4. Evaluation metric results were compared, and the binary logistic regression with class weights achieved the strongest validation F1-score of 0.33.
5. Binary logistic regression was selected for the final submission and applied to the Kaggle test set after utilizing identical preprocessing steps to the test data.
 
## Algorithms used
1. Logistic Regression (baseline, with class weights)
2. KNN (k optimized using cross-validation)
3. L2-Regularized Logistic Regression

## Evaluation metrics on training/validation split 
All three models performed similarly on the validation set. The binary logistic regression with class weights achieved the best hold-out F1-score of 0.33, and a relatively good balance of precision and recall. When submitted to the Kaggle leaderboard, the final score was 0.28

| Model | F1   | Precision(1)| Recall(1) | PR-AUC |
|------:|:----:|------------:|----------:|-------:|
| LogReg| 0.33 | 0.24        | 0.55      | 0.25   |
| KNN   | 0.31 | 0.25        | 0.41      | 0.20   |
| L2Reg | 0.24 | 0.14        | 0.85      | 0.25   |

*Note that due to the competition being over, my score cannot be added to the leaderboard; instead, a screenshot was included of my final score.*

<img width="999" height="152" alt="Leaderboard:Submission Results Screenshot" src="https://github.com/user-attachments/assets/5afd82a1-c773-405b-8bb1-e8e243f4cfd5" />

## Reflections
This was some of the most fun I have had working on an assignment, and great practice for building an ML pipeline. My Python coding greatly improved, and I learned how to actually tune parameters. I definitely made some mistakes along the way, such as almost using the test set mean to center my predictors for the final submission, which would have resulted in leakage. I also spent a lot of time debugging. For example, I was having trouble with model convergence and realized I accidentally included both my centered predictors and regular predictors in the final model, which meant high collinearity!

There was a drop in score from my original F1 to the Kaggle submission, indicating potential sampling differences, overfitting, or better models to consider. There were a few things I would do differently next time, such as exploring different feature engineering, comparing other regularization methods, and utilizing SMOTE to better account for class imbalance. I am excited to re-try this submission with different methods next time!

References:

Gan, Y., Wu, J., Li, L., Zhang, S., Yang, T., Tan, S., Mkandawire, N., Zhong, Y., Jiang, J., Wang,Z., & Lu, Z. (2018). Association of smoking with risk of stroke in middle-aged and older Chinese: Evidence from the China National Stroke Prevention Project. Medicine, 97(47), e13260. https://doi.org/10.1097/MD.0000000000013260

Zhou, L., Chen, G., Liu, C., Liu, L., Zhang, S., & Zhao, X. (2008). Body mass index, blood pressure, and mortality from stroke: a prospective cohort study. Stroke, 39(7), 2065–2071. https://doi.org/10.1161/STROKEAHA.107.495374
