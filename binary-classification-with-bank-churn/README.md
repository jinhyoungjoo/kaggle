# Binary Classification with a Bank Churn Dataset
## 2024 Kaggle Playground Series (Season 4 Episode 1)
2024.01.02 ~ 2024.01.31 | [Competition Link](https://www.kaggle.com/competitions/playground-series-s4e1/overview)

### Task
Predict whether a customer continues with their account or closes it (e.g., churns) given
the input data.

### Score
Scores are evaluated using the area under the ROC curve in this competition.
- Private Leaderboard: 0.89405 (596 / 3632)
- Public Leaderboard: 0.89086 (582 / 3633)

### My Solution
I used a voting classifier with XGBoost, CatBoost, and LightGBM. Hyperparameters are tuned using
Optuna. I used the ideas from [this notebook](https://www.kaggle.com/code/rohangulati14/easy-solution-for-binary-classification) for feature engineering. The following scripts will run the code.
```bash
python main.py              # For generating submission.
python main.py --optimize   # For hyperparameter optimization.
```
