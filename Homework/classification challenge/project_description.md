#  🚘💥 Description: Classification Challenge🚀

Hello Class!

Welcome to the **Vehicle Claims Prediction Project**, where you get the chance to apply your machine learning skills in predicting vehicle claims in an insurance context! 🚘💥 This project is designed to challenge your problem-solving abilities and sharpen your data science toolkit. The students with the best scores will earn the spotlight! You'll be invited to present your solution in class and share your insides!

## 🔍 What’s the Goal?

Your mission is to **predict the missing labels** for a dataset of vehicle claims. You'll be working with real-world data which we messed up, so careful data preprocessing, smart feature engineering will be crucial for success. 

Your model will be evaluated using the **F1 Score** with `average='macro'`.  
This means the score equally weighs all classes — a great way to measure performance on imbalanced datasets.

Here's the code snippet you'll use for evaluation:

```python
from sklearn.metrics import accuracy_score, classification_report, f1_score

score = f1_score(y_true, y_pred, average='macro')
```
Note: Reaching a Score of 65 % is possible with basic code.

## 🗂 Files Overview

You’ll be working with three key files:

- **unlabeled_data.csv**: This file has 30% of the dataset, but **no labels**. Your task? Predict those missing labels using the magic of machine learning! 🔮
  
- **labeled_data.csv**: Here’s where the training happens! This file contains 70% of the dataset with labels, giving you the data needed to train your model.
  
- **example_prediction.csv**: This is your guide! It shows how your final submission should be structured, including the **ID** column from labeled_data.csv and a placeholder label column set to 0. Your final predictions need to follow this structure, but with **your predicted labels** instead!

## Steps by step
1. **📊 Preprocess the Datasets**: 
   - Load both labeled_data.csv and unlabeled_data.csv
   - Handle missing values, outliers, and normalize/standardize if needed
   - Feature Engineering: Clean up the data and craft smarter features that help models learn better.
   - Encode categorical variables (e.g., One-Hot or Target Encoding)
   - ...


2. **🧠 Train Your Model**
   - Split labeled_data.csv into training and validation sets
   - Train machine learning models(e.g., Logistic Regression, RandomForest, XGBoost, ...) and find the optimal model and hyperparameter with k-fold cross calidation 

   - Use the given validation metrics f1_score(y_true, y_pred, average='macro') 

3. **Prepare Your Predictions**
   - After applying your model on the unlabeled_data.csv data, save your output in the same format as example_prediction.csv.

4. **Follow the Structure**: Make sure your final file includes:
   - **ID Column**: Identical to labeled_data.csv.
   - **Label Column**: Replace the placeholder 0s with your actual predicted labels.

5. **Upload Your File**
   - Once your prediction file is ready and follows the correct format, upload it to this link for evaluation:
   https://s25redi-preproject.streamlit.app/

## 📅 Important Dates
### 🗓️ Homework Deadline: 29.04.2025 – 18:00
- Do step 1. 📊 Preprocess the Datasets* and submit your code to clean the data via Google Classroom.
- If you want you can also start and train your first model but this is optional. 
- Timo will review your code and share feedback in class on 30.04.2025, along with practical tips🚀

### 🏁 Challenge Deadline: 18.05.2025 23:59
- The main challenge continues until this date!
The best-performing student will earn honor and recognition 🎉 – and the opportunity to present their solution and approach in class on.

### 📢 Final Presentation: 21.05.2025 
- Get ready to share your code, ideas, and strategy with everyone! It's your time to shine! 🌟

## 🛠 Teamwork

You want to work in a group? Feel free to share ideas and help with errors, but everybody should write there own code.

## 🛠 Resources

You can choose any ML methode you want. 
Have questions? Stuck on a challenge? We’re here to support you! Don’t hesitate to reach out for guidance along the way in slack. 

Best of luck—get ready to dive into the data, explore possibilities, and create your best model yet! 💪🎉
