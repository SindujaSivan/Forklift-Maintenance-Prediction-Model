### Introduction
Forklifts play a crucial role in warehouse and manufacturing operations, facilitating the movement of goods and materials. Timely maintenance of forklifts is essential to ensure smooth operations, minimize downtime, and enhance safety. This project aims to develop a predictive maintenance model for forklifts, leveraging machine learning techniques to predict maintenance requirements based on various operational parameters.

### Objectives
- Develop a predictive maintenance model to forecast forklift maintenance requirements.
- Improve operational efficiency by proactively scheduling maintenance tasks.
- Reduce downtime and associated costs by identifying maintenance needs in advance.
- Enhance safety by ensuring that forklifts are properly maintained and in optimal working condition.

### Sample Input Data
```
| Forklift  | Usage Hours | Temperature (°C) | Vibration (Hz)  | Age (years) | Weight Capacity (kg)  | Fuel Type    | Maintenance  |
------------------------------------------------------------------------------------------------------------------------------------
| 4TGQF56   |   2500 hrs  |       35         |      2.8        |      5      |         3500          |   Electric   |   Yes        |
| 5EGRY94   |   4200 hrs  |       40         |      3.5        |      8      |         4500          |   Gas        |   Yes        |
| 9WGPJ59   |   1800 hrs  |       30         |      2.0        |      3      |         3000          |   Diesel     |   No         |
| 8QFKL23   |   3800 hrs  |       38         |      3.2        |      6      |         4000          |   Electric   |   Yes        |
------------------------------------------------------------------------------------------------------------------------------------
```
**Data Usage:**
- **Forklift**: The unique identfiier for each forklift. We do not use this for modelling as it may lead to overfitting. 
- **Usage Hours**: The model utilizes data on the number of hours each forklift has been in operation. Forklifts with higher usage hours are more likely to require maintenance due to wear and tear.
- **Temperature**: Temperature data is collected from sensors installed on the forklifts. High temperatures may indicate overheating or inefficient cooling systems, which can lead to mechanical issues.
- **Vibration**: Vibration levels are measured using accelerometers attached to the forklifts. Increased vibration can be a sign of worn-out components or misalignment, indicating the need for maintenance.
- **Age**: The age of each forklift is considered as a factor since older forklifts may have higher probabilities of component degradation and failure.
- **Weight Capacity**: The weight capacity of the forklifts is taken into account as heavier loads may exert more stress on the components, leading to increased maintenance requirements.
- **Fuel Type**: The type of fuel used by the forklifts is also considered, as different fuel types may have varying effects on the forklift's performance and maintenance needs.
- **Maintenance**: Historical data of whether the maintence is required or not 

### Methodology
1. **Data Collection**: Gather historical data on forklift operations, including usage hours, temperature, vibration levels, age, weight capacity, and fuel type. This data will serve as the basis for training and evaluating the predictive maintenance model.

2. **Data Preprocessing**: Clean and preprocess the collected data, including handling missing values, encoding categorical variables, and scaling numerical features.

3. **Model Development**: Train machine learning models using the preprocessed data to predict forklift maintenance requirements. Several classification algorithms such as Random Forest, Logistic Regression, Support Vector Machine, and Gradient Boosting will be evaluated for their performance.

4. **Model Evaluation**: Evaluate the trained models using performance metrics such as accuracy, precision, and recall. Select the best-performing model based on these metrics for deployment.

5. **Deployment**: Deploy the selected model into the production environment, where it can be used to predict maintenance requirements for forklifts in real-time. Integrate the model into existing maintenance management systems or develop a standalone application for easy access.

6. **Monitoring and Maintenance**: Continuously monitor the performance of the deployed model and update it as necessary. Collect feedback from maintenance technicians and stakeholders to improve the model's accuracy and effectiveness over time.

```python
# Preprocess data
X = data.drop(columns=['MaintenanceRequired'])
y = data['MaintenanceRequired']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with different parameters
models = {
    "Random Forest (100 estimators)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Random Forest (500 estimators)": RandomForestClassifier(n_estimators=500, random_state=42),
    "Logistic Regression (C=1.0)": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Logistic Regression (C=0.5)": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "Support Vector Machine (C=1.0)": SVC(C=1.0, random_state=42),
    "Support Vector Machine (C=0.5)": SVC(C=0.5, random_state=42),
    "Gradient Boosting (n_estimators=100)": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting (n_estimators=200)": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# Train and evaluate models
accuracy_scores = {}
for name, model in models.items():
    if "Logistic Regression" in name:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test_scaled if "Logistic Regression" in name else X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    results[name] = {'F1-score': f1, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
```
![Forklift predictive maintenance](assets/Forklift_predictive_maintenance.png)

### Results & Recommendation
The developed forklift maintenance prediction model demonstrates promising results, achieving high accuracy, precision, and recall in forecasting maintenance requirements. By leveraging this model, organizations can proactively schedule maintenance tasks, reduce downtime, and enhance overall operational efficiency.

**Report Sample:**
When a new reading it recorded , the model will predict if the maintenance is required or not based on the pre-trained model and matches it against the forklift number to find if maintenance is required or not. 
A sample report below

```
Forklift Maintenance Prediction Report - ABC Company

Date: 12-September-2022

Predicted Maintenance Requirements:
-------------
| Forklift  |  
-------------
| 8GTIF98   |   
| 9WWPJ76   |   
-------------
```

**Recommendations:**
1. Schedule preventive maintenance for 8GTIF98 & 9WWPJ76 as it has exceeded the threshold for usage hours and exhibits high temperature and vibration levels.
2. Record the critical input information regularly.

### Conclusion
The forklift maintenance prediction model offers a proactive approach to maintenance management, enabling organizations to anticipate maintenance needs and take timely action. By leveraging machine learning techniques, businesses can optimize forklift operations, minimize disruptions, and ensure a safe working environment.
