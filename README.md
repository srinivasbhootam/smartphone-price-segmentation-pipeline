# smartphone-price-segmentation-pipeline
## Dataset Description  
We use the *Mobile Price Classification* dataset from Kaggle (Abhishek, 2018). It contains *2,000* smartphone records with *20* technical features—battery power, RAM, internal memory, screen size, camera pixels, processor cores, connectivity options (3G/4G/WiFi/Bluetooth), and more. Each phone is labeled with a price tier from *0 (budget)* to *3 (premium)*.

## Analysis Overview  
1. *Exploratory Data Analysis (EDA)*  
   - Plotted the distribution of price tiers to confirm balanced classes.  
   - Visualized feature histograms and boxplots to inspect ranges and outliers.  
   - Created a correlation heatmap to identify strong relationships (e.g., pixel height vs. pixel width).  

2. *Preprocessing & Dimensionality Reduction*  
   - Scaled all features with StandardScaler to center at zero mean and unit variance.  
   - Applied *PCA* to retain components explaining *95%* of variance, reducing feature count for comparison.

3. *Model Training & Tuning*  
   - Trained five classifiers: Decision Tree, K-Nearest Neighbors, Random Forest, Gaussian Naive Bayes, and Logistic Regression.  
   - Conducted *GridSearchCV* (5-fold CV) to tune parameters such as tree depth, number of neighbors, regularization strength, and number of estimators.  
   - Evaluated each model in three scenarios:  
     1. *Baseline* (default parameters, full features)  
     2. *Hypertuned* (best parameters, full features)  
     3. *PCA* (best parameters, reduced features)

4. *Model Evaluation*  
   - Assessed test-set performance using *accuracy, **precision, **recall, and **F1-score*.  
   - Compiled results into a consolidated table for clear comparison (see Table 1).  
   - Plotted metrics with bar charts, radar charts, dot plots, area charts, and horizontal bars to highlight strengths and weaknesses of each model.

5. *Interpretability with SHAP*  
   - Applied *SHAP* values to the best model (Random Forest) for global and local explanations.  
   - Identified *RAM, **battery power, and **pixel width* as the top drivers of price predictions.  
   - Visualized feature contributions with bar and beeswarm plots for transparent insights.

## Key Results  
- *Random Forest (Hypertuned)* reached the highest accuracy of *0.8925*, with balanced precision (0.8961), recall (0.8925), and F1-score (0.8933).  
- *Logistic Regression (Hypertuned)* performed closely at *0.8800* accuracy and *0.8810* F1-score, showing that a sparse linear model can be effective.  
- *Decision Tree* and *Naive Bayes* achieved moderate accuracies around *0.8350* and *0.7975*, respectively.  
- *KNN* lagged at *0.5600*, indicating distance-based methods require more feature engineering.  
- Models trained on PCA-reduced features saw performance drops for non-linear learners (Random Forest fell to *0.7125*), while Logistic Regression remained unchanged, highlighting that aggressive dimensionality reduction may discard important information for complex models.

