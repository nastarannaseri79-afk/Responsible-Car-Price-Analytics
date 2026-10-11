# Car Price Predictive Analysis

## Project Overview

This project investigates which vehicle characteristics are associated with car prices and how accurately these characteristics can predict price. It was developed as part of the Code Institute Data Analytics and AI Bootcamp.

I reused the car-price dataset from Project 1. Due to directory-structure issues that made committing the original project to GitHub difficult, I decided to rebuild the project from the beginning. I kept the same dataset but repeated the data preparation, analysis, and modelling workflow to improve the project structure and my understanding of the data analytics process.

## Project Questions

- How are car prices distributed?
- Which vehicle characteristics have the strongest relationship with price?
- Do prices differ between selected vehicle groups?
- How accurately can regression models predict car prices?
- Which features are most important to the best-performing model?

## Project Workflow

The project follows a three-notebook workflow covering data preparation, exploratory data analysis, statistical testing, and machine learning.

### Notebook 1: Data Preparation and Cleaning

- **Data inspection:** Examined the dataset's dimensions, column data types, missing values, and duplicate records. Confirmed that there were no missing values or exact duplicate rows.
- **Outlier screening:** Applied the Interquartile Range (IQR) method to screen continuous variables for potential outliers. Flagged observations were retained because they could represent genuine differences between standard, high-performance, and luxury vehicles rather than data-entry errors.
- **Data cleaning:** Removed the redundant `car_ID` identifier and standardised inconsistent manufacturer names, including `maxda` to `mazda`, `porcshce` to `porsche`, `toyouta` to `toyota`, and `vokswagen` and `vw` to `volkswagen`.
- **Feature engineering:** Extracted `car_model` from `CarName` and created additional features:
  - `price_per_hp` = `price / horsepower`
  - `power_to_weight` = `horsepower / curbweight`
  - `avg_mpg` = `(citympg + highwaympg) / 2`
- **Export:** Saved the cleaned dataset to `outputs/car_prices_cleaned.csv`.

The original dataset contains 205 rows and 26 columns. The cleaned dataset contains 205 rows and 31 columns.

**Important modelling consideration:** Since `price_per_hp` is calculated directly from the target variable, `price`, it must not be used as a predictor when predicting price. Including it would introduce target leakage and produce misleading evaluation results.

### Notebook 2: Exploratory Data Analysis and Statistics

- **Descriptive statistics:** Calculated measures of central tendency, variance, standard deviation, and skewness. The price distribution had a skewness of 1.78, indicating a positively skewed distribution.
- **Correlation analysis:** Examined relationships between car prices and numerical features. The strongest reported correlations included:
  - `enginesize`: \(r = 0.8741\)
  - `curbweight`: \(r = 0.8353\)
  - `horsepower`: \(r = 0.8081\)
  - `avg_mpg`: \(r = -0.6968\)
- **Hypothesis testing:** Used Welch's independent-samples t-test to compare prices between front-wheel-drive (FWD) and rear-wheel-drive (RWD) vehicles. The result was statistically significant (\(p \approx 9.63 \times 10^{-16}\)), leading to rejection of the null hypothesis for this comparison.
- **Group comparisons:** Used Welch's ANOVA to investigate price differences across body styles and drivewheel groups, applying Holm correction for multiple comparisons.

These analyses helped identify important patterns in the dataset and investigate whether observed differences between vehicle groups were statistically significant.

### Notebook 3: Machine Learning and Car Price Prediction

The objective of this notebook was to compare regression models and evaluate their ability to predict car prices on unseen data.

- **Data splitting:** Split the dataset into an 80% training set (164 rows) and a 20% test set (41 rows).
- **Preprocessing pipeline:** Used `ColumnTransformer` and `OneHotEncoder` to process categorical variables. The preprocessing pipeline was fitted on the training data to prevent data leakage from the test set.
- **Baseline model:** Implemented a `DummyRegressor` that predicts the mean training price, providing a baseline against which more advanced models could be evaluated.
- **Linear Regression:** Trained an ordinary least squares regression model using the encoded feature set.
- **Random Forest Regression:** Trained a Random Forest model to capture potentially nonlinear relationships between vehicle characteristics and price.
- **Model evaluation:** Compared model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (\(R^2\)).

The reported Linear Regression results were approximately MAE = 2,702, RMSE = 3,812, and \(R^2 = 0.816\). The reported Random Forest results were MAE = 1,352, RMSE = 1,886, and \(R^2 = 0.955\).

These results are provisional until the notebooks have been rerun and the reported metrics reconciled.

## Visualisation

I used different visualisation libraries depending on the analysis:

- **Matplotlib and Seaborn:** Used for statistical plots, price distributions, boxplots, scatter plots, and correlation heatmaps.
- **Plotly Express:** Used to create interactive visualisations.

One useful learning experience occurred when I plotted the 15 most expensive cars. The chart did not look as expected, so I investigated the underlying data and found two records with the same `CarName`. This highlighted the importance of checking data quality, grouping logic, and the assumptions behind a visualisation rather than relying on the chart alone.

## Main Findings

The analysis identified several important patterns:

- Engine size, curb weight, and horsepower showed strong positive correlations with car prices.
- Average fuel economy showed a negative correlation with price.
- The analysis found a statistically significant difference in mean price between FWD and RWD vehicles in this sample.
- Random Forest Regression achieved the best reported predictive performance among the compared models.
- Engine size was reported as the most important feature in the fitted Random Forest model.

### Model Comparison

The following table presents the provisional results reported in the model-comparison section of the notebook.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 6,339.80 | 8,889.04 | -0.001 |
| Linear Regression | 2,702.39 | 3,812.40 | 0.816 |
| Random Forest Regression | 1,352.20 | 1,885.51 | 0.955 |

*Note: The model metrics differ slightly from the figures reported elsewhere in the notebook. These results must be verified by rerunning the complete workflow. The price units are those used in the original dataset.*

MAE measures the average absolute difference between predicted and actual prices. RMSE gives greater weight to larger prediction errors, while \(R^2\) indicates how much variation in the test-set prices is explained by the model relative to a mean-prediction baseline.

## Use of AI Tools

I used AI tools throughout the project to support my learning, problem-solving, and development process.

- **ChatGPT:** Helped me develop a project roadmap and notebook structure based on the Code Institute assessment instructions. I also used it to formulate hypotheses, suggest code, debug errors, write Markdown explanations, and improve grammar.
- **GitHub Copilot in VS Code:** Mainly helped me investigate coding errors and debug my work.
- **Gemini:** Provided alternative suggestions when I wanted to compare different coding approaches.

I treated AI-generated code and explanations as suggestions rather than automatically accepting them. I checked the suggestions against my data, code, and results.

For example, when the visualisation of the 15 most expensive cars did not look as expected, I investigated the underlying data and identified two records with the same `CarName`. This reinforced the importance of independent investigation, data validation, and critical evaluation when using AI-assisted development tools.

## Technologies

- **Programming language:** Python
- **Development environment:** Jupyter Notebook, VS Code
- **Data manipulation:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, Plotly Express
- **Statistical analysis:** SciPy
- **Machine learning:** scikit-learn
- **Version control:** Git and GitHub

## How to Run

1. Clone or download the repository.
2. Open the project in Jupyter Notebook or VS Code.
3. Install the required Python libraries using the project's `requirements.txt` file, if provided.
4. Check that the input dataset is located in the expected directory.
5. Run the notebooks in the following order:
   - Data Preparation and Cleaning
   - Exploratory Data Analysis and Statistics
   - Machine Learning and Car Price Prediction

Check that file paths match the repository structure before running the notebooks on another computer.

## Limitations and Next Steps

This project has several limitations:

- **Dataset size:** The dataset contains only 205 records, which limits the extent to which the findings can be generalised.
- **Market relevance:** The dataset may not reflect current car prices or the wider used-car market.
- **Statistical interpretation:** Correlations and statistically significant group differences indicate associations, not causation.
- **Model validation:** The reported results are based on a single train/test split and may vary with a different split.
- **Feature leakage:** Features derived directly from the target variable must be excluded from the predictors to ensure a valid evaluation.
- **Reproducibility:** The model metrics and final conclusions need to be reconciled.

My next steps are to:

1. Rerun all three notebooks from beginning to end and verify the reported results.
2. Confirm that target-derived features are excluded from the machine learning predictors.
3. Reconcile the model-comparison table with the final notebook conclusions.
4. Improve file-path portability and document the project setup.
5. Consider cross-validation and further model tuning to assess the stability of predictive performance.

## References

- [Code Institute Data Analytics README Template](https://github.com/Code-Institute-Solutions/da-README-template)
- [Code Institute Unit 2 Assessment Handbook](https://code-institute-org.github.io/5P-Assessments-Handbook/da-ai-bootcamp-unit2.html)

The original dataset source and its licence should be confirmed and acknowledged before redistributing the dataset.

---

**Final check before publishing:** Verify the model metrics, confirm that `price_per_hp` is excluded from the predictors, and check that the notebooks run successfully in the order described above.