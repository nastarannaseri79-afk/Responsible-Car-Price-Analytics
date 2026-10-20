# Car Price Predictive Analysis

## Project Overview

This project investigates which vehicle characteristics are associated with car prices and how accurately these characteristics can predict price. It was developed as part of the Code Institute Data Analytics and AI Bootcamp.

I reused the car-price dataset from Project 1. Due to directory-structure issues that made committing the original project to GitHub difficult, I decided to rebuild the project from the beginning. I kept the same dataset but repeated the data preparation, analysis, and modelling workflow to improve the project structure and my understanding of the data analytics process.

## Interactive Web Application

To translate my analysis and machine learning model into a fully functional data product, I built an interactive **Streamlit web application** (`app.py`). 

* **Real-Time Price Prediction:** Users can adjust vehicle specifications (such as horsepower, engine size, and curb weight) using interactive widgets to instantly get a predicted market price.
* **Interactive Visualisations:** Explore dynamic charts that break down relationships between vehicle characteristics and prices.
* **Model Insights & Ethics:** The dashboard presents technical evaluation metrics alongside a clear summary of data governance and ethics.

## Project Questions

- How are car prices distributed?
- Which vehicle characteristics have the strongest relationship with price?
- Do prices differ between selected vehicle groups?
- How accurately can regression models predict car prices?
- Which features are most important to the best-performing model?

## Project Lifecycle and Planning (LO3)

To keep my development structured and organized, I followed an iterative project lifecycle:
1. **Ideation & Requirements:** Defined the business problem, target audience, and key performance indicators.
2. **Data Preparation (`Notebook 1`):** Inspected data quality, cleaned strings, handled outliers, and engineered features.
3. **Exploratory Analysis (`Notebook 2`):** Conducted descriptive statistics, correlation checks, and hypothesis testing.
4. **Machine Learning (`Notebook 3`):** Built preprocessing pipelines, trained baseline, linear regression, and random forest models, and evaluated test performance.
5. **Interactive Dashboard (`app.py`):** Developed a responsive Streamlit application to showcase findings and predictions.
6. **Maintenance & Version Control:** Tracked changes incrementally using Git and GitHub, documented limitations, and structured deployment steps.

## Project Workflow

The project follows a three-notebook workflow covering data preparation, exploratory data analysis, statistical testing, and machine learning.

### Notebook 1: Data Preparation and Cleaning

- **Data inspection:** Examined the dataset's dimensions, column data types, missing values, and duplicate records. Confirmed that there were no missing values or exact duplicate rows.
- **Outlier screening:** Applied the Interquartile Range (IQR) method to screen continuous variables for potential outliers. Flagged observations were retained because they could represent genuine differences between standard, high-performance, and luxury vehicles rather than data-entry errors.
- **Data cleaning:** Removed the redundant `car_ID` identifier and standardised inconsistent manufacturer names, including `maxda` to `mazda`, `porcshce` to `porsche`, `toyouta` to `toyota`, and `vokswagen` and `vw` to `volkswagen`.
- **Feature engineering:** Extracted `car_model` from `CarName` and created additional features:
  - `price_per_hp` = `price / horsepower`
  - `power_to_weight` = `horsepower / curbweight`[cite: 5, 6]
  - `avg_mpg` = `(citympg + highwaympg) / 2`[cite: 5, 6]
- **Export:** Saved the cleaned dataset to `outputs/car_prices_cleaned.csv`[cite: 5, 6].

The original dataset contains 205 rows and 26 columns[cite: 5, 6]. The cleaned dataset contains 205 rows and 31 columns[cite: 5, 6].

**Important modelling consideration:** Since `price_per_hp` is calculated directly from the target variable, `price`, it must not be used as a predictor when predicting price[cite: 5, 6]. Including it would introduce target leakage and produce misleading evaluation results[cite: 5, 6].

### Notebook 2: Exploratory Data Analysis and Statistics

- **Descriptive statistics:** Calculated measures of central tendency, variance, standard deviation, and skewness[cite: 5, 6]. The price distribution had a skewness of 1.78, indicating a positively skewed distribution[cite: 5, 6].
- **Correlation analysis:** Examined relationships between car prices and numerical features[cite: 5, 6]. The strongest reported correlations included[cite: 5, 6]:
  - `enginesize`: \(r = 0.8741\)[cite: 5, 6]
  - `curbweight`: \(r = 0.8353\)[cite: 5, 6]
  - `horsepower`: \(r = 0.8081\)[cite: 5, 6]
  - `avg_mpg`: \(r = -0.6968\)[cite: 5, 6]
- **Hypothesis testing:** Used Welch's independent-samples t-test to compare prices between front-wheel-drive (FWD) and rear-wheel-drive (RWD) vehicles[cite: 5, 6]. The result was statistically significant (\(p \approx 9.63 \times 10^{-16}\)), leading to rejection of the null hypothesis for this comparison[cite: 5, 6].
- **Group comparisons:** Used Welch's ANOVA to investigate price differences across body styles and drivewheel groups, applying Holm correction for multiple comparisons[cite: 5, 6].

These analyses helped identify important patterns in the dataset and investigate whether observed differences between vehicle groups were statistically significant[cite: 5, 6].

### Notebook 3: Machine Learning and Car Price Prediction

The objective of this notebook was to compare regression models and evaluate their ability to predict car prices on unseen data[cite: 5, 6].

- **Data splitting:** Split the dataset into an 80% training set (164 rows) and a 20% test set (41 rows)[cite: 5, 6].
- **Preprocessing pipeline:** Used `ColumnTransformer` and `OneHotEncoder` to process categorical variables[cite: 5, 6]. The preprocessing pipeline was fitted on the training data to prevent data leakage from the test set[cite: 5, 6].
- **Baseline model:** Implemented a `DummyRegressor` that predicts the mean training price, providing a baseline against which more advanced models could be evaluated[cite: 5, 6].
- **Linear Regression:** Trained an ordinary least squares regression model using the encoded feature set[cite: 5, 6].
- **Random Forest Regression:** Trained a Random Forest model to capture potentially nonlinear relationships between vehicle characteristics and price[cite: 5, 6].
- **Model evaluation:** Compared model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (\(R^2\))[cite: 5, 6].

The reported Linear Regression results were approximately MAE = 2,702, RMSE = 3,812, and \(R^2 = 0.816\)[cite: 5, 6]. The reported Random Forest results were MAE = 1,352, RMSE = 1,886, and \(R^2 = 0.955\)[cite: 5, 6].

These results are provisional until the notebooks have been rerun and the reported metrics reconciled[cite: 5, 6].

## Ethics, Privacy, and Governance (LO1)

- **Data Bias & Fairness:** To ensure fair predictions, I avoided incorporating subjective brand tiers into the model, allowing it to rely strictly on objective engineering specifications. Target-derived features like `price_per_hp` were explicitly excluded to eliminate data leakage.
- **Data Privacy & GDPR:** The project relies on public, anonymized vehicle specifications with zero Personally Identifiable Information (PII), aligning fully with privacy guidelines.
- **Social Impact:** This tool helps reduce information asymmetry for used-car buyers, though users should treat predictions as data-driven recommendations rather than absolute financial guarantees.

## Visualisation

I used different visualisation libraries depending on the analysis[cite: 5, 6]:

- **Matplotlib and Seaborn:** Used for statistical plots, price distributions, boxplots, scatter plots, and correlation heatmaps[cite: 5, 6].
- **Plotly Express:** Used to create interactive visualisations[cite: 5, 6].

One useful learning experience occurred when I plotted the 15 most expensive cars. The chart did not look as expected, so I investigated the underlying data and found two records with the same `CarName`. This highlighted the importance of checking data quality, grouping logic, and the assumptions behind a visualisation rather than relying on the chart alone[cite: 5, 6].

## Main Findings

The analysis identified several important patterns[cite: 5, 6]:

- Engine size, curb weight, and horsepower showed strong positive correlations with car prices[cite: 5, 6].
- Average fuel economy showed a negative correlation with price[cite: 5, 6].
- The analysis found a statistically significant difference in mean price between FWD and RWD vehicles in this sample[cite: 5, 6].
- Random Forest Regression achieved the best reported predictive performance among the compared models[cite: 5, 6].
- Engine size was reported as the most important feature in the fitted Random Forest model[cite: 5, 6].

### Model Comparison

The following table presents the provisional results reported in the model-comparison section of the notebook[cite: 5, 6].

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 6,339.80 | 8,889.04 | -0.001[cite: 5, 6] |
| Linear Regression | 2,702.39 | 3,812.40 | 0.816[cite: 5, 6] |
| Random Forest Regression | 1,352.20 | 1,885.51 | 0.955[cite: 5, 6] |

*Note: The model metrics differ slightly from the figures reported elsewhere in the notebook. These results must be verified by rerunning the complete workflow. The price units are those used in the original dataset[cite: 5, 6].*

MAE measures the average absolute difference between predicted and actual prices. RMSE gives greater weight to larger prediction errors, while \(R^2\) indicates how much variation in the test-set prices is explained by the model relative to a mean-prediction baseline[cite: 5, 6].

## Use of AI Tools

I used AI tools throughout the project to support my learning, problem-solving, and development process[cite: 5, 6].

- **ChatGPT:** Helped me develop a project roadmap and notebook structure based on the Code Institute assessment instructions. I also used it to formulate hypotheses, suggest code, debug errors, write Markdown explanations, and improve grammar[cite: 5, 6].
- **GitHub Copilot in VS Code:** Mainly helped me investigate coding errors and debug my work[cite: 5, 6].
- **Gemini:** Provided alternative suggestions when I wanted to compare different coding approaches[cite: 5, 6].

I treated AI-generated code and explanations as suggestions rather than automatically accepting them. I checked the suggestions against my data, code, and results[cite: 5, 6].

For example, when the visualisation of the 15 most expensive cars did not look as expected, I investigated the underlying data and identified two records with the same `CarName`. This reinforced the importance of independent investigation, data validation, and critical evaluation when using AI-assisted development tools[cite: 5, 6].

## Technologies

- **Programming language:** Python[cite: 5, 6]
- **Web Framework:** Streamlit
- **Development environment:** Jupyter Notebook, VS Code[cite: 5, 6]
- **Data manipulation:** Pandas, NumPy[cite: 5, 6]
- **Visualisation:** Matplotlib, Seaborn, Plotly Express[cite: 5, 6]
- **Statistical analysis:** SciPy[cite: 5, 6]
- **Machine learning:** scikit-learn[cite: 5, 6]
- **Version control:** Git and GitHub[cite: 5, 6]

## How to Run

1. Clone or download the repository[cite: 5, 6].
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate