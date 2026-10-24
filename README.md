# Responsible Car Price Analytics

## Project overview

This project develops my earlier **Car Price Predictive Analysis** work into a more user-facing data application, with a stronger focus on **ethical practice and communication in data**.

I carried forward the three-notebook workflow and the Power BI dashboard from Project 2. For Project 3, I improved and extended the machine-learning visualisations and added a Streamlit app that brings together a market overview, interactive charts, a price-prediction interface, and a section about model performance and ethics.

The project uses vehicle specifications to explore how car prices vary, investigate relationships between features and price, and estimate prices using regression models. It is intended as an educational and exploratory tool for people who want to understand the dataset and how a predictive model works. It is **not** intended to provide a professional valuation or a guaranteed current market price.

The central question for Project 3 is not only whether a model can make useful predictions, but also how those predictions should be explained, what their limitations are, and how the application can communicate them responsibly.

## Project purpose and intended audience

The intended audience includes non-technical users who want to explore car-price patterns, as well as learners or analysts who want to inspect the data preparation, statistical analysis, and model evaluation behind the dashboard.

The application aims to:

- make the main patterns in the dataset easy to explore;
- explain which vehicle characteristics are associated with price;
- compare the performance of a simple baseline and regression models;
- show that prediction errors can differ between vehicles and categories;
- communicate uncertainty, limitations, and ethical considerations in accessible language.

## Project hypotheses

The analysis investigates the following hypotheses. These are expectations to test against the available data, not assumptions that the project treats as facts.

1. **Vehicle characteristics and price:** Engine size, horsepower, and curb weight will show positive associations with price, while average MPG will show a negative association.
2. **Differences between vehicle groups:** Mean prices will differ between at least some drivetrain and body-style groups.
3. **Predictive performance:** Random Forest Regression will predict car prices more accurately than Linear Regression on the selected test set.
4. **Variation in prediction error:** Model errors will not be identical across all vehicle categories and ranges of numerical features, so overall metrics alone will not tell the whole story.

The results support the first hypothesis in this dataset: engine size, curb weight, and horsepower have strong positive correlations with price, while average MPG has a negative correlation. The Welch t-test found a statistically significant difference in mean prices between FWD and RWD vehicles, and Welch's ANOVA with Holm correction found significant differences across at least some body-style and drivetrain groups. Random Forest performed better than Linear Regression on the held-out test set. The prediction-error analysis also shows variation between categories and predictor ranges, although group sizes are small and should be considered when interpreting those differences.

These findings describe this dataset. They do not prove that a characteristic causes a price change or that the same patterns will hold in every market.

## Project structure

The project follows a staged workflow using three Jupyter notebooks, a Power BI dashboard, and a Streamlit app.

| File or component | Purpose |
|---|---|
| `jupyter_notebooks/Data_Preparing_and_Cleaning.ipynb` | Inspects, cleans, and prepares the original dataset; standardises manufacturer names; creates derived features; exports the cleaned data. |
| `jupyter_notebooks/Exploratory_Data_Analysis_Statistics.ipynb` | Examines distributions and relationships, creates visualisations, and conducts selected statistical tests. |
| `jupyter_notebooks/Machin_Learning.ipynb` | Builds and compares regression models, evaluates predictions, analyses errors, and examines feature importance. |
| `app.py` | Streamlit application with overview metrics, filters, EDA charts, a price-prediction interface, and model-performance and ethics information. |
| `car_prices_dashboard.pbix` | Power BI dashboard carried forward from Project 2. |
| `outputs/car_prices_cleaned.csv` | Cleaned dataset used by the later analysis and Streamlit app. |
| `outputs/random_forest_model.pkl` | Expected location of the saved model pipeline used by the Streamlit predictor, if included in the repository. |


## Data and preparation

The notebooks use the Car Price Assignment dataset, which contains **205 vehicle records**. The data-preparation notebook inspects dimensions, data types, missing values, duplicate records, and potential outliers. It standardises inconsistent manufacturer names, removes the redundant `car_ID` identifier, extracts a `car_model` field from `CarName`, and creates derived features including:

- `price_per_hp` — price divided by horsepower;
- `power_to_weight` — horsepower divided by curb weight;
- `avg_mpg` — the mean of city and highway MPG.

Potential outliers were investigated rather than automatically removed, because unusually expensive or high-performance vehicles may be genuine observations rather than data errors. The cleaned dataset contains 205 rows and 31 columns.

One important modelling safeguard is that `price_per_hp` is derived directly from the target variable, `price`. It must therefore be excluded from predictors when the goal is to predict price. Otherwise, the model could benefit from target leakage and produce misleadingly optimistic results. The machine-learning notebook excludes this feature and fits the categorical encoder on training data before applying it to the test data.


## Exploratory analysis and statistical testing

The exploratory analysis uses descriptive statistics and visualisations to examine price distribution, manufacturer differences, numerical relationships, vehicle categories, and potential outliers. Visualisations include histograms, box plots, bar charts, scatter plots, and a correlation heatmap. Plotly Express is used for interactive charts where appropriate.

The price distribution is positively skewed, with most observations at lower prices and a smaller number of much more expensive vehicles. The analysis reports strong positive correlations between price and:

- **Engine size:** \(r = 0.8741\)
- **Curb weight:** \(r = 0.8353\)
- **Horsepower:** \(r = 0.8081\)

Average MPG has a negative correlation with price (\(r = -0.6968\)). These numerical features are related to one another, so the correlations should not be interpreted as independent causal effects.

The statistical analysis includes:

- **Welch's independent-samples t-test** to compare mean prices for front-wheel-drive (FWD) and rear-wheel-drive (RWD) vehicles;
- **Pearson correlation tests**, with Holm correction for multiple testing;
- **Welch's ANOVA**, with Holm correction, to investigate differences in mean price across body-style, fuel-type, and drivetrain groups.

The t-test found a statistically significant difference between the FWD and RWD sample means. In this dataset, the RWD mean price was higher. After Holm correction, Welch's ANOVA found evidence of differences in mean price across at least some body-style and drivetrain groups, but not sufficient evidence of a difference between the fuel-type means. A significant ANOVA result does not identify which specific groups differ; further post-hoc comparisons would be needed for that.

Statistical significance is not the same as practical importance. These are observational comparisons, and they do not establish causation.

## Machine-learning approach and results

The target variable is `price`, and the task is regression because the target is continuous. The dataset is split into 80% training data (164 rows) and 20% test data (41 rows), using a fixed random state. Categorical features are processed with one-hot encoding. The encoder is fitted on the training data to reduce leakage from the test set.

Three models are evaluated:

- **Dummy Regressor:** predicts the mean training price and provides a baseline;
- **Linear Regression:** provides a simple linear model for comparison;
- **Random Forest Regression:** can capture nonlinear relationships and interactions between features.

The principal model-comparison output in the machine-learning notebook reports the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 6,339.80 | 8,889.04 | -0.001 |
| Linear Regression | 2,703.331 | 3,811.845 | 0.816 |
| Random Forest Regression | 1,355.177 | 1,862.830 | 0.956 |

MAE is the mean absolute difference between predicted and actual prices. RMSE gives greater weight to larger errors. R² describes how much variation in test-set prices is explained by the model relative to a mean-prediction baseline; it is **not** a percentage of individual predictions that are correct.

On this test split, Random Forest has lower MAE and RMSE and a higher R² than Linear Regression. The notebook also reports five-fold cross-validation for Random Forest with a mean R² of **0.883**, mean MAE of **1,596.85**, and mean RMSE of **2,415.11**. The cross-validation results are less strong than the single test-split result, which demonstrates why performance should not be judged from one score alone.

The feature-importance analysis identifies engine size and curb weight as the strongest predictors in the fitted Random Forest model. Feature importance shows how useful a feature was to this fitted model; it does not establish that the feature causes a price change.

The test set contains only 41 vehicles. These results are promising for this dataset, but they are not proof of performance on new models, newer market conditions, or a wider population of vehicles.

## Streamlit app

The Streamlit application is organised into four sections:

1. **Project Overview & README** — shows the number of cars, average price, median price, maximum price, average price by brand, and the price distribution.
2. **Real-Time Price Predictor** — allows users to enter or select horsepower, engine size, cylinder count, curb weight, average MPG, and drive-wheel type to request a price estimate.
3. **EDA & Dashboard Insights** — presents interactive charts for engine size versus price, horsepower versus price, price by body style, and curb weight versus price. Brand, fuel-type, and drivetrain filters apply to the overview and EDA pages.
4. **Model Performance & Ethics** — presents model evaluation metrics, explains important predictors and intended use, and discusses fairness, target leakage, privacy, and limitations.

The app uses Streamlit for the interface, Pandas for data handling, Plotly Express for interactive charts, and Joblib to load the saved model. The Power BI `.pbix` dashboard is retained from Project 2; Streamlit provides an additional interactive interface rather than replacing the earlier dashboard.

## Ethical practice, privacy, and governance

Ethical practice is the main focus of Project 3. A model can have strong evaluation metrics and still be misunderstood, used outside its intended purpose, or perform less well for some groups. The app and README should therefore explain both the usefulness and the limits of the analysis.

### Privacy and data minimisation

The model uses vehicle attributes and price rather than information about vehicle owners. Personal data is not required for the current analytical purpose, which reduces the amount of sensitive information involved. However, the data source, provenance, and reuse conditions still need to be verified. It would be too strong to claim zero privacy risk or complete legal compliance without checking the source and the actual data-processing context.

If a future version introduces owner-level or other personal data, the legal basis, purpose, retention, access controls, and applicable data-protection requirements—including GDPR where applicable—would need to be assessed before processing it.

### Fairness and potential bias

Using engineering characteristics instead of personal demographic attributes is a design choice, but it does not prove that a model is fair or free from bias. The dataset is small and its categories are unevenly represented. A model may therefore be more reliable for common vehicle types than for rare categories or unusual high-price vehicles.

The machine-learning notebook compares prediction errors across numerical ranges and across categories such as body style, fuel type, and drivetrain. These checks help reveal whether errors vary between groups. They are an initial diagnostic, not a complete fairness audit: group sizes, uncertainty, data coverage, and the intended use all matter when interpreting the results.

### Transparency and responsible interpretation

The app provides a price estimate from selected vehicle characteristics. This estimate is not a guaranteed sale price, a professional appraisal, or proof of fair market value. The dataset is small and historical, and the evaluation does not establish how the model would perform under current market conditions.

R² should be described as a model-fit metric rather than “accuracy” in the classification sense. A high R² does not mean every prediction is close to the actual price. MAE and RMSE provide additional information about error magnitude, while residual plots and group-based error analysis help show where errors occur.

### Target leakage and validation

The notebook excludes `price_per_hp`, which is derived from the target price, from the predictor set. It also fits the one-hot encoder on the training set before transforming the test set. These are important safeguards. They do not eliminate every possible source of bias or guarantee that the test results will generalise to other data.

The fixed train/test split makes the evaluation reproducible, but the test set contains only 41 observations. Cross-validation provides an additional view of model stability, and the difference between its scores and the single test-set scores is a reason to report results cautiously.

### Prediction interface and model integration

The Streamlit app loads the trained Random Forest model from `outputs/random_forest_model.pkl` to generate real-time price estimates based on vehicle specifications. A lightweight fallback error-handling method is retained defensively in the application code for robustness, but primary deployment relies fully on the serialized machine learning pipeline. Both the cleaned dataset (`outputs/car_prices_cleaned.csv`) and the saved model (`outputs/random_forest_model.pkl`) are present at the expected paths, ensuring the predictor operates directly on the trained model outputs.

### Legal and social implications

A misleadingly confident price estimate could influence a user's expectations or decisions. The application should communicate that the output is exploratory, show appropriate error information, and avoid describing a prediction as objectively fair or authoritative. If the project is expanded beyond an educational demonstration, it should undergo more extensive validation, documentation, and review of the dataset's provenance and intended use.

## Project plan and development lifecycle

The project followed these stages:

| Stage | Work undertaken | Output |
|---|---|---|
| 1. Define the question | Reuse the car-price business problem and identify the need to explain price patterns and predictions. | Project questions and hypotheses |
| 2. Prepare the data | Inspect data quality, standardise names, create features, and export the cleaned dataset. | Cleaned CSV |
| 3. Explore and test | Create visualisations, calculate correlations, compare selected groups, and perform statistical tests. | EDA and statistical notebook |
| 4. Build and evaluate models | Establish a Dummy Regressor baseline, train Linear Regression and Random Forest, and compare metrics. | Model-comparison results |
| 5. Examine model behaviour | Review actual-versus-predicted plots, residuals, feature importance, and errors across groups. | Additional model visualisations and error analysis |
| 6. Communicate the results | Retain the Power BI dashboard and build a Streamlit app with navigation, filters, charts, predictions, and an ethics section. | Interactive dashboard interfaces |
| 7. Review ethical implications | Consider privacy, bias, data leakage, uncertainty, limitations, and the risk of overclaiming model reliability. | Ethics discussion in the app and README |
| 8. Test and maintain | Re-run notebooks, check application behaviour and paths, verify units and metrics, and record changes in Git. | Reproducible project and documented future fixes |

## Testing and quality assurance

The notebooks contain model evaluation, cross-validation, visual checks, and statistical tests. I have not assumed that every application interaction has been tested successfully. The table below separates analysis already represented in the notebook outputs from checks that still need to be completed and recorded.

| Test area | Check | Status based on the supplied files |
|---|---|---|
| Data preparation | Check dataset dimensions, missing values, duplicate rows, and cleaned export. | Documented in Notebook 1; rerun end-to-end before submission. |
| Statistical analysis | Reproduce correlation tests, Welch's t-test, and Welch's ANOVA with Holm correction. | Results are recorded in Notebook 2; rerun to confirm reproducibility. |
| Model evaluation | Reproduce test-set MAE, RMSE, and R² for all three models. | Results are recorded in Notebook 3; metric labels and displayed values need reconciliation. |
| Cross-validation | Reproduce the five-fold Random Forest scores. | Results are recorded in Notebook 3; rerun to confirm. |
| Streamlit startup | Launch `app.py` and confirm all four sections load. | Needs a recorded manual test. |
| Required files | Confirm the cleaned CSV and saved model exist at the paths expected by the app. | Must be checked in the final repository. |
| Predictor | Test representative valid inputs and confirm the trained model generates a prediction. | Needs a recorded manual test. |
| Missing-model/error handling | Ensure a missing model or failed prediction is not silently replaced with a formula-based result. | Improvement needed in the current app. |
| Filters and charts | Check that filters update the overview and EDA pages and that charts remain readable. | Needs a recorded manual test. |
| Units and labels | Ensure the notebook, Power BI dashboard, and Streamlit app use consistent currency and metric descriptions. | Improvement needed; currency labels currently conflict. |
| Power BI | Open the `.pbix` file and check visuals and interactions. | Needs a recorded manual test. |

## Limitations and future improvements

The current project has several limitations:

- **Small dataset:** There are 205 vehicle records and only 41 observations in the held-out test set.
- **Market relevance:** The dataset may not represent current prices or the full range of vehicles in a wider market.
- **Uneven groups:** Some vehicle categories have relatively few observations, making group-level error estimates less stable.
- **Metric consistency:** The machine-learning notebook contains slightly different displayed metric values in different sections, and currency symbols are not consistent across all outputs.
- **Prediction fallback:** The Streamlit app can return a formula-based estimate if the saved model is missing or prediction fails.
- **Testing:** End-to-end notebook execution and interactive application checks need to be recorded before claiming that the complete project has passed all tests.
- **Data provenance:** The exact dataset source and reuse conditions should be verified and cited.

My next priorities are to standardise the currency and metrics, improve the app's model-error handling, rerun the notebooks from beginning to end, complete manual checks of the filters and predictor, verify the dataset source, and document the results. Online deployment is a possible future step after these issues have been addressed.

## Tools and technologies

- **Python** — programming language for analysis and the app.
- **Pandas and NumPy** — data manipulation and numerical calculations.
- **Matplotlib and Seaborn** — statistical and diagnostic visualisations.
- **Plotly Express** — interactive charts.
- **SciPy and Statsmodels** — statistical analysis and hypothesis testing.
- **Scikit-learn** — preprocessing, regression models, evaluation metrics, and cross-validation.
- **Jupyter Notebook and VS Code** — development environments.
- **Power BI Desktop** — dashboard carried forward from Project 2.
- **Streamlit** — interactive web-app interface.
- **Joblib** — loading the saved machine-learning pipeline.
- **Git and GitHub** — version control and project documentation.

The required library versions should be recorded in the repository's `requirements.txt` file.

## Run the project locally

These instructions assume the repository includes `app.py`, a dependency file, and the required cleaned data and saved model files. Check the repository structure before running the commands.

### 1. Clone the Project 3 repository

```bash
git clone https://github.com/nastarannaseri79-afk/Responsible-Car-Price-Analytics.git
cd Responsible-Car-Price-Analytics
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository includes `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the dependency file is not included, create and verify one listing the libraries required by the notebooks and app before expecting this command to work.

### 4. Check the expected files

The Streamlit app loads:

- `outputs/car_prices_cleaned.csv`
- `outputs/random_forest_model.pkl`

Make sure both files are present at those exact paths. If the cleaned CSV needs to be regenerated, run the data-preparation notebook first. The saved model must be created or supplied separately if it is not committed to the repository.

### 5. Launch Streamlit

```powershell
streamlit run app.py
```

Open the local address shown in the terminal. To work with the notebooks, open the `jupyter_notebooks` folder in VS Code or Jupyter and run them in this order: data preparation and cleaning, exploratory analysis and statistics, then machine learning.

## Deployment status

The Streamlit app currently runs locally in VS Code. It has not been deployed to a public hosting service, so there is no live application link yet. The Power BI dashboard is provided as a `.pbix` file and requires compatible Power BI software to open.

Before deployment, the app's fallback prediction behaviour, data paths, dependency file, metric labels, and currency labels should be corrected and tested. Deployment should only be described as complete once a public build has been launched and checked.

## Use of AI tools and acknowledgement

During development, I used AI tools as learning and problem-solving support. In my Project 2 workflow, I used ChatGPT for planning the project structure, developing hypotheses, debugging support, Markdown explanations, and grammar; GitHub Copilot in VS Code mainly for debugging; and Gemini to compare alternative suggestions.

AI-generated suggestions were treated as suggestions rather than automatically accepted answers. They needed to be checked against the data, code, and actual outputs. For example, when a chart of the most expensive cars did not look as expected, I investigated the underlying records and found duplicate `CarName` values. This reinforced the importance of checking data and assumptions instead of relying on a suggested fix without validation.

Any external code, tutorials, data sources, or other material incorporated into the project should be acknowledged in the repository. The source and licence of the car-price dataset should be verified and cited before redistribution.

## Reflection

Building on Project 2 allowed me to focus this stage on how analysis is communicated and how users interact with results. I retained the Power BI dashboard, extended the machine-learning visualisations, and added a Streamlit app that brings key findings, model evaluation, and ethical considerations together.

One of the most important lessons is that a strong model score does not automatically make a model reliable for every situation. The dataset is small, the model still makes substantial errors for some vehicles, and the current application needs safer handling when the saved model is unavailable. The currency labels and repeated metric values also need to be made consistent. Addressing these issues is part of responsible analytics: documenting what works, being open about what has not yet been verified, and making improvements before users rely on the results.

## References

- Code Institute, *Ethical Practice and Communication in Data — Assessment Handbook* (Project 3 handbook supplied with this project).
- Car-price dataset and accompanying data dictionary — verify and add the exact source citation and reuse conditions before final submission.
- Any external tutorials, code snippets, or libraries used should be acknowledged where appropriate.
