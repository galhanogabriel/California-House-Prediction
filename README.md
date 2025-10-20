# California House Prediction Model

Source: https://www.kaggle.com/datasets/camnugent/california-housing-prices/data

This dataset was derived from the 1990 U.S. Census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

This project focuses on predicting housing prices in California using machine learning techniques based on data from the 1990 U.S. Census. The dataset includes demographic, geographic, and housing-related attributes for each census block group — the smallest unit for which the Census Bureau publishes sample data. By analyzing features such as median income, house age, population density, and proximity to the ocean, the model aims to estimate the median house value within each area. In addition to predictive modeling, the project incorporates interactive maps to visualize regional price patterns and explore how socioeconomic and geographic factors influence housing values across California.

## Project Organization

```
├── .gitignore         <- Files and directories to be ignored by Git.
├── requirements.txt   <- The requirements file to reproduce the analysis environment.
├── LICENSE            <- Open source license (MIT).
├── README.md          <- Main README for developers using this project.
|
├── dados              <- Data files for the project.
|
├── models             <- Trained and Serialized Models, Model Predictions, or Model Summaries.
|
├── notebooks          <- Jupyter Notebooks.
│
|   └──src             <- Source code for use in this project.
|      │
|      ├── __init__.py  <- Makes it a Python module.
|      ├── config.py    <- Basic project settings.
|      ├── graficos.py  <- Scripts for Exploratory and Outcome-Oriented Visualizations.
|      └── helpers.py   <- Functions created specifically for this project.
|
├── references        <- Data dictionaries.
```

## Environment Setup

1. Clone the repository that will be created from this template.

    ```bash
    git clone git@github.com:galhanogabriel/California-House-Prediction.git
    ```

2. Create a virtual environment for your project using conda.

      ```bash
      conda env create -f ambiente.yml --name machine_learning_project
      ```

## More About the Dataset

[Click here](references/data_dictionary.md) to view the data dictionary

## Summary of Key Results

The modeling process demonstrated that using a RobustScaler for preprocessing numerical features yielded the best performance, effectively handling outliers common in housing data. Incorporating polynomial features (up to degree 3) allowed the model to capture more complex relationships without overfitting, while Ridge regularization balanced model flexibility and stability by shrinking less relevant coefficients instead of eliminating them. 

The results highlighted that median income and location (latitude and longitude) were the most influential factors in predicting house values, with geographic features often interacting with other variables. Although the Ridge model was computationally slower than simpler baselines like Dummy and Linear Regression, it achieved superior predictive metrics and demonstrated improved alignment between predicted and actual values. Further optimization could be explored, but substantial performance gains are unlikely beyond this stage.
