# Data Dictionary

Source: https://www.kaggle.com/datasets/camnugent/california-housing-prices/data

This dataset was derived from the 1990 U.S. Census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing in a home. Since the average number of rooms and bedrooms in this dataset is provided per household, these columns may show surprisingly high values for block groups with few households and many vacant homes, such as in vacation resorts.

The target variable is the median house value for California districts, expressed in dollars.

- `median_income`: median income in the block group (in tens of thousands of dollars)
- `housing_median_age`: median age of houses in the block group
- `total_rooms`: number of rooms in the block group
- `total_bedrooms`: number of bedrooms in the block group
- `population`: population of the block group
- `households`: households in the block group
- `latitude`: latitude of the block group
- `longitude`: longitude of the block group
- `ocean_proximity`: proximity to the ocean
  - `NEAR BAY`: near the bay
  - `<1H OCEAN`: less than one hour from the ocean
  - `INLAND`: inland
  - `NEAR OCEAN`: near the ocean
  - `ISLAND`: island
- `median_house_value`: median house value in the block group (in dollars)