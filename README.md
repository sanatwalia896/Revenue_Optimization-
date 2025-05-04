# Game Purchase Price Prediction and Optimization App

This Streamlit application loads trained machine learning models to predict and optimize in-game purchase prices based on user behavior and characteristics.

## Features

- Predicts purchase prices using a Random Forest Regressor model
- Optimizes prices to maximize revenue while keeping prices within reasonable bounds
- Interactive input form for testing individual scenarios
- Visual comparison between predicted and optimized prices
- Support for batch processing via CSV upload

## Required Files

For the application to work, you need the following files in the same directory:

- `app.py` - The Streamlit application code
- `scaler.pkl` - The trained StandardScaler model for numerical features
- `label_encoder.pkl` - The trained LabelEncoder for categorical features
- `random_forest_model.pkl` - The trained Random Forest Regressor model

These files can be in the root directory or in a `models` directory. The app will search in both locations.

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

### Troubleshooting Model Loading

If the app cannot find your model files, run the included fix script:

```bash
python fix_model_paths.py
```

Or specify the path to your models directory:

```bash
python fix_model_paths.py /path/to/your/models
```

## Usage

1. Use the sidebar to enter user and purchase information:

   - User Segment (Casual, Hardcore, Whales)
   - Item Purchased (Season Pass, Premium Weapon, etc.)
   - Purchase Type (Subscription, One-time)
   - Promo Applied (Yes, No)
   - Session Time, Level Reached, Prior Purchases
   - Install Date and Purchase Date

2. Click "Predict and Optimize Price" to get results
3. View the comparison between predicted and optimized prices
4. For batch processing, upload a CSV file with user data

## Input Features

The application uses the following input features:

### Categorical Features (Label Encoded):

- User Segment (Casual, Hardcore, Whales)
- Purchase Type (Subscription, One-time)
- Promo Applied (Yes, No)
- Item-Purchased (Season Pass, Premium Weapon, etc.)

### Numerical Features (Scaled):

- Session Time (minutes)
- Level Reached
- Prior Purchases
- Days Installed Before Last Purchase

### Date Features (Used to calculate Days Installed):

- Date of Install
- Date of Purchase

## Price Optimization

The optimization algorithm uses the `scipy.optimize.minimize` function with constraints to:

- Find prices that maximize revenue
- Keep prices within 80-110% of the predicted price
- Ensure the optimized revenue is at least 96% of the predicted revenue

## Debugging

If you encounter errors:

1. Check the Debug Info section in the app (click the expander in the sidebar)
2. Enable Debug Mode to see more detailed information about the data processing
3. Make sure your model files are correctly loaded (check paths in the sidebar)
4. Verify that the categorical options match those used during model training
