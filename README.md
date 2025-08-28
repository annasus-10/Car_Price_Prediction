# 🚗 Car Price Prediction  

This repository contains my machine learning project for predicting car prices.  

I went through **data cleaning, feature selection, model training, evaluation, and finally deployment** as a web app.  

---

## 📂 Project Structure
```

Car\_Price/
│
├── app/                 # Streamlit web app
│   ├── app.py           # main app
│   └── model.pkl        # trained RandomForestRegressor model
│
├── A1.ipynb             # Jupyter notebook (EDA, preprocessing, training)
├── Cars.csv             # dataset
└── README.md

```

---

## 🛠️ Steps I did

### 1. Data Preparation & Cleaning
- Dropped `LPG` and `CNG` cars (different mileage units).  
- Removed “Test Drive Car” entries (very unrealistic prices).  
- Converted text fields into numeric:
  - `engine` → removed "CC"  
  - `max_power` → removed "bhp"  
  - `mileage` → removed "kmpl" (though not used in final features)  
- Mapped categorical values into numbers:  
  - **fuel:** Petrol = 1, Diesel = 2  
  - **transmission:** Manual = 1, Automatic = 2  
  - **seller_type:** Individual = 1, Dealer = 2, Trustmark Dealer = 3  

### 2. Feature Selection
- Checked correlation with selling price.  
- Selected **6 features**:  
```

['year', 'fuel', 'seller\_type', 'transmission', 'engine', 'max\_power']

````

### 3. Model Training
- Applied log-transform on target (`selling_price`) to stabilize variance.  
- Trained a `RandomForestRegressor (n_estimators=15, random_state=1)`.  
- Evaluation metrics (from notebook):  
- **MSE:** 1.71e10  
- **RMSE:** ~130,991  
- **R²:** ~0.975  

### 4. Deployment Attempts
- First tried **Dash** (as suggested in the assignment).  
- Faced multiple errors with:  
- pickle loading (custom classes)  
- port conflicts on Windows (8050 reserved)  
- mismatch between training preprocessing and app preprocessing  
- After many retries and confusion, I switched to **Streamlit** as a last resort.  
- It was easier to build a working UI.  
- Required fewer lines of code.  
- Allowed me to run the app locally with fewer errors.  

### 5. Final Web App (Streamlit)
- Located in `app/app.py`.  
- Loads my trained model (`model.pkl`).  
- Lets the user input the **6 features** directly (numeric only).  
- Applies category mappings internally.  
- Returns predicted price in rupees (after `np.exp()` since the model was trained on log price).  

Run locally with:
```bash
cd Car_Price
.\.venv\Scripts\activate   # activate your venv
python -m streamlit run app/app.py
````

---

## 📊 Example Prediction

* Input:

  ```
  year = 2015
  fuel = Petrol
  seller_type = Individual
  transmission = Manual
  engine = 1248
  max_power = 75
  ```
* Output:

  ```
  Estimated Price: 598,904 
  ```

---

## 🙋 Notes

* This project was primarily a **learning exercise**.
* I had **no prior experience with deployment**, so I tried Dash first (lots of errors), then went with Streamlit because it was the simplest way to get a working local web app.
* Streamlit may not be “production-ready”, but it demonstrates the idea clearly.


