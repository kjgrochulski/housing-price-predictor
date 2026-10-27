# Melbourne House Price Predictor

**Live App:** [House Price Predictor on Streamlit](https://kjg123-hub-housing-price-predictor-app-nro78q.streamlit.app/#property-details)

<p align="center">
  <img width="500" height="310" alt="melb_app_ss1" src="https://github.com/user-attachments/assets/292a3696-d487-4ead-9989-dff2a464fd86" />
</p>

A machine learning web app that estimates residential property sale prices in Melbourne, Australia. Users enter a street address and property details to get a price estimate with an explanation of the key factors driving it.

---

## What It Does

1. User enters a Melbourne street address, the app geocodes  via OpenStreetMap Nominatim to extract latitude, longitude, suburb, postcode, and distance from the Melbourne Central Business District (CBD).
2. Users fill other parameters: property type, bedrooms, bathrooms, car spaces, and plot size.

<p align="center">
  <img width="500" height="335" alt="image" src="https://github.com/user-attachments/assets/180e0fac-1d3d-4cbc-a1e9-e7737d753098" />
</p>

3. LightGBM model predicts a sale price, and SHAP values explain which features pushed the estimate and by how much.

<p align="center">
  <img width="500" height="335" alt="image" src="https://github.com/user-attachments/assets/2efd9cfe-9726-4430-acc2-3a7a00e1723e" />
</p>

---

## Model

| | |
|---|---|
| **Algorithm** | LightGBM (gradient boosting) |
| **Target** | log1p(Price) — predictions are transformed back to AUD |
| **Training data** | 13,580 Melbourne property sales |
| **Test R²** | 0.87 |
| **Test MAE** | ~$155,000 (approx. 17% of median sale price) |
| **Cross-validated R²** | 0.85 (5-fold) |

The model was selected after benchmarking against XGBoost, Gradient Boosting, Random Forest, Extra Trees, Decision Tree, and ElasticNet. LightGBM and XGBoost performed very similarly; LightGBM was chosen for deployment.

**Note**
During testing phase, the number of car spaces was often not publicly available for many properties, and the current model would be skewed slightly in the case that the user defaults the value to 0 or leaves it at 1. To resolve this, the input box was edited to allow a None state, so if the user leaves it blank, it will pass np.nan into input_df for the Car column instead of an integer. In this case, LightGBM will treat it as a true "Unknown" rather than "Zero parking."

### Features Used by Importance

<img width="800" height="590" alt="image" src="https://github.com/user-attachments/assets/03e14f40-9798-4783-938f-68ac395af3ac" />


Location signals (latitude, longitude, distance, suburb, postcode) collectively account for roughly 57% of the model's predictive weight.

---

## Dataset

**Source:** [Melbourne Housing Snapshot — Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)

The dataset contains 13,580 residential property sales with 21 features. Sales in the dataset were recorded between **January 2016 and September 2017**.

> **Important:** This data is from 2016–2017. Melbourne property prices have changed significantly since 2017. Estimates produced by this app reflect market conditions at the time of training, not today's market.

---

## Known Limitations

**Data age.** The training data is from 2016–2017. The model has no awareness of market movements since then.

**Missing building area.** Building area (internal floor space) was excluded from the model because 47.5% of rows in the dataset had no value for it. Including it with median imputation offered only a marginal accuracy gain while creating unreliable behaviour for users entering different values. Instead, plot size is used as a proxy, which is unfortunately not a perfect substitute.

**Accuracy range.** A test MAE of ~$155,000 means predictions can realistically be off by $100,000–$250,000 on an individual property. The model performs best for typical houses and units in well-represented suburbs; it is less reliable for unusual properties (very large land, heritage homes, luxury market) and suburbs with few training examples.

**Car spaces instability.** Because building area is absent from the model, car spaces imperfectly carry some of its signal. In some suburbs, the training data shows counterintuitive price patterns by car space count, which can cause unexpected swings when this input is edited.

**Suburb coverage.** The model was trained on 314 Melbourne suburbs. For addresses in suburbs not seen during training, the model falls back to ordinal encoding for unknown categories, which slightly reduces accuracy.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | LightGBM via scikit-learn Pipeline |
| Preprocessing | scikit-learn ColumnTransformer (OrdinalEncoder + SimpleImputer) |
| Explainability | SHAP TreeExplainer |
| Geocoding | geopy / OpenStreetMap Nominatim |
| App | Streamlit, pandas, NumPy, joblib |
| Deployment | Streamlit Cloud |

---

## Repo Structure

```
├── app.py                  # Streamlit app
├── melbourne_model.pkl     # Trained LightGBM pipeline
├── model_features.json     # Feature list expected by the model
├── melb_data.csv           # Source dataset
├── melbourne.ipynb         # Training notebook (EDA, feature selection, model comparison)
├── requirements.txt        # App dependencies
├── requirements-dev.txt    # Adds notebook dependencies
├── .devcontainer/          # Codespaces config
├── .gitignore
└── LICENSE
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To run the training notebook, install `requirements-dev.txt` instead.

Requires `melbourne_model.pkl` and `model_features.json` in the same directory as `app.py`.

---

## License

The code is released under the [MIT License](LICENSE). The dataset comes from Kaggle and has its own terms; see the dataset page linked above.
