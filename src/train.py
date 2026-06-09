import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_pipeline(df):
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Trained. MAE: {mean_absolute_error(y_test, y_pred):.2f}, R² Score: {r2_score(y_test, y_pred):.2f}")

    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model, f)