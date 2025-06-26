import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures




df = pd.read_csv(r"C:\Users\ahmed\Documents\Machine Learning\assignment1\notebook\NY-House-Dataset.csv", index_col=1)

#Drop columns
columns_to_drop = [
    'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2',
    'LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS','BROKERTITLE'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')



#Remove duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
df.drop_duplicates(inplace=True)

#Check for nulls
print(df.isnull().sum())

#Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)



# POLYNOMIAL REGRESSION
numericdata= df[["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE", "PRICE"]]

X = numericdata.iloc[:, :-1].values
Y = numericdata.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)





#starting with deg = 2
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

reg = LinearRegression()

poly = PolynomialFeatures(degree=2)
X_train_Poly = poly.fit_transform(X_train_scaler)
X_test_Poly = poly.transform(X_test_scaler)



poly.fit(X_train_Poly, y_train)
reg.fit(X_train_Poly, y_train)



# Sort the values for a clean curve
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
X_plot_poly = poly.transform(X_plot_scaled)
y_plot = lin.predict(X_plot_poly)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data points')  # Original data
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial regression (degree=2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression Fit (Degree = 2)')
plt.legend()
plt.grid(True)
plt.show()





numericdata= df[["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE", "PRICE"]]

corelation=numericdata.corr()





