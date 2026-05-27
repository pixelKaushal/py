import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
import io

csv_data = """Income,Employment_Type,Credit_History,Max_Loan_Allowed,Approved
45000,Salaried,Good,15000,1
120000,Self-Employed,Excellent,50000,1
30000,Unemployed,Poor,5000,0
85000,Salaried,Good,35000,1
50000,Self-Employed,Fair,18000,1
95000,Salaried,Excellent,42000,1
20000,Unemployed,Poor,2000,0
110000,Salaried,Fair,40000,0
55000,Salaried,Good,20000,1
130000,Self-Employed,Excellent,55000,1
25000,Unemployed,Poor,1000,0
80000,Salaried,Fair,28000,1
60000,Self-Employed,Good,22000,1
105000,Salaried,Excellent,45000,1
22000,Unemployed,Fair,3000,0
115000,Salaried,Poor,0,0
40000,Salaried,Fair,12000,1
140000,Self-Employed,Excellent,60000,1
15000,Unemployed,Poor,0,0
75000,Salaried,Good,30000,1
65000,Self-Employed,Fair,21000,1
90000,Salaried,Excellent,40000,1
28000,Unemployed,Poor,4000,0
125000,Salaried,Good,48000,1
48000,Salaried,Good,17000,1
115000,Self-Employed,Excellent,48000,1
32000,Unemployed,Fair,6000,0
87000,Salaried,Good,36000,1
52000,Self-Employed,Poor,10000,0
98000,Salaried,Excellent,44000,1
18000,Unemployed,Poor,1000,0
100000,Salaried,Fair,35000,1
42000,Salaried,Good,14000,1
125000,Self-Employed,Excellent,52000,1
31000,Unemployed,Poor,4500,0
82000,Salaried,Good,33000,1
58000,Self-Employed,Fair,19000,1
92000,Salaried,Excellent,41000,1
21000,Unemployed,Poor,1500,0
108000,Salaried,Poor,0,0
46000,Salaried,Fair,14500,1
135000,Self-Employed,Excellent,58000,1
27000,Unemployed,Good,8000,0
78000,Salaried,Good,31000,1
62000,Self-Employed,Fair,20000,1
96000,Salaried,Excellent,43000,1
24000,Unemployed,Poor,2500,0
112000,Salaried,Good,42000,1
50000,Salaried,Excellent,22000,1
70000,Self-Employed,Good,26000,1"""

df = pd.read_csv(io.StringIO(csv_data))


root = tk.Tk()
root.title("Credit Approval Prediction")
root.geometry("1200x800")
root.configure(bg="#13ebeb")

program_title = tk.Label(root, text="Credit Approval Prediction", font=("Arial", 24, "bold"), bg="#13ebeb")
program_title.pack(pady=20)

program_description = tk.Label(root, text="This application predicts whether a loan will be approved and the maximum loan amount based on customer data.", font=("Arial", 14), bg="#13ebeb")
program_description.pack(pady=10)

income_frame = tk.Frame(root, bg="#13ebeb")
income_frame.pack(pady=5) 

income_input_label = tk.Label(income_frame, text="Income:", font=("Arial", 12), bg="#13ebeb")
income_input_label.pack(side=tk.LEFT, padx=5)

income_input = tk.Entry(income_frame, font=("Arial", 12))
income_input.pack(side=tk.LEFT, padx=5)

emp_type_frame = tk.Frame(root, bg="#13ebeb")
emp_type_frame.pack(pady=5)
emp_type_label = tk.Label(emp_type_frame, text="Employment Type:", font=("Arial", 12), bg="#13ebeb")
emp_type_label.pack(side=tk.LEFT, padx=5)
emp_type_var = tk.StringVar()
emp_type_options = ['Salaried', 'Self-Employed', 'Unemployed']
emp_type_menu = tk.OptionMenu(emp_type_frame, emp_type_var, *emp_type_options )
emp_type_menu.config(font=("Arial", 12))
emp_type_menu.pack(side=tk.LEFT, padx=5)

credit_history_frame = tk.Frame(root, bg="#13ebeb")
credit_history_frame.pack(pady=5)
credit_history_label = tk.Label(credit_history_frame, text="Credit History:", font=("Arial", 12), bg="#13ebeb")
credit_history_label.pack(side=tk.LEFT, padx=5)
credit_history_var = tk.StringVar()
credit_history_options = ['Poor', 'Fair', 'Good', 'Excellent']
credit_history_menu = tk.OptionMenu(credit_history_frame, credit_history_var, *credit_history_options)
credit_history_menu.config(font=("Arial", 12))
credit_history_menu.pack(side=tk.LEFT, padx=5)

finalButton = tk.Button(root, text="Predict", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=lambda: predict())
finalButton.pack(pady=20)

output_label = tk.Label(root, text="", font=("Arial", 12), bg="#13ebeb")
output_label.pack(pady=10)



credit_encoder = OrdinalEncoder(categories=[['Poor', 'Fair', 'Good', 'Excellent']])
df['Credit_History'] = credit_encoder.fit_transform(df[['Credit_History']])

emp_encoder = OneHotEncoder(sparse_output=False, drop='first')
emp_encoded_array= emp_encoder.fit_transform(df[['Employment_Type']])
emp_columns = emp_encoder.get_feature_names_out(['Employment_Type'])
emp_encoded_df = pd.DataFrame(emp_encoded_array, columns=emp_columns)

df.drop('Employment_Type', axis=1, inplace=True)
df = pd.concat([df, emp_encoded_df], axis=1)

X = df.drop(['Max_Loan_Allowed','Approved'], axis=1)
y_classification = df['Approved']
y_regression = df['Max_Loan_Allowed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#KNN CLASSIFIER
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y_classification)
#linear regression
regressor = LinearRegression()
regressor.fit(X_scaled, y_regression)

def predict():
    raw_income = income_input.get()
    if not raw_income.isdigit():
        output_label.config(text="Please enter a valid numeric income!", fg="red")
        return
    new_customer = {
        'Income':[float(raw_income)],
        'Employment_Type':[emp_type_var.get()],
        'Credit_History':[credit_history_var.get()],
    }
    if not new_customer['Employment_Type'][0] or not new_customer['Credit_History'][0]:
        output_label.config(text="Please select options for all dropdowns!", fg="red")
        return
    user_df = pd.DataFrame(new_customer)
    user_df['Credit_History'] = credit_encoder.transform(user_df[['Credit_History']])

    user_emp_encoded_array = emp_encoder.transform(user_df[['Employment_Type']])
    user_emp_columns = emp_encoder.get_feature_names_out(['Employment_Type'])
    user_emp_encoded_df = pd.DataFrame(user_emp_encoded_array, columns=user_emp_columns)
    user_df.drop('Employment_Type', axis=1, inplace=True)
    user_df = pd.concat([user_df, user_emp_encoded_df], axis=1)

    prediction = knn.predict(scaler.transform(user_df))
    
    loan_amount_prediction = regressor.predict(scaler.transform(user_df))
    if(prediction[0] == 1):
        output_label.config(text=f"Approved: Yes\nMax Loan Allowed: ${loan_amount_prediction[0]:.2f}")
    elif (loan_amount_prediction[0] < 0):
        output_label.config(text=f"Approved: No\nMax Loan Allowed: ${loan_amount_prediction[0]:.2f}")
    else:
        output_label.config(text=f"Approved: No\nMax Loan Allowed: ${loan_amount_prediction[0]:.2f}")

root.mainloop()