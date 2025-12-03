import streamlit as st
import pandas as pd
import numpy as np
import pickle

## Prediction fnction
def predict_new_data(encoder, scaler, model, new_data_df):
    df_processing = new_data_df.copy()
    
    product_map = {"low": 0, "medium": 1, "high": 2}
    df_processing["Product_importance"] = df_processing["Product_importance"].map(product_map)
    
    df_processing["Gender"] = np.where(df_processing["Gender"] == "F", 0, 1)

    cols_to_encode = ["Warehouse_block", "Mode_of_Shipment"]
    encoded_array = encoder.transform(df_processing[cols_to_encode])
    encoded_df = pd.DataFrame(encoded_array.toarray(), columns=encoder.get_feature_names_out())
    

    df_numeric = df_processing.drop(columns=cols_to_encode, errors='ignore')
    
    df_final = pd.concat([df_numeric.reset_index(drop=True), 
                          encoded_df.reset_index(drop=True)], axis=1)

    df_final = df_final[getattr(scaler, "feature_names_in_", df_final.columns)]
    
    df_final_scaled = scaler.transform(df_final)
    
    prediction = model.predict(df_final_scaled)
    
    return prediction




## Load data
df = pd.read_csv('data/Train.csv')
st.title("Route Ready")

## Load model, encoder, scaler
scaler = pickle.load(open('models/route_ready_scaler.pickle', 'rb'))
encoder = pickle.load(open('models/route_ready_encoder.pickle', 'rb')) 
model = pickle.load(open('models/route_ready_rf.pickle', 'rb'))

## Make user input UI 
warehouse_block = st.selectbox("Warehouse Block", df['Warehouse_block'].unique())
mode_of_shipment = st.selectbox("Mode of Shipment", df['Mode_of_Shipment'].unique())
product_importance = st.selectbox("Product Importance", df['Product_importance'].unique())
gender = st.selectbox("Gender", df['Gender'].unique())
customer_care_calls = st.number_input("Customer Care Calls", min_value=0, max_value=20, value=1)
customer_rating = st.number_input("Customer Rating", min_value=1, max_value=5, value=3)
cost_of_the_product = st.number_input("Cost of the Product", min_value=0, value=1000)
prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=100, value=10)
discount_offered = st.number_input("Discount Offered", min_value=0, value=100)
weight_in_gms = st.number_input("Weight in grams", min_value=1000, value=10000)

input_data = {
    "Warehouse_block": [warehouse_block],
    "Mode_of_Shipment": [mode_of_shipment],
    "Customer_care_calls": [customer_care_calls],
    "Customer_rating": [customer_rating],
    "Cost_of_the_Product": [cost_of_the_product],
    "Prior_purchases": [prior_purchases],
    "Product_importance": [product_importance],
    "Gender": [gender],
    "Discount_offered": [discount_offered],
    "Weight_in_gms": [weight_in_gms]
}

df_test = pd.DataFrame(input_data)


if st.button("Predict Delivery Timeliness"):
    prediction = predict_new_data(encoder, scaler, model, df_test)
    
    if prediction[0] == 1:
        st.success("The product is predicted to reach on time.")
        st.balloons()
    else:
        st.error("The product is predicted to be delayed.")
        st.snow()

