
import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Layered Market Mix Modeling")

# Initialize session state
if 'trained_skus' not in st.session_state:
    st.session_state.trained_skus = []
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False

st.header("1. Upload Your Data")

uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # --- Data Preview ---
    st.subheader("Data Preview")
    try:
        # Read the file into a pandas DataFrame
        if uploaded_file.name.endswith('.csv'):
            df_preview = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        else:
            df_preview = pd.read_excel(uploaded_file)
        
        st.dataframe(df_preview.head(20)) # Display the first 20 rows
    except Exception as e:
        st.error(f"Could not read or preview the file. Error: {e}")

    # --- Model Training ---
    if st.button("Process Data and Train Models"):
        # Rewind the file buffer to be read again by the API
        uploaded_file.seek(0)
        with st.spinner("Processing data and training models..."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            response = requests.post("http://127.0.0.1:8000/upload_and_process/", files=files)

            if response.status_code == 200:
                st.success("Processing and training complete!")
                results = response.json()
                st.session_state.trained_skus = results.get('trained_skus', [])
                st.session_state.data_uploaded = True
            else:
                st.error(f"An error occurred: {response.text}")
                st.session_state.data_uploaded = False

if st.session_state.data_uploaded:
    st.header("2. Overall Portfolio Analysis")
    summary_response = requests.get("http://127.0.0.1:8000/get_all_sku_summary/")
    if summary_response.status_code == 200:
        summary_data = summary_response.json()
        summary_df = pd.DataFrame(summary_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Sales by SKU")
            fig_sales = px.pie(summary_df, values='Sales', names='SKU', title='Sales Distribution', hole=.3)
            st.plotly_chart(fig_sales, use_container_width=True)
        with col2:
            st.subheader("Total Spend by SKU")
            spend_df = summary_df.melt(id_vars=['SKU'], value_vars=['TV_Spend', 'Digital_Spend', 'Print_Spend'], var_name='Channel', value_name='Spend')
            fig_spend = px.sunburst(spend_df, path=['SKU', 'Channel'], values='Spend', title='Marketing Spend Breakdown')
            st.plotly_chart(fig_spend, use_container_width=True)

    st.header("3. SKU-Specific Analysis & Simulation")
    
    # Use a text input for SKU selection
    sku_input = st.text_input("Enter SKU for Analysis (e.g., SKU_001)", "")

    if sku_input:
        # Validate that the entered SKU is one of the trained SKUs
        if sku_input in st.session_state.trained_skus:
            selected_sku = sku_input
            # --- Historical Data Visualization ---
            st.subheader(f"Historical Performance for {selected_sku}")
            hist_response = requests.get(f"http://127.0.0.1:8000/get_historical_data/{selected_sku}")
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                hist_df = pd.DataFrame(hist_data)
                if 'Date' in hist_df.columns:
                    hist_df['Date'] = pd.to_datetime(hist_df['Date'])

                col1, col2 = st.columns(2)
                with col1:
                    if 'Date' in hist_df.columns and 'Sales' in hist_df.columns:
                        fig_hist_sales = px.line(hist_df, x='Date', y='Sales', title='Historical Sales Trend')
                        st.plotly_chart(fig_hist_sales, use_container_width=True)
                with col2:
                    spend_cols = ['TV_Spend', 'Digital_Spend', 'Print_Spend']
                    hist_spend_df = hist_df[spend_cols].sum().reset_index()
                    hist_spend_df.columns = ['Channel', 'Spend']
                    fig_hist_spend = px.pie(hist_spend_df, values='Spend', names='Channel', title='Historical Spend Mix', hole=.3)
                    st.plotly_chart(fig_hist_spend, use_container_width=True)

            # --- Interactive Prediction ---
            st.subheader(f"Simulate Sales")
            avg_spend_response = requests.get(f"http://127.0.0.1:8000/get_average_spend/{selected_sku}")
            avg_spend = avg_spend_response.json().get('average_spend', {})

            sim_tv_spend = st.slider("TV Spend", 0, 500000, int(avg_spend.get('TV_Spend', 100000)))
            sim_digital_spend = st.slider("Digital Spend", 0, 500000, int(avg_spend.get('Digital_Spend', 50000)))
            sim_print_spend = st.slider("Print Spend", 0, 500000, int(avg_spend.get('Print_Spend', 20000)))

            # --- Dynamic Prediction Call ---
            spend_data = {'sku': selected_sku, 'spend_data': {'TV_Spend': sim_tv_spend, 'Digital_Spend': sim_digital_spend, 'Print_Spend': sim_print_spend}}
            prediction_response = requests.post("http://127.0.0.1:8000/predict/", json=spend_data)

            if prediction_response.status_code == 200:
                prediction = prediction_response.json()['prediction_breakdown']
                total_sales = prediction['total_predicted_sales']
                base_sales = prediction['base_sales']
                channel_contributions = prediction['channel_contributions']

                st.metric(label="Total Predicted Sales", value=f"${total_sales:,.2f}")

                # Contribution Chart
                contrib_data = {'Base Sales': base_sales, **channel_contributions}
                contrib_df = pd.DataFrame(list(contrib_data.items()), columns=['Component', 'Sales'])
                
                fig_contrib = px.bar(contrib_df, x='Component', y='Sales', title='Predicted Sales Contribution Breakdown', color='Component')
                st.plotly_chart(fig_contrib, use_container_width=True)

                st.subheader(f"Explanation for {selected_sku} Analysis")
                st.write(f"""
                This section provides a detailed breakdown of the predicted sales for **{selected_sku}** based on the current marketing spend simulation.

                -   **Total Predicted Sales (${total_sales:,.2f}):** This is the overall sales forecast for {selected_sku} given the specified TV, Digital, and Print spend levels.
                -   **Base Sales (${base_sales:,.2f}):** This represents the estimated sales for {selected_sku} that would occur even without any marketing spend. It's the foundational sales volume.
                -   **Channel Contributions:** These figures indicate the incremental sales generated by each marketing channel (TV, Digital, Print) beyond the base sales.
                    -   **TV Spend:** ${channel_contributions.get('TV_Spend', 0):,.2f}
                    -   **Digital Spend:** ${channel_contributions.get('Digital_Spend', 0):,.2f}
                    -   **Print Spend:** ${channel_contributions.get('Print_Spend', 0):,.2f}

                The sum of Base Sales and individual Channel Contributions should approximately equal the Total Predicted Sales, illustrating how each component contributes to the overall forecast.
                """)
            else:
                # Display a placeholder or error if prediction fails
                st.error(f"Could not retrieve prediction: {prediction_response.text}")
        else:
            st.warning("Invalid SKU. Please enter a valid SKU from the list above.")
