import streamlit as st
from testdata_tcp import preprocess_and_generate_test_data_tcp  # Import the function for TCP Congestion use case
from testdata_ecomm import preprocess_and_generate_test_data_ecomm
from testdata_health import preprocess_and_generate_test_data_health
from testdata_stocks import preprocess_and_generate_test_data_stocks
from testdata_students import preprocess_and_generate_test_data_students

# Streamlit app code
def main():
    st.title("Automated Test Data Generation using ML")

    st.markdown("The purpose of this is to generate few samples of testdata based on the seleceted usecase. This sample data can be used for testing like Regression")
    st.caption("How to use:")
    st.caption("Step 1: select a use case for which you need a sample test data")
    st.caption("Step 2: Click on the Generate button")
    st.caption("Step 3: wait for it to generate the test data")
    st.caption("Step 4: Once the data is generated you can use it for testing")
    # Select the use case
    selected_use_case = st.selectbox("Select Use Case", ["E-commerce", "Health", "Stock Market", "Student Performance", "TCP Congestion"])  # Add other use cases as needed

     # Check if the "Generate" button is clicked
    if st.button("Generate"):
        if selected_use_case == "E-commerce":
            test_data, mse = preprocess_and_generate_test_data_ecomm()
        elif selected_use_case == "Health":
            test_data, mse = preprocess_and_generate_test_data_health()
        elif selected_use_case == "Stock Market":
            test_data, mse = preprocess_and_generate_test_data_stocks()
        elif selected_use_case == "Student Performance":
            test_data, mse = preprocess_and_generate_test_data_students()
        elif selected_use_case == "TCP Congestion":
            test_data, mse = preprocess_and_generate_test_data_tcp()
        # Display the generated test data
        st.subheader("Generated Test Data:")
        display_test_data(test_data)

        # Display the Mean Squared Error
        st.subheader("Mean Squared Error:")
        st.write(mse)

def display_test_data(test_data):
    # Format and display the test data in a table
    st.table(test_data)

if __name__ == "__main__":
    main()
