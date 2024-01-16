import streamlit as st
from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a root `/` endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}

def main():
    st.title("Detection of AI-generated Text")
    st.markdown("Under construction")

    st.markdown("Alternatively, you can upload a text file. Please make sure that the file is in .txt format.")
    tab1, tab2 = st.tabs(["Text input", "File upload"])
    with tab1:
        user_input = st.text_input("Please enter the text you want to check for AI-generated content")

    with tab2:
        uploaded_file = st.file_uploader("Choose a file")
        # read the uploaded file
        if uploaded_file is not None:
            uploaded_file.seek(0)
            user_input = uploaded_file.read().decode("utf-8")
    
    # Display text for user
    st.markdown("You entered the following text:")
    st.markdown(f"> {user_input}")

    # Display prediction
    st.markdown("The model predicts:")


    st.markdown("To test a new input simply enter a new text or upload a new file.")

        

if __name__ == "__main__":
    main() 