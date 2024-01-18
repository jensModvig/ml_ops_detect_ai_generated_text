import streamlit as st

from cloud_functions import *

def main():
    st.title("Detection of AI-generated Text")
    st.markdown("Under construction")

    if "model" not in st.session_state:
        # Load in model
        st.markdown("The model is currently being loaded. This might take up to a minute.")
        model = get_model()
        # put the model in session state for streamlit
        st.session_state["model"] = model
        st.rerun()

    st.markdown("The model is loaded.")

    tab1, tab2 = st.tabs(["Text input", "File upload"])
    valid_input = False
    with tab1:
        user_input = st.text_input("Please enter the text you want to check for AI-generated content")
        if user_input:
            valid_input = True

    with tab2:
        uploaded_file = st.file_uploader("Choose a file. Please make sure that the file is in .txt format.")
        # read the uploaded file
        if uploaded_file is not None:
            # Check if file is in .txt format
            if uploaded_file.type != "text/plain":
                st.error("Please upload a text file in .txt format")
                st.stop()
            uploaded_file.seek(0)
            user_input = uploaded_file.read().decode("utf-8")
            valid_input = True

    if not valid_input:
        st.stop()
    
    # Display text for user
    st.markdown("You entered the following text:")
    st.markdown(f"> {user_input}")

    # Display prediction
    st.header("The model predicts:")
    pred = app_predict(user_input, st.session_state["model"])[0]
    if pred == 0:
        st.subheader("The text is **not** AI-generated.")
    else:
        st.subheader("The text is AI-generated.")

    st.markdown("To test a new input simply enter a new text or upload a new file.")
    st.markdown("If you uploaded a file, please remove it before giving a text input.")
if __name__ == "__main__":
    main() 