import streamlit as st
from main import answer_question
from prompt import prompt # Assuming prompt.py exists and contains the 'prompt' variable

# Define the available models with descriptions
AVAILABLE_MODELS_WITH_DESC = {
    "qwen3:0.6b-q8_0 (Accuracy: 80%)": "qwen3:0.6b-q8_0",
    "qwen3:4b-q8_0 (Accuracy: 98%, 8x slower)": "qwen3:4b-q8_0",
}

st.title("Question Answering with Selectable Model")

# Model selection
selected_model_desc = st.selectbox("Choose a model:", list(AVAILABLE_MODELS_WITH_DESC.keys()))
selected_model = AVAILABLE_MODELS_WITH_DESC[selected_model_desc]

# Input fields
context = st.text_area("Context:", height=200)
question = st.text_input("Question:")

# Button to get answer
if st.button("Get Answer"):
    if not context:
        st.warning("Please provide context.")
    elif not question:
        st.warning("Please provide a question.")
    else:
        with st.spinner(f"Getting answer using {selected_model}..."):
            try:
                answer = answer_question(context=context, question=question, model_name=selected_model)
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display the prompt template being used (optional)
# with st.expander("Show Prompt Template"):
#     st.text(prompt) 