import streamlit as st
from main import answer_question

st.title("Автоматическая генерация ответов на вопросы")

context = st.text_area("Введите контекст здесь:", height=200)
question = st.text_area("Введите ваш вопрос здесь:", height=100)

if st.button("Получить ответ"):
    if context and question:
        with st.spinner("Генерация ответа..."):
            try:
                answer = answer_question(context, question)
                st.subheader("Ответ:")
                st.write(answer)
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
                st.error("Пожалуйста, проверьте соединение с LLM и попробуйте снова.")
    else:
        st.warning("Пожалуйста, предоставьте и контекст, и вопрос.") 
