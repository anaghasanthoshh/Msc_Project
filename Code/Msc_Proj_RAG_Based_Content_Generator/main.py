import streamlit as st

# Page Title
st.title("🛍️ AI Chatbot - Product Recommendations")

# Initialize chat history in session state
if "product_chat" not in st.session_state:
    st.session_state.product_chat = []

# Display previous chat history
for msg in st.session_state.product_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input for chatbot
product_input = st.chat_input("Ask me about products...")

if product_input:
    # Store user message
    st.session_state.product_chat.append({"role": "user", "content": product_input})

    # Call RAG model here
    recommended_products = [f"Product {i+1}" for i in range(3)]  # Placeholder recommendations

    # Bot response
    bot_response = f"Here are some recommended products: {', '.join(recommended_products)}"

    # Store bot response
    st.session_state.product_chat.append({"role": "assistant", "content": bot_response})

    # Display bot response
    with st.chat_message("assistant"):
        st.write(bot_response)