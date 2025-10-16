import streamlit as st
import requests
import json
from typing import Dict

# ---------------------------
# CONFIGURATION
# ---------------------------
API_URL = "http://localhost:8000/query"  # replace with your orchestrator API endpoint

st.set_page_config(page_title="AeroRAG Interface", page_icon="🛰️", layout="wide")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("AeroRAG Control Panel")
st.sidebar.markdown("### System Settings")

temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 512, 50)

if st.sidebar.button("Clear Conversation"):
    st.session_state.conversation = []

# ---------------------------
# MAIN APP
# ---------------------------
st.title("🛰️ NASA AeroRAG - Offline Retrieval-Augmented Generation")
st.markdown("Ask aerospace-related questions using the locally deployed LLaMA4 model and NASA STI data.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_query = st.text_area("Enter your question:", placeholder="e.g., What propulsion technologies are studied at NASA Glenn?")

if st.button("Run Query"):
    if user_query.strip():
        st.session_state.conversation.append({"role": "user", "content": user_query})

        with st.spinner("Processing your query..."):
            try:
                payload = {
                    "query": user_query,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                response = requests.post(API_URL, json=payload)
                result = response.json()
                answer = result.get("answer", "No response returned.")
                sources = result.get("sources", [])

                st.session_state.conversation.append({"role": "assistant", "content": answer})

                st.subheader("🧠 Model Response")
                st.markdown(answer)

                if sources:
                    with st.expander("🔗 Sources Used"):
                        for s in sources:
                            st.markdown(f"- {s}")

            except Exception as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a question first.")

# ---------------------------
# CHAT HISTORY
# ---------------------------
st.markdown("### 🗂️ Conversation History")
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AeroRAG:** {msg['content']}")