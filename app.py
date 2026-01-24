import streamlit as st
from src.complete_rag_system import create_complete_rag_system
import os

st.set_page_config(page_title="AeroRAG Interface", page_icon="🛰️", layout="wide")

# Initialize RAG system (cached)
@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)."""
    try:
        rag_system = create_complete_rag_system(
            embedding_model="all-MiniLM-L6-v2",
            vector_db_type="faiss",
            llm_model="llama3.2:latest",
        )
        
        # Load from saved embeddings
        embeddings_path = "data/embeddings/ntrs_rag"
        if os.path.exists(f"{embeddings_path}.index"):
            rag_system.load_from_saved(embeddings_path)
            return rag_system
        else:
            st.error(f"Embeddings not found at {embeddings_path}. Please build the system first.")
            return None
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        return None

# Sidebar
st.sidebar.title("AeroRAG Control Panel")
st.sidebar.markdown("### System Settings")

temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 512, 50)
num_sources = st.sidebar.slider("Number of Sources", 1, 10, 5, 1)
use_grounding = st.sidebar.checkbox("Use Hallucination Prevention", value=True)

if st.sidebar.button("Clear Conversation"):
    st.session_state.conversation = []

# Main app
st.title("🛰️ NASA AeroRAG - Offline Retrieval-Augmented Generation")
st.markdown("Ask aerospace-related questions using the locally deployed LLaMA model and NASA STI data.")

# Load RAG system
rag_system = load_rag_system()

if rag_system is None:
    st.stop()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_query = st.text_area("Enter your question:", placeholder="e.g., What propulsion technologies are studied at NASA Glenn?")

if st.button("Run Query"):
    if user_query.strip():
        st.session_state.conversation.append({"role": "user", "content": user_query})

        with st.spinner("Processing your query..."):
            try:
                response = rag_system.query(
                    user_query,
                    k=num_sources,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_grounding=use_grounding
                )
                
                answer = response.get('answer', 'No response returned.')
                sources = response.get('sources', [])
                
                st.session_state.conversation.append({"role": "assistant", "content": answer})

                st.subheader("🧠 Model Response")
                st.markdown(answer)

                if sources:
                    with st.expander(f"🔗 Sources Used ({len(sources)} documents)"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(f"**{i}. {s.get('title', 'Unknown')}**")
                            st.caption(f"Similarity: {s.get('similarity', 0):.3f} | Document ID: {s.get('document_id', 'N/A')}")
                
                # Show validation info if available
                if 'validation' in response:
                    with st.expander("✅ Response Validation"):
                        validation = response['validation']
                        st.metric("Confidence", f"{validation['confidence']:.2f}")
                        st.metric("Source Overlap", f"{validation['source_overlap']:.2f}")
                        if validation['warnings']:
                            st.warning(f"Warnings: {', '.join(validation['warnings'])}")

            except Exception as e:
                st.error(f"Query failed: {e}")
                st.exception(e)
    else:
        st.warning("Please enter a question first.")

# Conversation history
st.markdown("### 🗂️ Conversation History")
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AeroRAG:** {msg['content']}")