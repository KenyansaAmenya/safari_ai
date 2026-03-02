import streamlit as st
import sys
from pathlib import Path



current_file = Path(__file__).resolve()
app_dir = current_file.parent
project_root = app_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline import TourismRAG
from src.config import config

# Page config
st.set_page_config(
    page_title="Kenya Tourism RAG",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'sources_visible' not in st.session_state:
        st.session_state.sources_visible = False

def initialize_rag():
    """Initialize or load RAG system."""
    with st.spinner("🚀 Initializing system..."):
        try:
            rag = TourismRAG()
            rag.load_knowledge_base()
            st.session_state.rag = rag
            return True
        except FileNotFoundError:
            try:
                count = rag.build_knowledge_base()
                st.session_state.rag = rag
                return True
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
                return False
        except Exception as e:
            st.error(f"Initialization error: {e}")
            return False

def sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.title("🦁 Kenya Tourism RAG")
        st.markdown("---")
        
        # API Key check
        if not config.GROK_API_KEY or config.GROK_API_KEY == "your_grok_api_key_here":
            st.error("⚠️ Grok API Key not set!")
            api_key = st.text_input("Enter API Key:", type="password")
            if api_key:
                import os
                os.environ["GROK_API_KEY"] = api_key
                config.GROK_API_KEY = api_key
                st.success("Key set! Refreshing...")
                st.rerun()
            st.stop()
        
        # System status
        if st.session_state.rag:
            st.success("✅ System Ready")
            if st.button("🔄 Reload Knowledge Base"):
                st.session_state.rag = None
                st.rerun()
        else:
            st.warning("⏳ System not initialized")
            if st.button("🚀 Initialize System"):
                if initialize_rag():
                    st.rerun()
        
        st.markdown("---")
        st.subheader("⚙️ Filters")
        
        location_filter = st.text_input("Filter by Location:", 
                                       placeholder="e.g., Mombasa")
        category_filter = st.selectbox(
            "Category:",
            ["All", "attraction", "hotel", "restaurant", "activity"]
        )
        
        st.markdown("---")
        st.subheader(" Stats")
        if st.session_state.rag:
            stats = st.session_state.rag.get_stats()
            st.json(stats)
        
        return {
            'location': location_filter if location_filter else None,
            'category': category_filter if category_filter != "All" else None
        }

def main_interface(filters):
    """Render main interface."""
    st.markdown('<p class="main-header">🦁 Kenya Tourism Assistant</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me about destinations, hotels, '
                'activities, and travel tips in Kenya!</p>', 
                unsafe_allow_html=True)
    
    # Example questions
    with st.expander(" Example Questions"):
        cols = st.columns(2)
        examples = [
            "Best wildlife destinations in Kenya",
            "3-day itinerary in Mombasa",
            "Best time to visit Mount Kenya",
            "Budget-friendly hotels near Maasai Mara",
            "Family-friendly coastal activities",
            "What to pack for a safari?"
        ]
        
        for i, ex in enumerate(examples):
            col = cols[i % 2]
            if col.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.current_query = ex
                st.rerun()
    
    # Query input
    query = st.text_input(
        "Your question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What are the best national parks for bird watching?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_btn = st.button(" Search", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button(" Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.current_query = ""
        st.rerun()
    
    # Process query
    if search_btn and query and st.session_state.rag:
        with st.spinner(" Searching knowledge base..."):
            # Build filters
            active_filters = {}
            if filters['location']:
                active_filters['location'] = filters['location']
            if filters['category']:
                active_filters['category'] = filters['category']
            
            result = st.session_state.rag.query(
                query, 
                filters=active_filters if active_filters else None
            )
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(result['answer'])
            
            # Sources toggle
            st.session_state.sources_visible = st.toggle(
                "📚 Show Retrieved Sources", 
                value=st.session_state.sources_visible
            )
            
            if st.session_state.sources_visible:
                st.markdown("###  Sources")
                for i, doc in enumerate(result.get('retrieved_documents', []), 1):
                    meta = doc.get('metadata', {})
                    with st.container():
                        st.markdown(f"""
                        <div class="source-box">
                            <b>[{i}] {meta.get('title', 'Unknown')}</b><br>
                            <small>
                            Location: {meta.get('location', 'N/A')} | 
                            Category: {meta.get('category', 'N/A')} | 
                            Score: {doc.get('score', 0):.3f}
                            </small><br>
                            <small>{doc.get('content', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add to history
            st.session_state.history.append({
                'query': query,
                'answer': result['answer']
            })
    
    # Query history
    if st.session_state.history:
        with st.expander(" Query History"):
            for item in reversed(st.session_state.history[-5:]):
                st.markdown(f"**Q:** {item['query']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.markdown("---")

def main():
    init_session_state()
    filters = sidebar()
    
    if not st.session_state.rag:
        st.info("👈 Click 'Initialize System' in the sidebar to start")
    else:
        main_interface(filters)

if __name__ == "__main__":
    main()