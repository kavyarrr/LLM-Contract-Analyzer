import streamlit as st
import sys
import os
import json
import time
from datetime import datetime

# Add subfolders to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from query_and_respond import run_query_with_context

# Page configuration
st.set_page_config(
    page_title="Insurance Contract Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        color: #333;
        font-size: 1.2rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .answer-yes {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .answer-no {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .answer-unknown {
        color: #6c757d;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Make text area bigger */
    .stTextArea textarea {
        font-size: 1.1rem;
        min-height: 150px;
    }
</style>
""", unsafe_allow_html=True)

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_answer_class(answer):
    """Get CSS class based on answer"""
    if answer == "YES":
        return "answer-yes"
    elif answer == "NO":
        return "answer-no"
    else:
        return "answer-unknown"

def display_result(result):
    """Display the analysis result in an attractive format"""
    if not result:
        st.error("Unable to process the query. Please try again.")
        return
    
    try:
        # Parse the JSON result
        if isinstance(result, str):
            result_data = json.loads(result)
        else:
            result_data = result
            
        answer = result_data.get("answer", "UNKNOWN")
        justification = result_data.get("justification", "")
        source_clause = result_data.get("source_clause", "")
        confidence = result_data.get("confidence", 0.0)
        
        # Create the result display
        st.markdown("### Analysis Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("**Answer:**")
            answer_class = get_answer_class(answer)
            st.markdown(f'<div class="{answer_class}">{answer}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Confidence:**")
            confidence_class = get_confidence_class(confidence)
            st.markdown(f'<div class="{confidence_class}">{confidence:.1%}</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Source:**")
            if source_clause and source_clause != "null":
                st.info(source_clause)
            else:
                st.warning("No specific source found")
        
        # Justification - improved display
        st.markdown("### Justification")
        if justification and justification.strip():
            # Debug: show justification length
            st.text(f"Justification length: {len(justification)} characters")
            # Use st.markdown with proper HTML formatting
            st.markdown(f"""
            <div class="result-box">
                {justification.replace(chr(10), '<br>').replace(chr(13), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No justification provided")
            st.text(f"Justification value: '{justification}'")
        
    except Exception as e:
        st.error(f"Error displaying result: {e}")
        # Debug: show raw result
        st.text(f"Raw result: {result}")

def main():
    # Header
    st.markdown('<h1 class="main-header">Insurance Contract Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis of insurance policy documents</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["Mixtral-8x7B-Instruct", "Mistral-7B-Instruct"],
            index=0
        )
        
        # Number of chunks - this will be passed to the function
        num_chunks = st.slider(
            "Number of Context Chunks",
            min_value=5,
            max_value=20,
            value=15,
            help="Number of document chunks to retrieve for analysis"
        )
        
        # Temperature - this will be passed to the function
        temperature = st.slider(
            "Response Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower values = more consistent, Higher values = more creative"
        )
        
        st.markdown("---")
        st.markdown("### Question Guidelines")
        st.markdown("""
        **For best results, please ask YES/NO questions about insurance coverage.**
        
        **Examples:**
        - Does the policy cover AYUSH treatments?
        - Is cashless facility available?
        - Are maternity expenses covered?
        
        **Note:** "What" questions may return UNKNOWN answers as the system is optimized for coverage verification.
        """)
        
        st.markdown("---")
        st.markdown("### Available Documents")
        st.markdown("""
        - **Policy 1**: Global Health Care
        - **Policy 2**: Travel Insurance  
        - **Policy 3**: Edelweiss Well Mother
        - **Policy 4**: HDFC Easy Health
        - **Policy 5**: ICICI Golden Shield
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Ask Your Question")
        
        # Query input - made bigger
        query = st.text_area(
            "Enter your insurance policy question:",
            height=150,
            placeholder="Example: Does the Global Health Care policy cover AYUSH Day Care treatments?"
        )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze Policy", use_container_width=True)
    
    with col2:
        st.markdown("### Quick Stats")
        
        # Placeholder stats (you can make these dynamic)
        st.metric("Documents Indexed", "5")
        st.metric("Total Chunks", "748")
        st.metric("Avg Response Time", "~15s")
    
    # Analysis section - moved below the button
    if analyze_button and query.strip():
        st.markdown("---")
        
        # Progress bar
        with st.spinner("Analyzing your query..."):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            try:
                # Call the analysis function with the slider values
                result = run_query_with_context(query, num_chunks, temperature)
                
                # Display result
                display_result(result)
                
                # Success message
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.info("Please check your query and try again.")
    
    # Stats section - moved below analysis results
    if analyze_button and query.strip():
        st.markdown("---")
        st.markdown("### Analysis Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Query Length", f"{len(query)} chars")
        
        with col2:
            st.metric("Context Chunks", f"{num_chunks}")
        
        with col3:
            st.metric("Temperature", f"{temperature}")
        
        with col4:
            st.metric("Processing Time", "~15s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with Streamlit | Insurance Contract Analyzer v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 