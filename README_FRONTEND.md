# 📋 Insurance Contract Analyzer - Streamlit Frontend

A modern, user-friendly web interface for analyzing insurance policy documents using AI.

## 🚀 Features

### ✨ Modern UI/UX
- **Clean, professional design** with intuitive navigation
- **Responsive layout** that works on desktop and mobile
- **Real-time progress indicators** during analysis
- **Color-coded results** based on confidence levels

### 🔧 Configurable Settings
- **Model Selection**: Choose between different LLM models
- **Context Chunks**: Adjust number of document chunks (5-20)
- **Temperature Control**: Fine-tune response creativity (0.0-1.0)

### 📊 Smart Results Display
- **Answer Classification**: YES/NO/UNKNOWN with color coding
- **Confidence Scoring**: Visual confidence indicators
- **Source Attribution**: Direct references to policy documents
- **Detailed Justifications**: Clear explanations with context

## 🎯 How to Use

### 1. Start the Application
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

### 2. Navigate the Interface
- **Sidebar**: Configure settings and view available documents
- **Main Area**: Enter your insurance policy questions
- **Results**: View detailed analysis with confidence scores

### 3. Example Queries
- "Does the Global Health Care policy cover AYUSH Day Care treatments?"
- "Is cashless facility available under the Golden Shield plan?"
- "Are maternity expenses covered under the ICICI Golden Shield policy?"
- "Does the HDFC Easy Health plan cover kidney failure requiring regular dialysis?"

## 🎨 Design Features

### Color-Coded Results
- **🟢 High Confidence (80%+)**: Green
- **🟡 Medium Confidence (50-79%)**: Yellow  
- **🔴 Low Confidence (<50%)**: Red

### Answer Types
- **✅ YES**: Green (Covered)
- **❌ NO**: Red (Not Covered)
- **❓ UNKNOWN**: Gray (Insufficient Information)

### Interactive Elements
- **Progress bars** during analysis
- **Hover effects** on buttons
- **Responsive metrics** display
- **Real-time feedback**

## 📈 Performance Metrics

The interface displays:
- **Documents Indexed**: 5 insurance policies
- **Total Chunks**: 748 semantic chunks
- **Average Response Time**: ~15 seconds
- **Model**: Mixtral-8x7B-Instruct

## 🔧 Technical Details

### Backend Integration
- **Vector Search**: FAISS with sentence transformers
- **Re-ranking**: Cross-encoder for improved relevance
- **LLM**: Together AI API with Mixtral model
- **JSON Parsing**: Robust extraction with fallbacks

### Frontend Stack
- **Framework**: Streamlit 1.48.0
- **Styling**: Custom CSS with modern design
- **Components**: Native Streamlit widgets
- **Responsive**: Mobile-friendly layout

## 🎯 Use Cases

### For Insurance Professionals
- **Quick Policy Checks**: Verify coverage details
- **Client Consultations**: Real-time policy analysis
- **Document Review**: Efficient policy interpretation

### For Policy Holders
- **Coverage Verification**: Understand what's covered
- **Claim Preparation**: Know what to expect
- **Policy Comparison**: Compare different plans

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

2. **Set Environment Variables**:
   ```bash
   # Create .env file with your API key
   TOGETHER_API_KEY=your_api_key_here
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the Interface**:
   - Open your browser to `http://localhost:8501`
   - Start asking insurance policy questions!

## 📱 Mobile Experience

The interface is fully responsive and works great on:
- **Desktop browsers** (Chrome, Firefox, Safari, Edge)
- **Mobile devices** (iOS Safari, Android Chrome)
- **Tablets** (iPad, Android tablets)

## 🔒 Security & Privacy

- **No data storage**: Queries are processed in real-time
- **Secure API calls**: HTTPS encryption for all requests
- **Local processing**: Document embeddings stored locally
- **No personal data**: No user information is collected

## 🎨 Customization

### Styling
The interface uses custom CSS for a professional look:
- **Color scheme**: Blue-based professional theme
- **Typography**: Clean, readable fonts
- **Spacing**: Consistent padding and margins
- **Animations**: Smooth hover effects and transitions

### Configuration
Easily modify settings in the sidebar:
- **Model parameters**: Temperature, chunk count
- **Display options**: Result formatting
- **Performance**: Response time optimization

---

**Built with ❤️ using Streamlit | Insurance Contract Analyzer v2.0** 