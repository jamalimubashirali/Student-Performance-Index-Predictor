# 🎓 Student Performance Index Predictor

A comprehensive machine learning web application that predicts student academic performance using multiple ML algorithms. Built with Streamlit and powered by 16 different trained models.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20App-brightgreen?style=for-the-badge)](https://student-performance-index-predictor-project.streamlit.app/)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [🚀 Live Demo](#-live-demo)
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Live Demo

🌟 **Try the app live**: [Student Performance Index Predictor](https://student-performance-index-predictor-project.streamlit.app/)

Experience all features of the application without any setup required!

## 🎯 Overview

This project aims to predict student academic performance based on various factors such as study hours, previous scores, sleep patterns, and extracurricular activities. The application provides an interactive web interface where users can:

- Make single predictions using different ML models
- Compare model performances
- Upload CSV files for batch predictions
- Download prediction results
- View study optimization tips

## ✨ Features

### 🔮 **Interactive Prediction Interface**
- **Model Selection**: Choose from 16 different trained ML models
- **Real-time Predictions**: Get instant performance predictions
- **Performance Interpretation**: Visual feedback with color-coded results
- **Input Validation**: Comprehensive error handling and validation

### 📊 **Advanced Analytics**
- **Model Comparison**: Side-by-side performance metrics comparison
- **Batch Processing**: Upload CSV files for multiple predictions
- **Performance Metrics**: R², RMSE, MAE, and MSE for all models
- **Top Model Ranking**: Automatic identification of best-performing models

### 💡 **Educational Features**
- **Study Tips**: Evidence-based recommendations for academic improvement
- **Performance Enhancement**: Guidelines for optimizing student outcomes
- **Model Insights**: Understanding which models work best for different scenarios

## 📈 Dataset

The model is trained on student performance data with the following features:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `hours_studied` | Daily study hours | Numerical | 0-24 |
| `previous_score` | Previous academic score | Numerical | 0-100 |
| `extra_curr_activities` | Participation in extracurricular activities | Categorical | Yes/No |
| `sleep_hours` | Daily sleep hours | Numerical | 0-24 |
| `simple_ques_papers_prac` | Number of sample question papers practiced | Numerical | 0-20 |

**Target Variable**: `performance_index` - Academic performance score

## 🤖 Models

The application includes 16 trained machine learning models:

### **Linear Models**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- SGD Regressor

### **Robust Models**
- Huber Regressor
- Theil-Sen Regressor
- RANSAC Regressor

### **Tree-Based Models**
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

### **Instance-Based Models**
- K-Nearest Neighbors Regressor

### **Support Vector Models**
- SVR with RBF Kernel
- SVR with Linear Kernel

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Student Performance Index Predictor"
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. **Verify model files**
   Ensure all `.joblib` files are present in the project directory:
   ```
   ├── preprocessor.joblib
   ├── LinearRegression.joblib
   ├── Ridge.joblib
   ├── Lasso.joblib
   └── ... (other model files)
   ```

## 💻 Usage

### Local Development

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8501`

### Deployment to Streamlit Cloud

✅ **Live Demo Available**: [https://student-performance-index-predictor-project.streamlit.app/](https://student-performance-index-predictor-project.streamlit.app/)

**To deploy your own version:**

1. **Push your code to GitHub** (make sure to include `requirements.txt`)
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Connect your GitHub repository**
4. **Deploy with these settings:**
   - **Main file path**: `app.py`
   - **Requirements file**: `requirements.txt` (automatically detected)
   - **Python version**: 3.8+ (recommended)

**Important**: Ensure all `.joblib` files are committed to your repository as they contain the trained models.

### Using the Interface

#### **Single Prediction**
1. Select a model from the sidebar dropdown
2. Fill in the student details form:
   - Hours studied per day
   - Previous score (%)
   - Sleep hours per day
   - Extra curricular activities (Yes/No)
   - Sample question papers practiced
3. Click "🔮 Predict Performance"
4. View the prediction result and interpretation

#### **Batch Prediction**
1. Go to the "📊 Batch Prediction" tab
2. Upload a CSV file with the required columns
3. Click "🔮 Generate Predictions"
4. Download the results as CSV

#### **Model Comparison**
1. Use the sidebar "🔍 Compare All Models" section
2. View the "🧠 Model Insights" tab for detailed performance metrics
3. Check the "🏆 Top 3 Models" ranking

## 📊 Model Performance

### Top Performing Models

| Rank | Model | R² Score | RMSE | MAE | MSE |
|------|-------|----------|------|-----|-----|
| 🥇 | Linear Regression | 0.9884 | 2.0751 | 1.6470 | 4.3059 |
| 🥇 | Ridge Regression | 0.9884 | 2.0751 | 1.6470 | 4.3060 |
| 🥇 | Bayesian Ridge | 0.9884 | 2.0751 | 1.6470 | 4.3059 |
| 🥈 | RANSAC Regressor | 0.9884 | 2.0751 | 1.6470 | 4.3059 |
| 🥈 | SGD Regressor | 0.9884 | 2.0776 | 1.6483 | 4.3164 |

### Performance Metrics Explanation
- **R² Score**: Coefficient of determination (higher is better, max = 1.0)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Square Error (lower is better)

## 📁 Project Structure

```
Student Performance Index Predictor/
│
├── 📄 app.py                           # Main Streamlit application
├── 📄 README.md                        # Project documentation
├── 📄 student_performace_predictor.ipynb  # Jupyter notebook for model training
├── 📄 export_models.py                 # Script to export trained models
│
├── 🤖 Model Files/
│   ├── preprocessor.joblib             # Data preprocessor
│   ├── LinearRegression.joblib         # Linear regression model
│   ├── Ridge.joblib                    # Ridge regression model
│   ├── Lasso.joblib                    # Lasso regression model
│   ├── ElasticNet.joblib              # Elastic Net model
│   ├── BayesianRidge.joblib           # Bayesian Ridge model
│   ├── SGDRegressor.joblib            # SGD regressor model
│   ├── HuberRegressor.joblib          # Huber regressor model
│   ├── TheilSenRegressor.joblib       # Theil-Sen regressor model
│   ├── RANSACRegressor.joblib         # RANSAC regressor model
│   ├── DecisionTreeRegressor.joblib   # Decision tree model
│   ├── RandomForestRegressor.joblib   # Random forest model
│   ├── GradientBoostingRegressor.joblib # Gradient boosting model
│   ├── AdaBoostRegressor.joblib       # AdaBoost model
│   ├── KNeighborsRegressor.joblib     # KNN regressor model
│   ├── SVR_rbf.joblib                 # SVR with RBF kernel
│   ├── SVR_linear.joblib              # SVR with linear kernel
│   ├── KernelRidge.joblib             # Kernel Ridge model
│   ├── MLPRegressor.joblib            # Multi-layer Perceptron model
│   └── PLSRegression.joblib           # PLS regression model
│
└── 📁 Data/
    └── Student_Performance.csv        # Original dataset
```

## 🎯 Key Features Explained

### **Model Selection & Performance Display**
- **Dynamic Model Loading**: Automatically detects and loads all available model files
- **Real-time Performance Metrics**: Shows accuracy metrics for the selected model
- **Interactive Comparison**: Compare all models side-by-side

### **Intelligent Input Validation**
- **Range Validation**: Ensures input values are within realistic ranges
- **Type Checking**: Validates data types for each input field
- **Error Handling**: Graceful error messages for invalid inputs

### **Professional UI/UX**
- **Responsive Design**: Works seamlessly on different screen sizes
- **Intuitive Navigation**: Clear information hierarchy and user flow
- **Visual Feedback**: Color-coded results and performance indicators

## 🔧 Customization

### Adding New Models
1. Train your model using the same preprocessor
2. Save the model as `YourModelName.joblib`
3. Add performance metrics to the `get_model_performance()` function
4. Restart the application

### Modifying Input Features
1. Update the preprocessing pipeline in the notebook
2. Retrain all models with new features
3. Update the input form in `app.py`
4. Re-export the preprocessor

## 📚 Dependencies

```python
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -am 'Add some feature'
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**


## 🙏 Acknowledgments

- **Scikit-learn**: For providing comprehensive machine learning algorithms
- **Streamlit**: For the amazing web application framework
- **Pandas & NumPy**: For efficient data manipulation and numerical computing

## 📞 Contact

**👤 Developer**: [Mubashir Ali](https://github.com/jamalimubashirali)

For questions, suggestions, or issues, please:
- Create an issue in this repository
- Connect with me on GitHub: [@jamalimubashirali](https://github.com/jamalimubashirali)

---

**⭐ If you found this project helpful, please give it a star!**

*Built with ❤️ using Python, Streamlit, and Machine Learning*
