# AI-Based Financial Budget Deviation Prediction System 
A modern, interactive Flask web application that empowers finance teams to clean, analyze, and visualize 
corporate budget deviations (Actual vs. Budgeted spending). Built with machine learning models and 
dynamic visualizations, the system classifies budget risks and predicts future deviation levels to guide 
financial forecasting and audits. --- 
##         
Features - **Secure Authentication System**: User registration, login, and secure password hashing using SHA-256 
with SQLite as database storage. - **Dynamic File Upload & Parsing**: Supports CSV and Excel (`.xlsx`) files. Automatically validates 
structure, columns, and data integrity. - **Detailed Financial Analytics**: - Aggregates metrics (Total Budget, Actual Cost, Deviation, and Deviation Percentage) grouped by 
**Department**, **Category**, and **Region**. - Auto-calculated deviation percentages with corresponding risk tiers: -   
**Low Risk (≤ 5%):** Budget allocation is optimal. -   -   
**Medium Risk (5% - 15%):** Minor overruns, monitor closely. 
**High Risk (> 15%):** Significant overspending, immediate audit required. - **Interactive Visualizations (Plotly)**: 
1. **Budget vs. Actual by Department** (Grouped Bar Chart) 
2. **Total Budget Deviation by Department** (Color-scaled Bar Chart) 
3. **Region-Wise Budget Deviation** (Pie Chart) 
4. **Monthly Budget Trend** (Multi-line Chart) 
5. **Risk Distribution** (Pie Chart representing data volume in each risk tier) 
6. **Top 10 Departments by Deviation** (Horizontal Bar Chart) - **AI-Powered Deviation Prediction**: Trains a Random Forest Regressor on features including 
Department, Category, Region, Payment Method, and Budget Amount to forecast expected deviations. 
Evaluates model performance using standard metrics (Mean Absolute Error, Root Mean Squared Error, and 
$R^2$ Score). - **Automated Financial Recommendations**: Real-time actionable insights based on predictive risk level, 
detailing suggestions like cost review, quarterly monitoring, or immediate spending control. - **Export Capabilities**: - **Excel Export**: Download preprocessed details alongside Department and Region summaries as a 
multi-sheet Excel file. - **PDF Export**: Generate a corporate-styled, professional PDF report (powered by ReportLab) 
containing summary metrics, tables, risk distributions, and model performance. --- 
##      
Tech Stack - **Core Framework**: Flask (Python) - **Database**: SQLite3 - **Data Engineering**: Pandas, NumPy - **Machine Learning**: Scikit-Learn (Random Forest Regressor, LabelEncoder, train_test_split) - **Charts & Visuals**: Plotly - **Reporting**: ReportLab (PDF formatting) - **Frontend UI**: Bootstrap 5 + custom premium styling (style.css, main.js) 
--- 
##    
```text 
Project Structure 
AI_Budget_Prediction/ 
├── app.py                 
    # Core Flask web application & API handlers 
├── generate_sample_data.py    # CLI script to generate mock financial datasets (.csv / .xlsx) 
├── requirements.txt           
# Python library dependencies 
├── render.yaml               
├── Procfile                   
├── database.db                
├── templates/                
│   ├── index.html             
│   ├── login.html             
│   ├── register.html          
│   ├── dashboard.html      
│   └── analysis.html          
├── static/                   
│   ├── css/ 
│   │   └── style.css     
│   ├── js/ 
│   │   └── main.js            
│   └── images/                
 # Deploy configuration for Render PaaS 
# Process configuration for Gunicorn 
# SQLite database (auto-generated at runtime) 
 # Jinja2 HTML views 
# Landing / welcome page 
# Sign-in page 
# Sign-up page 
   # User dashboard (File upload, status check, previews) 
# Analytical cockpit (Visualizations, ML scores, tables) 
 # Asset directory 
     # Premium stylesheet (Glassmorphism, animations, colors) 
# Custom interactive scripts and Plotly handlers 
# Static image assets 
└── uploads/, reports/, models/ # Auto-created folders for storing files, reports, and models 
``` --- 
##       
Upload Data Specifications 
Uploaded CSV or Excel files **must** contain the following headers (order-independent): 
| Column Name | Description | Example Value | 
| :--- | :--- | :--- | 
| `Date` | Transaction date (YYYY-MM-DD format preferred) | `2024-05-18` | 
| `Department` | Department responsible for transaction | `Marketing` | 
| `Category` | Category of expense | `Software` | 
| `Region` | Regional center | `North` | 
| `Budget Amount` | Projected budget allocated | `75000` | 
| `Actual Amount` | Real spending incurred | `82000` | 
| `Payment Method` | Method used to complete payment | `Credit Card` | 
| `Transaction ID` | Alphanumeric transaction key | `TXN0128` | --- 
##     
Installation & Local Setup 
Follow these steps to run the application locally on your machine: 
### 1. Prerequisites 
Ensure you have **Python 3.8+** installed on your system. 
### 2. Clone the Repository 
```bash 
git clone https://github.com/your-username/AI_Budget_Prediction.git 
cd AI_Budget_Prediction 
``` 
### 3. Create a Virtual Environment & Install Dependencies 
```bash 
# Create virtual environment 
python -m venv venv 
# Activate virtual environment 
# On Windows (CMD/PowerShell): 
venv\Scripts\activate 
# On Linux/macOS: 
source venv/bin/activate 
# Install required packages 
pip install -r requirements.txt 
``` 
### 4. Generate Sample Test Data (Optional) 
If you don't have a dataset ready, run the helper script to create realistic dummy datasets containing 500 
records: 
```bash 
python generate_sample_data.py 
``` 
This generates `sample_budget_data.csv` and `sample_budget_data.xlsx` inside the root folder. 
### 5. Launch the Server 
Start the Flask application local server: 
```bash 
python app.py 
``` 
The server will boot by default on **`http://127.0.0.1:5000`**. Open this URL in your web browser. --- 
##    
Deployment on Render 
This project contains configuration ready for deploying on [Render](https://render.com). 
1. Push your repository to GitHub. 
2. Sign in to Render and click **New > Blueprint**. 
3. Connect your GitHub repository. Render will automatically parse the `render.yaml` file. 
4. The service is defined as a Python web service with a **1 GB Persistent Disk** mounted at `/data` to store 
the database, uploaded datasets, and generated reports across deployments. 
5. Create and enjoy your live production instance! 
