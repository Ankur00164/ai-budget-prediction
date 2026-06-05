import os
import io
import json
import pickle
import sqlite3
import hashlib
import secrets
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from werkzeug.utils import secure_filename

# ─── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On Render, use /data (persistent disk). Locally use BASE_DIR.
DATA_DIR = os.environ.get('RENDER_DATA_DIR', os.path.join(BASE_DIR))
PERSISTENT_DIR = '/data' if os.path.isdir('/data') else DATA_DIR

UPLOAD_FOLDER = os.path.join(PERSISTENT_DIR, 'uploads')
REPORTS_FOLDER = os.path.join(PERSISTENT_DIR, 'reports')
MODELS_FOLDER  = os.path.join(PERSISTENT_DIR, 'models')
DB_PATH        = os.path.join(PERSISTENT_DIR, 'database.db')

# Ensure directories exist
for _d in [UPLOAD_FOLDER, REPORTS_FOLDER, MODELS_FOLDER]:
    os.makedirs(_d, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

REQUIRED_COLUMNS = [
    'Date', 'Department', 'Category', 'Region',
    'Budget Amount', 'Actual Amount', 'Payment Method', 'Transaction ID'
]

# ─── Database ───────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT    NOT NULL,
                email    TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                created  TEXT    DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                filename   TEXT    NOT NULL,
                filepath   TEXT    NOT NULL,
                rows       INTEGER,
                cols       INTEGER,
                uploaded   TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        ''')


init_db()

# ─── Helpers ────────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the uploaded dataframe."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Budget Amount', 'Actual Amount'], inplace=True)
    df['Budget Amount'] = pd.to_numeric(df['Budget Amount'], errors='coerce').fillna(0)
    df['Actual Amount'] = pd.to_numeric(df['Actual Amount'], errors='coerce').fillna(0)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Deviation'] = df['Actual Amount'] - df['Budget Amount']
    df['Deviation Percentage'] = np.where(
        df['Budget Amount'] != 0,
        (df['Deviation'] / df['Budget Amount']) * 100,
        0
    )
    df['Risk Level'] = df['Deviation Percentage'].apply(classify_risk)
    return df


def classify_risk(pct: float) -> str:
    if pct <= 5:
        return 'Low'
    elif pct <= 15:
        return 'Medium'
    return 'High'


def sanitize(obj):
    """Recursively convert numpy/pandas scalar types to native Python types
    so Flask's JSON session serialiser never chokes on int64 / float64."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_model(df: pd.DataFrame):
    """Train RandomForest and return model + metrics."""
    feature_cols = ['Department', 'Category', 'Region', 'Payment Method',
                    'Budget Amount', 'Month', 'Year']
    target_col = 'Deviation'

    work = df[feature_cols + [target_col]].dropna().copy()
    encoders = {}
    for col in ['Department', 'Category', 'Region', 'Payment Method']:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    X = work[feature_cols]
    y = work[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = round(float(mean_absolute_error(y_test, y_pred)), 2)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    r2   = round(float(r2_score(y_test, y_pred)), 4)

    # Save model
    model_path = os.path.join(MODELS_FOLDER, 'budget_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'encoders': encoders}, f)

    # Future prediction: use mean feature values
    future = X_test.copy()
    future_pred = model.predict(future)
    avg_future = round(float(np.mean(future_pred)), 2)

    return {'mae': mae, 'rmse': rmse, 'r2': r2,
            'future_deviation': avg_future, 'model_path': model_path}


# ─── Chart builders ─────────────────────────────────────────────────────────────
def build_charts(df: pd.DataFrame) -> dict:
    charts = {}

    dept = df.groupby('Department').agg(
        Budget=('Budget Amount', 'sum'),
        Actual=('Actual Amount', 'sum')
    ).reset_index()

    # 1. Budget vs Actual (Bar)
    fig1 = go.Figure(data=[
        go.Bar(name='Budget', x=dept['Department'], y=dept['Budget'],
               marker_color='#4361ee'),
        go.Bar(name='Actual', x=dept['Department'], y=dept['Actual'],
               marker_color='#f72585')
    ])
    fig1.update_layout(barmode='group', title='Budget vs Actual Amount by Department',
                       template='plotly_white', legend=dict(orientation='h', y=-0.2))
    charts['budget_vs_actual'] = pio.to_json(fig1)

    # 2. Department Wise Deviation (Bar)
    dept_dev = df.groupby('Department')['Deviation'].sum().reset_index().sort_values('Deviation', ascending=False)
    fig2 = px.bar(dept_dev, x='Department', y='Deviation',
                  color='Deviation', color_continuous_scale='RdYlGn_r',
                  title='Department Wise Total Deviation')
    fig2.update_layout(template='plotly_white')
    charts['dept_deviation'] = pio.to_json(fig2)

    # 3. Region Wise Deviation (Pie)
    region_dev = df.groupby('Region')['Deviation'].sum().abs().reset_index()
    fig3 = px.pie(region_dev, names='Region', values='Deviation',
                  title='Region-Wise Budget Deviation',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(template='plotly_white')
    charts['region_deviation'] = pio.to_json(fig3)

    # 4. Monthly Budget Trend (Line)
    monthly = df.groupby(['Year', 'Month']).agg(
        Budget=('Budget Amount', 'sum'),
        Actual=('Actual Amount', 'sum')
    ).reset_index()
    monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
    monthly.sort_values('Period', inplace=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=monthly['Period'], y=monthly['Budget'],
                              mode='lines+markers', name='Budget', line=dict(color='#4361ee', width=2)))
    fig4.add_trace(go.Scatter(x=monthly['Period'], y=monthly['Actual'],
                              mode='lines+markers', name='Actual', line=dict(color='#f72585', width=2)))
    fig4.update_layout(title='Monthly Budget Trend', template='plotly_white',
                       xaxis_title='Period', yaxis_title='Amount',
                       legend=dict(orientation='h', y=-0.3))
    charts['monthly_trend'] = pio.to_json(fig4)

    # 5. Risk Distribution (Pie)
    risk_cnt = df['Risk Level'].value_counts().reset_index()
    risk_cnt.columns = ['Risk Level', 'Count']
    colors_map = {'Low': '#06d6a0', 'Medium': '#ffd166', 'High': '#ef476f'}
    fig5 = px.pie(risk_cnt, names='Risk Level', values='Count',
                  title='Risk Distribution',
                  color='Risk Level', color_discrete_map=colors_map)
    fig5.update_layout(template='plotly_white')
    charts['risk_distribution'] = pio.to_json(fig5)

    # 6. Top 10 Departments by Deviation (Horizontal Bar)
    top10 = dept_dev.head(10)
    fig6 = px.bar(top10, y='Department', x='Deviation',
                  orientation='h', title='Top 10 Departments by Deviation',
                  color='Deviation', color_continuous_scale='Reds')
    fig6.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
    charts['top10_dept'] = pio.to_json(fig6)

    return charts


def generate_insights(df: pd.DataFrame, ml_results: dict) -> list:
    insights = []

    dept_dev = df.groupby('Department')['Deviation'].sum()
    top_dept = dept_dev.idxmax()
    insights.append({
        'icon': 'bi-building-exclamation',
        'color': 'danger',
        'title': 'Highest Overspending Department',
        'text': f"<strong>{top_dept}</strong> has the highest total deviation of "
                f"₹{dept_dev[top_dept]:,.2f}. Immediate budget review recommended."
    })

    region_dev = df.groupby('Region')['Deviation'].sum()
    top_region = region_dev.idxmax()
    insights.append({
        'icon': 'bi-geo-alt-fill',
        'color': 'warning',
        'title': 'Region with Highest Deviation',
        'text': f"<strong>{top_region}</strong> region shows the highest cumulative deviation. "
                f"Total deviation: ₹{region_dev[top_region]:,.2f}."
    })

    cat_dev = df.groupby('Category')['Actual Amount'].sum()
    top_cat = cat_dev.idxmax()
    insights.append({
        'icon': 'bi-tags-fill',
        'color': 'info',
        'title': 'Category Causing Most Overruns',
        'text': f"<strong>{top_cat}</strong> category has the highest actual spending "
                f"of ₹{cat_dev[top_cat]:,.2f}, suggesting cost control is needed."
    })

    avg_dev = round(df['Deviation Percentage'].mean(), 2)
    insights.append({
        'icon': 'bi-percent',
        'color': 'primary',
        'title': 'Average Deviation Percentage',
        'text': f"The overall average budget deviation is <strong>{avg_dev}%</strong>. "
                f"{'This is within acceptable range.' if avg_dev <= 10 else 'This exceeds the 10% threshold — action required.'}"
    })

    fd = ml_results['future_deviation']
    risk = classify_risk(abs(fd) / max(df['Budget Amount'].mean(), 1) * 100)
    insights.append({
        'icon': 'bi-robot',
        'color': 'success',
        'title': 'Predicted Future Budget Deviation',
        'text': f"ML model predicts an average future deviation of <strong>₹{fd:,.2f}</strong> "
                f"(Risk: <span class='badge bg-{'danger' if risk=='High' else 'warning text-dark' if risk=='Medium' else 'success'}'>{risk}</span>)."
    })

    return insights


# ─── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([fullname, email, password, confirm]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')

        try:
            with get_db() as conn:
                conn.execute(
                    'INSERT INTO users (fullname, email, password) VALUES (?, ?, ?)',
                    (fullname, email, hash_password(password))
                )
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        with get_db() as conn:
            user = conn.execute(
                'SELECT * FROM users WHERE email = ? AND password = ?',
                (email, hash_password(password))
            ).fetchone()

        if user:
            session['user_id']   = user['id']
            session['user_name'] = user['fullname']
            session['user_email']= user['email']
            flash(f"Welcome back, {user['fullname']}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Get latest dataset for this user if any
    dataset_info = None
    preview_html = None
    with get_db() as conn:
        ds = conn.execute(
            'SELECT * FROM datasets WHERE user_id = ? ORDER BY id DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
    if ds and os.path.exists(ds['filepath']):
        dataset_info = dict(ds)
        try:
            if ds['filepath'].endswith('.csv'):
                df = pd.read_csv(ds['filepath'])
            else:
                df = pd.read_excel(ds['filepath'])
            preview_html = df.head(10).to_html(
                classes='table table-sm table-striped table-bordered', index=False)
        except Exception:
            pass

    return render_template('dashboard.html',
                           dataset_info=dataset_info,
                           preview_html=preview_html)


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'dataset' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['dataset']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename):
        flash('Only CSV and Excel (.xlsx) files are allowed.', 'danger')
        return redirect(url_for('dashboard'))

    filename = secure_filename(file.filename)
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(session['user_id']))
    os.makedirs(user_folder, exist_ok=True)
    filepath = os.path.join(user_folder, filename)
    file.save(filepath)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error reading file: {e}', 'danger')
        return redirect(url_for('dashboard'))

    # Validate columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        os.remove(filepath)
        flash(f'Missing required columns: {", ".join(missing)}', 'danger')
        return redirect(url_for('dashboard'))

    with get_db() as conn:
        conn.execute(
            'INSERT INTO datasets (user_id, filename, filepath, rows, cols) VALUES (?, ?, ?, ?, ?)',
            (session['user_id'], filename, filepath, len(df), len(df.columns))
        )

    flash(f'Dataset uploaded successfully! {len(df)} rows × {len(df.columns)} columns.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/analyze')
@login_required
def analyze():
    with get_db() as conn:
        ds = conn.execute(
            'SELECT * FROM datasets WHERE user_id = ? ORDER BY id DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()

    if not ds:
        flash('Please upload a dataset first.', 'warning')
        return redirect(url_for('dashboard'))

    filepath = ds['filepath']
    try:
        if filepath.endswith('.csv'):
            df_raw = pd.read_csv(filepath)
        else:
            df_raw = pd.read_excel(filepath)
    except Exception as e:
        flash(f'Error reading dataset: {e}', 'danger')
        return redirect(url_for('dashboard'))

    df = process_dataframe(df_raw)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = sanitize({
        'total_records':     len(df),
        'total_departments': int(df['Department'].nunique()),
        'total_regions':     int(df['Region'].nunique()),
        'total_categories':  int(df['Category'].nunique()),
        'total_budget':      round(float(df['Budget Amount'].sum()), 2),
        'total_actual':      round(float(df['Actual Amount'].sum()), 2),
        'total_deviation':   round(float(df['Deviation'].sum()), 2),
        'avg_deviation_pct': round(float(df['Deviation Percentage'].mean()), 2),
    })

    # ── Department table ──────────────────────────────────────────────────────
    dept_table = df.groupby('Department').agg(
        Total_Budget=('Budget Amount', 'sum'),
        Total_Actual=('Actual Amount', 'sum'),
        Deviation=('Deviation', 'sum'),
    ).reset_index()
    dept_table['Deviation_Pct'] = (dept_table['Deviation'] / dept_table['Total_Budget'] * 100).round(2)
    dept_table.sort_values('Deviation', ascending=False, inplace=True)
    dept_table = dept_table.round(2)
    dept_records = sanitize(dept_table.to_dict(orient='records'))

    # ── Region table ──────────────────────────────────────────────────────────
    region_table = df.groupby('Region').agg(
        Total_Budget=('Budget Amount', 'sum'),
        Total_Actual=('Actual Amount', 'sum'),
        Deviation=('Deviation', 'sum'),
    ).reset_index().round(2)
    region_records = sanitize(region_table.to_dict(orient='records'))

    # ── Category table ────────────────────────────────────────────────────────
    cat_table = df.groupby('Category').agg(
        Total_Budget=('Budget Amount', 'sum'),
        Total_Actual=('Actual Amount', 'sum'),
        Deviation=('Deviation', 'sum'),
    ).reset_index().sort_values('Total_Actual', ascending=False).round(2)
    cat_records = sanitize(cat_table.to_dict(orient='records'))

    # ── Risk ──────────────────────────────────────────────────────────────────
    risk_counts = df['Risk Level'].value_counts().to_dict()
    risk = sanitize({
        'Low':    int(risk_counts.get('Low', 0)),
        'Medium': int(risk_counts.get('Medium', 0)),
        'High':   int(risk_counts.get('High', 0)),
    })

    # ── ML ────────────────────────────────────────────────────────────────────
    ml_results = train_model(df)

    # ── Insights ──────────────────────────────────────────────────────────────
    insights = generate_insights(df, ml_results)

    # ── Charts ────────────────────────────────────────────────────────────────
    charts = build_charts(df)

    # ── Prediction result ────────────────────────────────────────────────────
    fd   = ml_results['future_deviation']
    avg_budget = df['Budget Amount'].mean()
    fd_pct = abs(fd) / max(avg_budget, 1) * 100
    pred_risk = classify_risk(fd_pct)
    if pred_risk == 'Low':
        recommendation = 'Budget allocation appears optimal. Continue monitoring quarterly.'
    elif pred_risk == 'Medium':
        recommendation = 'Review departmental budgets and identify cost-saving opportunities.'
    else:
        recommendation = 'Immediate action required. Review spending patterns and reduce non-essential expenses.'

    prediction = {
        'future_deviation': fd,
        'risk_level':       pred_risk,
        'recommendation':   recommendation,
    }

    # ── Store analysis in session for PDF export ──────────────────────────────
    session['analysis'] = {
        'summary':      summary,
        'dept_records': dept_records[:20],
        'risk':         risk,
        'ml_results':   ml_results,
        'prediction':   prediction,
        'filename':     ds['filename'],
    }

    return render_template('analysis.html',
                           summary=summary,
                           dept_records=dept_records,
                           region_records=region_records,
                           cat_records=cat_records,
                           risk=risk,
                           ml_results=ml_results,
                           insights=insights,
                           charts=charts,
                           prediction=prediction,
                           filename=ds['filename'])


@app.route('/download_excel')
@login_required
def download_excel():
    with get_db() as conn:
        ds = conn.execute(
            'SELECT * FROM datasets WHERE user_id = ? ORDER BY id DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
    if not ds:
        flash('No dataset found.', 'warning')
        return redirect(url_for('dashboard'))

    if ds['filepath'].endswith('.csv'):
        df = pd.read_csv(ds['filepath'])
    else:
        df = pd.read_excel(ds['filepath'])

    df = process_dataframe(df)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Processed Data', index=False)

        dept = df.groupby('Department').agg(
            Total_Budget=('Budget Amount', 'sum'),
            Total_Actual=('Actual Amount', 'sum'),
            Deviation=('Deviation', 'sum'),
        ).reset_index()
        dept.to_excel(writer, sheet_name='Department Analysis', index=False)

        region = df.groupby('Region').agg(
            Total_Budget=('Budget Amount', 'sum'),
            Total_Actual=('Actual Amount', 'sum'),
            Deviation=('Deviation', 'sum'),
        ).reset_index()
        region.to_excel(writer, sheet_name='Region Analysis', index=False)

    output.seek(0)
    return send_file(output,
                     download_name='processed_budget_data.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/download_report')
@login_required
def download_report():
    analysis = session.get('analysis')
    if not analysis:
        flash('Please run analysis first.', 'warning')
        return redirect(url_for('dashboard'))

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                  fontSize=18, spaceAfter=6, textColor=colors.HexColor('#1e3a5f'))
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
                                fontSize=10, textColor=colors.grey, spaceAfter=14, alignment=TA_CENTER)
    story.append(Paragraph('AI-Based Financial Budget Deviation Prediction System', title_style))
    story.append(Paragraph(f'Analysis Report — {datetime.now().strftime("%d %B %Y %H:%M")}', sub_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#4361ee')))
    story.append(Spacer(1, 0.2*inch))

    # Dataset
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor('#4361ee'))
    story.append(Paragraph(f'Dataset: {analysis["filename"]}', h2))
    story.append(Spacer(1, 0.1*inch))

    s = analysis['summary']
    summary_data = [
        ['Metric', 'Value'],
        ['Total Records',         str(s['total_records'])],
        ['Total Departments',     str(s['total_departments'])],
        ['Total Regions',         str(s['total_regions'])],
        ['Total Categories',      str(s['total_categories'])],
        ['Total Budget Amount',   f"₹{s['total_budget']:,.2f}"],
        ['Total Actual Amount',   f"₹{s['total_actual']:,.2f}"],
        ['Total Deviation',       f"₹{s['total_deviation']:,.2f}"],
        ['Average Deviation %',   f"{s['avg_deviation_pct']}%"],
    ]
    t = Table(summary_data, colWidths=[3*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4361ee')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0c0c0')),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # Department table
    story.append(Paragraph('Department Analysis', h2))
    dept_data = [['Department', 'Total Budget', 'Total Actual', 'Deviation', 'Dev %']]
    for r in analysis['dept_records'][:15]:
        dept_data.append([
            str(r['Department']),
            f"₹{r['Total_Budget']:,.0f}",
            f"₹{r['Total_Actual']:,.0f}",
            f"₹{r['Deviation']:,.0f}",
            f"{r['Deviation_Pct']}%",
        ])
    t2 = Table(dept_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1.3*inch, 0.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f72585')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fff0f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0c0c0')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.2*inch))

    # Risk
    story.append(Paragraph('Risk Classification', h2))
    risk = analysis['risk']
    risk_data = [
        ['Risk Level', 'Count'],
        ['🟢 Low',   str(risk['Low'])],
        ['🟡 Medium', str(risk['Medium'])],
        ['🔴 High',  str(risk['High'])],
    ]
    t3 = Table(risk_data, colWidths=[3*inch, 3.5*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4361ee')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0c0c0')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.2*inch))

    # ML Results
    story.append(Paragraph('Machine Learning Model Performance', h2))
    ml = analysis['ml_results']
    ml_data = [
        ['Metric', 'Value'],
        ['Mean Absolute Error (MAE)', f"₹{ml['mae']:,.2f}"],
        ['Root Mean Square Error (RMSE)', f"₹{ml['rmse']:,.2f}"],
        ['R² Score', str(ml['r2'])],
    ]
    t4 = Table(ml_data, colWidths=[3*inch, 3.5*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4361ee')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0c0c0')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t4)
    story.append(Spacer(1, 0.15*inch))

    # Prediction
    pred = analysis['prediction']
    story.append(Paragraph('Prediction Result', h2))
    pred_text = (f"<b>Expected Future Deviation:</b> ₹{pred['future_deviation']:,.2f}<br/>"
                 f"<b>Risk Level:</b> {pred['risk_level']}<br/>"
                 f"<b>Recommendation:</b> {pred['recommendation']}")
    story.append(Paragraph(pred_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Footer
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        'AI-Based Financial Budget Deviation Prediction System | Generated by ML Analytics Engine',
        footer_style))

    doc.build(story)
    output.seek(0)
    return send_file(output,
                     download_name='budget_analysis_report.pdf',
                     as_attachment=True,
                     mimetype='application/pdf')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
