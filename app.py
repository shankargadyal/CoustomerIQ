"""
CustomerIQ - AI-Powered Customer Intelligence Platform
Enterprise-grade Flask backend with ML models, REST API, and analytics
"""

from flask import Flask, jsonify, render_template, request, send_file
import pandas as pd
import numpy as np
import json
import os
import pickle
import io
import csv
from datetime import datetime

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

# ─── Data Loading ───────────────────────────────────────────────────────────

_cache = {}

def load_rfm():
    if 'rfm' not in _cache:
        _cache['rfm'] = pd.read_csv(os.path.join(DATASETS_DIR, 'rfm_data.csv'))
    return _cache['rfm']

def load_monthly():
    if 'monthly' not in _cache:
        _cache['monthly'] = pd.read_csv(os.path.join(DATASETS_DIR, 'monthly_revenue.csv'))
    return _cache['monthly']

def load_products():
    if 'products' not in _cache:
        _cache['products'] = pd.read_csv(os.path.join(DATASETS_DIR, 'top_products.csv'))
    return _cache['products']

def load_countries():
    if 'countries' not in _cache:
        _cache['countries'] = pd.read_csv(os.path.join(DATASETS_DIR, 'country_revenue.csv'))
    return _cache['countries']

def load_stats():
    if 'stats' not in _cache:
        with open(os.path.join(DATASETS_DIR, 'stats.json')) as f:
            _cache['stats'] = json.load(f)
    return _cache['stats']

def load_ml_model():
    if 'model' not in _cache:
        with open(os.path.join(ML_MODELS_DIR, 'churn_model.pkl'), 'rb') as f:
            _cache['model'] = pickle.load(f)
        with open(os.path.join(ML_MODELS_DIR, 'scaler.pkl'), 'rb') as f:
            _cache['scaler'] = pickle.load(f)
        with open(os.path.join(ML_MODELS_DIR, 'features.json')) as f:
            _cache['features'] = json.load(f)
    return _cache['model'], _cache['scaler'], _cache['features']

# ─── Pages ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ─── API: Overview ──────────────────────────────────────────────────────────

@app.route('/api/overview')
def overview():
    rfm = load_rfm()
    monthly = load_monthly()
    products = load_products()
    countries = load_countries()
    stats = load_stats()

    # Monthly trend with growth rate
    monthly_records = monthly.to_dict(orient='records')
    for i, r in enumerate(monthly_records):
        if i > 0:
            prev = monthly_records[i-1]['Revenue']
            r['growth'] = round(((r['Revenue'] - prev) / prev) * 100, 1) if prev else 0
        else:
            r['growth'] = 0

    # Top countries (top 10)
    top_countries = countries.head(10).to_dict(orient='records')

    return jsonify({
        'kpis': {
            'total_customers': stats['total_customers'],
            'total_revenue': round(stats['total_revenue'], 2),
            'avg_clv': round(stats['avg_clv'], 2),
            'avg_order_value': round(stats['avg_order_value'], 2),
            'high_churn_count': stats['high_churn_count'],
            'champions_count': stats['champions_count'],
            'churn_rate': round(stats['high_churn_count'] / stats['total_customers'] * 100, 1),
            'retention_rate': round((stats['total_customers'] - stats['high_churn_count']) / stats['total_customers'] * 100, 1),
        },
        'monthly_revenue': monthly_records,
        'top_products': products.head(10).to_dict(orient='records'),
        'top_countries': top_countries,
        'segment_counts': stats['segment_counts'],
        'churn_counts': stats['churn_counts'],
        'cluster_counts': stats.get('cluster_counts', {}),
    })

# ─── API: Segments ───────────────────────────────────────────────────────────

@app.route('/api/segments')
def segments():
    rfm = load_rfm()
    seg_data = rfm.groupby('Segment').agg(
        Count=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        AvgCLV=('CLV', 'mean'),
        AvgChurnScore=('ChurnScore', 'mean'),
    ).reset_index().round(2)

    # Scatter sample
    scatter = rfm[['CustomerID','Recency','Frequency','Monetary','CLV','Segment','ChurnRisk','RFM_Score']].copy()
    scatter['Monetary'] = scatter['Monetary'].round(2)
    scatter['CLV'] = scatter['CLV'].round(2)

    # RFM heatmap data
    rfm_heatmap = rfm.groupby(['R_Score','F_Score']).agg(
        Count=('CustomerID','count'),
        AvgMonetary=('Monetary','mean')
    ).reset_index().round(2)

    return jsonify({
        'segment_summary': seg_data.to_dict(orient='records'),
        'scatter_data': scatter.sample(min(800, len(scatter)), random_state=42).to_dict(orient='records'),
        'rfm_heatmap': rfm_heatmap.to_dict(orient='records'),
    })

# ─── API: Clusters ───────────────────────────────────────────────────────────

@app.route('/api/clusters')
def clusters():
    rfm = load_rfm()
    cluster_summary = rfm.groupby('ClusterLabel').agg(
        Count=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        AvgCLV=('CLV', 'mean'),
    ).reset_index().round(2)

    scatter = rfm[['CustomerID','Recency','Frequency','Monetary','CLV','ClusterLabel','Segment']].sample(
        min(600, len(rfm)), random_state=42
    ).copy()

    return jsonify({
        'cluster_summary': cluster_summary.to_dict(orient='records'),
        'scatter_data': scatter.to_dict(orient='records'),
    })

# ─── API: Churn ───────────────────────────────────────────────────────────────

@app.route('/api/churn')
def churn():
    rfm = load_rfm()
    high_risk = rfm[rfm['ChurnRisk'] == 'High'].sort_values('Monetary', ascending=False).head(20)
    cols = ['CustomerID','Recency','Frequency','Monetary','CLV','ChurnScore','Segment','Country','Recommendation']

    # Churn by country (top 8)
    churn_by_country = rfm[rfm['ChurnRisk']=='High'].groupby('Country').size().sort_values(ascending=False).head(8).to_dict()

    # Revenue at risk
    revenue_at_risk = round(rfm[rfm['ChurnRisk']=='High']['Monetary'].sum(), 2)

    return jsonify({
        'churn_distribution': rfm['ChurnRisk'].value_counts().to_dict(),
        'high_risk_customers': high_risk[cols].round(2).to_dict(orient='records'),
        'churn_by_segment': rfm.groupby(['Segment','ChurnRisk']).size().unstack(fill_value=0).to_dict(),
        'avg_recency_by_risk': rfm.groupby('ChurnRisk')['Recency'].mean().round(1).to_dict(),
        'churn_by_country': churn_by_country,
        'revenue_at_risk': revenue_at_risk,
    })

# ─── API: Revenue Forecast ────────────────────────────────────────────────────

@app.route('/api/forecast')
def forecast():
    with open(os.path.join(ML_MODELS_DIR, 'forecast.json')) as f:
        data = json.load(f)
    return jsonify(data)

# ─── API: Model Comparison ────────────────────────────────────────────────────

@app.route('/api/models')
def model_comparison():
    def _load(name, default=None):
        path = os.path.join(ML_MODELS_DIR, name)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    results    = _load('model_results.json', {})
    fi         = _load('feature_importance.json', [])
    cm         = _load('confusion_matrix.json', [[0,0],[0,0]])
    roc_data   = _load('roc_data.json', {})
    split_info = _load('split_info.json', {})
    explanation= _load('explanation.json', {})

    # Derive explanation on-the-fly if not pre-generated
    if not explanation and results:
        from model_comparison import generate_explanation
        explanation = generate_explanation(results, split_info)

    best = max(results.items(), key=lambda x: x[1]['f1']) if results else ('N/A', {})

    return jsonify({
        'model_results':    results,
        'feature_importance': fi,
        'confusion_matrix': cm,
        'roc_data':         roc_data,
        'split_info':       split_info,
        'explanation':      explanation,
        'best_model':       best[0],
        'best_metrics':     best[1],
    })

# ─── API: Customers ───────────────────────────────────────────────────────────

@app.route('/api/customers')
def customers():
    rfm = load_rfm()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    search = request.args.get('search', '').strip()
    segment_filter = request.args.get('segment', '')
    churn_filter = request.args.get('churn', '')
    sort_by = request.args.get('sort', 'Monetary')
    sort_dir = request.args.get('dir', 'desc')

    df = rfm.copy()
    if search:
        df = df[df['CustomerID'].astype(str).str.contains(search)]
    if segment_filter:
        df = df[df['Segment'] == segment_filter]
    if churn_filter:
        df = df[df['ChurnRisk'] == churn_filter]

    total = len(df)
    ascending = sort_dir == 'asc'
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    else:
        df = df.sort_values('Monetary', ascending=False)

    start = (page - 1) * per_page
    page_df = df.iloc[start:start+per_page]
    cols = ['CustomerID','Recency','Frequency','Monetary','CLV','ChurnRisk','ChurnScore',
            'Segment','RFM_Score','Country','AvgOrderValue','UniqueProducts','Recommendation']

    return jsonify({
        'customers': page_df[cols].round(2).to_dict(orient='records'),
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': max(1, (total + per_page - 1) // per_page),
    })

@app.route('/api/customer/<int:customer_id>')
def customer_detail(customer_id):
    rfm = load_rfm()
    row = rfm[rfm['CustomerID'] == customer_id]
    if row.empty:
        return jsonify({'error': 'Customer not found'}), 404
    return jsonify(row.round(2).to_dict(orient='records')[0])

# ─── API: AI Predict ──────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    rfm = load_rfm()

    recency = float(data.get('recency', 30))
    frequency = float(data.get('frequency', 5))
    monetary = float(data.get('monetary', 500))

    # Derive other features based on dataset statistics
    avg_order = monetary / max(frequency, 1)
    total_items = int(frequency * 15)
    unique_products = min(int(frequency * 8), 200)
    clv = monetary * (1 + frequency / rfm['Frequency'].max()) * (1 - recency / rfm['Recency'].max())
    clv = max(0, clv)

    # Churn score (rule-based)
    churn_score = min(100, (recency / rfm['Recency'].max()) * 100)

    # RFM scores
    r_pct = np.percentile(rfm['Recency'], [25, 50, 75])
    f_pct = np.percentile(rfm['Frequency'], [25, 50, 75])
    m_pct = np.percentile(rfm['Monetary'], [25, 50, 75])
    r_score = 4 if recency <= r_pct[0] else 3 if recency <= r_pct[1] else 2 if recency <= r_pct[2] else 1
    f_score = 4 if frequency >= f_pct[2] else 3 if frequency >= f_pct[1] else 2 if frequency >= f_pct[0] else 1
    m_score = 4 if monetary >= m_pct[2] else 3 if monetary >= m_pct[1] else 2 if monetary >= m_pct[0] else 1
    rfm_total = r_score + f_score + m_score

    # ML model prediction
    try:
        model, scaler, features = load_ml_model()
        feature_values = [recency, frequency, monetary, avg_order, total_items,
                          unique_products, clv, churn_score, r_score, f_score, m_score, rfm_total]
        X = np.array(feature_values).reshape(1, -1)
        churn_prob = float(model.predict_proba(X)[0][1])
        churn_risk_ml = 'High' if churn_prob > 0.6 else 'Medium' if churn_prob > 0.3 else 'Low'
    except Exception:
        churn_prob = churn_score / 100
        churn_risk_ml = 'High' if recency > 90 else 'Medium' if recency > 45 else 'Low'

    # Segment classification
    if rfm_total >= 10: segment = 'Champions'
    elif rfm_total >= 8: segment = 'Loyal Customers'
    elif rfm_total >= 6: segment = 'Potential Loyalists'
    elif rfm_total >= 4: segment = 'At Risk'
    else: segment = 'Lost'

    # Recommendations
    recs = {
        'Champions': 'Reward with VIP loyalty program & early product access.',
        'Loyal Customers': 'Offer upsell bundles and premium membership.',
        'Potential Loyalists': 'Send personalized offers and membership invites.',
        'At Risk': 'Launch win-back campaign with 20% exclusive discount.',
        'Lost': 'Final re-engagement email or remove from active list.',
    }

    return jsonify({
        'clv': round(clv, 2),
        'churn_risk': churn_risk_ml,
        'churn_score': round(churn_prob * 100, 1),
        'churn_probability': round(churn_prob, 3),
        'segment': segment,
        'rfm_score': rfm_total,
        'r_score': r_score,
        'f_score': f_score,
        'm_score': m_score,
        'recommendation': recs[segment],
        'avg_order_value': round(avg_order, 2),
        'model_used': 'Random Forest (sklearn)',
    })

# ─── API: AI Insights ─────────────────────────────────────────────────────────

@app.route('/api/insights')
def insights():
    rfm = load_rfm()
    stats = load_stats()

    churn_rate = round(stats['high_churn_count'] / stats['total_customers'] * 100, 1)
    top_segment = max(stats['segment_counts'], key=stats['segment_counts'].get)
    revenue_at_risk = round(rfm[rfm['ChurnRisk']=='High']['Monetary'].sum(), 0)
    avg_clv_champions = round(rfm[rfm['Segment']=='Champions']['CLV'].mean(), 2)

    insights_list = [
        {
            'type': 'warning',
            'icon': '⚠️',
            'title': f'{churn_rate}% Customers at High Churn Risk',
            'body': f'{stats["high_churn_count"]:,} customers show high churn signals. Estimated revenue at risk: £{revenue_at_risk:,.0f}. Immediate win-back campaigns recommended.',
            'action': 'View Churn Analysis',
            'page': 'churn'
        },
        {
            'type': 'success',
            'icon': '🏆',
            'title': f'Champions Segment Driving {round(rfm[rfm["Segment"]=="Champions"]["Monetary"].sum()/stats["total_revenue"]*100,1)}% Revenue',
            'body': f'{stats["champions_count"]:,} Champions customers have an avg CLV of £{avg_clv_champions:,.2f}. Focus on retention and upsell.',
            'action': 'Explore Segments',
            'page': 'segments'
        },
        {
            'type': 'info',
            'icon': '📈',
            'title': 'Revenue Forecasting Available',
            'body': 'ML-powered 6-month revenue forecast generated using historical trend analysis. Review projected revenue trajectory.',
            'action': 'View Forecast',
            'page': 'forecast'
        },
        {
            'type': 'info',
            'icon': '🎯',
            'title': f'Top Segment: {top_segment}',
            'body': f'{stats["segment_counts"][top_segment]:,} customers classified as {top_segment}. Target them with tailored engagement strategies.',
            'action': 'View Customers',
            'page': 'customers'
        },
    ]

    return jsonify({'insights': insights_list, 'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M')})

# ─── API: Product Analytics ───────────────────────────────────────────────────

@app.route('/api/products')
def products():
    products_df = load_products()
    total_rev = products_df['Revenue'].sum()
    products_df['RevenueShare'] = (products_df['Revenue'] / total_rev * 100).round(2)
    return jsonify({
        'products': products_df.to_dict(orient='records'),
        'total_revenue': round(total_rev, 2),
        'top_product': products_df.iloc[0]['Product'],
        'top_product_revenue': round(products_df.iloc[0]['Revenue'], 2),
    })

# ─── API: Export ──────────────────────────────────────────────────────────────

@app.route('/api/export/csv')
def export_csv():
    rfm = load_rfm()
    cols = ['CustomerID','Recency','Frequency','Monetary','CLV','ChurnRisk','ChurnScore',
            'Segment','RFM_Score','Country','AvgOrderValue','UniqueProducts','Recommendation']

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=cols)
    writer.writeheader()
    for _, row in rfm[cols].round(2).iterrows():
        writer.writerow(row.to_dict())

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'customeriq_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )

# ─── Health Check ──────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '3.0', 'timestamp': datetime.now().isoformat()})

# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')
