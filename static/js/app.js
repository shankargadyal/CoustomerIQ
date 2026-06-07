/**
 * CustomerIQ – Enterprise Dashboard JS
 * Handles navigation, API calls, Chart.js rendering, ML features
 */

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  currentPage: 'overview',
  customerPage: 1,
  customerSearch: '',
  customerSegment: '',
  customerChurn: '',
  customerSort: 'Monetary',
  customerDir: 'desc',
  darkMode: true,
};

// ─── Chart defaults ──────────────────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'DM Sans', sans-serif";

const CHART_COLORS = {
  blue: 'rgba(59,130,246,0.85)',
  cyan: 'rgba(6,182,212,0.85)',
  green: 'rgba(16,185,129,0.85)',
  yellow: 'rgba(245,158,11,0.85)',
  red: 'rgba(239,68,68,0.85)',
  purple: 'rgba(139,92,246,0.85)',
  blueFill: 'rgba(59,130,246,0.12)',
  cyanFill: 'rgba(6,182,212,0.12)',
};

const SEG_COLORS = {
  'Champions': '#f59e0b',
  'Loyal Customers': '#10b981',
  'Potential Loyalists': '#3b82f6',
  'At Risk': '#f97316',
  'Lost': '#ef4444',
};

const CHURN_COLORS = { 'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444' };
const CLUSTER_COLORS = { 'VIP': '#f59e0b', 'High Value': '#10b981', 'Active Mid': '#3b82f6', 'Dormant Low': '#94a3b8' };

const charts = {};
let pageDataCache = {};

// ─── Utilities ───────────────────────────────────────────────────────────────

function fmt(n, prefix = '') {
  if (n === null || n === undefined) return '—';
  if (n >= 1_000_000) return prefix + (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return prefix + (n / 1_000).toFixed(1) + 'K';
  return prefix + Number(n).toLocaleString('en-GB', { maximumFractionDigits: 2 });
}

function fmtCurrency(n) { return '£' + fmt(n); }

function churnBadge(risk) {
  const cls = { High: 'danger', Medium: 'warning', Low: 'success' };
  return `<span class="badge badge-${cls[risk] || 'blue'}">${risk}</span>`;
}

function segBadge(seg) {
  const cls = { 'Champions': 'warning', 'Loyal Customers': 'success', 'Potential Loyalists': 'blue', 'At Risk': 'purple', 'Lost': 'danger' };
  return `<span class="badge badge-${cls[seg] || 'blue'}">${seg}</span>`;
}

function showToast(msg, type = 'info') {
  const icons = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  document.querySelector('.toast-container').appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function animateNumber(el, target, prefix = '', suffix = '') {
  const start = 0;
  const duration = 900;
  const t0 = performance.now();
  const isFloat = String(target).includes('.');
  function step(ts) {
    const progress = Math.min((ts - t0) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = start + (target - start) * ease;
    el.textContent = prefix + (isFloat ? val.toFixed(1) : Math.round(val).toLocaleString('en-GB')) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

// ─── Navigation ──────────────────────────────────────────────────────────────

function navigate(page) {
  if (state.currentPage === page) return;
  state.currentPage = page;

  // Hide all pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Show target page
  const pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.classList.add('active');
  const navEl = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (navEl) navEl.classList.add('active');

  // Update breadcrumb
  const titles = {
    overview: 'Overview', segments: 'Customer Segments', clusters: 'ML Clusters',
    churn: 'Churn Risk', customers: 'All Customers', predictor: 'AI Predictor',
    forecast: 'Revenue Forecast', models: 'Model Comparison', insights: 'AI Insights',
    products: 'Product Analytics', settings: 'Settings',
  };
  document.getElementById('page-title').textContent = titles[page] || page;

  // Load page data
  loadPage(page);

  // Close sidebar on mobile
  if (window.innerWidth <= 900) {
    document.getElementById('sidebar').classList.remove('open');
  }
}

function loadPage(page) {
  if (pageDataCache[page] && page !== 'customers') return; // Use cache except customers (has filters)
  switch (page) {
    case 'overview':  fetchOverview();  break;
    case 'segments':  fetchSegments();  break;
    case 'clusters':  fetchClusters();  break;
    case 'churn':     fetchChurn();     break;
    case 'customers': fetchCustomers(); break;
    case 'forecast':  fetchForecast();  break;
    case 'models':    fetchModels();    break;
    case 'insights':  fetchInsights();  break;
    case 'products':  fetchProducts();  break;
  }
  if (page !== 'customers') pageDataCache[page] = true;
}

// ─── Overview Page ────────────────────────────────────────────────────────────

async function fetchOverview() {
  const res = await fetch('/api/overview');
  const d = await res.json();

  // KPIs
  animateNumber(document.getElementById('kpi-customers'), d.kpis.total_customers);
  animateNumber(document.getElementById('kpi-revenue'), d.kpis.total_revenue, '£');
  animateNumber(document.getElementById('kpi-clv'), d.kpis.avg_clv, '£');
  animateNumber(document.getElementById('kpi-aov'), d.kpis.avg_order_value, '£');
  animateNumber(document.getElementById('kpi-churn-rate'), d.kpis.churn_rate, '', '%');
  animateNumber(document.getElementById('kpi-retention'), d.kpis.retention_rate, '', '%');
  document.getElementById('churn-badge').textContent = d.kpis.high_churn_count;

  // Monthly revenue chart
  renderMonthlyRevChart(d.monthly_revenue);

  // Segment donut
  renderSegmentDonut(d.segment_counts);

  // Churn donut
  renderChurnDonut(d.churn_counts);

  // Country bars
  renderCountryBars(d.top_countries);

  // Top products table
  renderTopProducts(d.top_products);
}

function renderMonthlyRevChart(monthly) {
  destroyChart('monthlyRev');
  const ctx = document.getElementById('chart-monthly-rev');
  if (!ctx) return;

  const labels = monthly.map(r => r.Month);
  const data = monthly.map(r => r.Revenue);

  charts.monthlyRev = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Revenue',
        data,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.08)',
        borderWidth: 2.5,
        pointRadius: 4,
        pointBackgroundColor: '#3b82f6',
        pointBorderColor: '#080c14',
        pointBorderWidth: 2,
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: '#0d1321',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleFont: { family: 'DM Sans', size: 12 },
        bodyFont: { family: 'Space Mono', size: 12 },
        callbacks: { label: ctx => '  £' + ctx.raw.toLocaleString('en-GB', {minimumFractionDigits:2}) },
      }},
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { family: 'Space Mono', size: 11 }, callback: v => '£' + (v/1000).toFixed(0) + 'K' } },
      },
    },
  });
}

function renderSegmentDonut(counts) {
  destroyChart('segDonut');
  const ctx = document.getElementById('chart-segment-donut');
  if (!ctx) return;

  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const colors = labels.map(l => SEG_COLORS[l] || '#94a3b8');

  charts.segDonut = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderColor: '#080c14', borderWidth: 3 }] },
    options: {
      responsive: true,
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#0d1321', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 },
      },
    },
  });

  // Legend
  const legend = document.getElementById('segment-legend');
  if (legend) {
    const total = values.reduce((a,b) => a+b, 0);
    legend.innerHTML = labels.map((l,i) => `
      <div class="legend-row">
        <div class="legend-dot" style="background:${colors[i]}"></div>
        <span>${l}</span>
        <span class="legend-val">${values[i].toLocaleString()}</span>
      </div>
    `).join('');
  }
}

function renderChurnDonut(counts) {
  destroyChart('churnDonut');
  const ctx = document.getElementById('chart-churn-donut');
  if (!ctx) return;

  const labels = ['Low', 'Medium', 'High'];
  const values = labels.map(l => counts[l] || 0);
  const colors = labels.map(l => CHURN_COLORS[l]);

  charts.churnDonut = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderColor: '#080c14', borderWidth: 3 }] },
    options: {
      responsive: true,
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#0d1321', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 },
      },
    },
  });

  const legend = document.getElementById('churn-legend');
  if (legend) {
    const total = values.reduce((a,b) => a+b, 0);
    legend.innerHTML = labels.map((l,i) => `
      <div class="legend-row">
        <div class="legend-dot" style="background:${colors[i]}"></div>
        <span>${l} Risk</span>
        <span class="legend-val">${((values[i]/total)*100).toFixed(1)}%</span>
      </div>
    `).join('');
  }
}

function renderCountryBars(countries) {
  const el = document.getElementById('country-bars');
  if (!el || !countries) return;
  const max = countries[0].Revenue;
  el.innerHTML = countries.slice(0, 8).map(c => `
    <div class="country-bar-row">
      <div class="country-name">${c.Country}</div>
      <div class="country-bar-bg"><div class="country-bar-fill" style="width:${(c.Revenue/max*100).toFixed(1)}%"></div></div>
      <div class="country-rev">${fmtCurrency(c.Revenue)}</div>
    </div>
  `).join('');
}

function renderTopProducts(products) {
  const tbody = document.getElementById('top-products-tbody');
  if (!tbody) return;
  tbody.innerHTML = products.slice(0,8).map((p, i) => `
    <tr>
      <td><span style="color:var(--text-muted);font-family:'Space Mono',monospace;font-size:11px;">#${i+1}</span></td>
      <td>${p.Product}</td>
      <td class="td-mono">${fmtCurrency(p.Revenue)}</td>
    </tr>
  `).join('');
}

// ─── Segments Page ─────────────────────────────────────────────────────────────

async function fetchSegments() {
  const res = await fetch('/api/segments');
  const d = await res.json();

  // Segment table
  const tbody = document.getElementById('segment-table-body');
  if (tbody) {
    tbody.innerHTML = d.segment_summary.map(s => `
      <tr>
        <td>${segBadge(s.Segment)}</td>
        <td class="td-mono">${s.Count.toLocaleString()}</td>
        <td class="td-mono">${s.AvgRecency}d</td>
        <td class="td-mono">${s.AvgFrequency}</td>
        <td class="td-mono">${fmtCurrency(s.AvgMonetary)}</td>
        <td class="td-mono">${fmtCurrency(s.AvgCLV)}</td>
      </tr>
    `).join('');
  }

  // Scatter chart
  renderSegmentScatter(d.scatter_data);

  // Bar chart: avg CLV by segment
  renderSegmentCLVBar(d.segment_summary);
}

function renderSegmentScatter(data) {
  destroyChart('segScatter');
  const ctx = document.getElementById('chart-seg-scatter');
  if (!ctx) return;

  const segments = [...new Set(data.map(d => d.Segment))];
  const datasets = segments.map(seg => {
    const pts = data.filter(d => d.Segment === seg);
    return {
      label: seg,
      data: pts.map(d => ({ x: d.Recency, y: d.Monetary, r: Math.min(Math.max(d.Frequency, 1.5), 10) })),
      backgroundColor: (SEG_COLORS[seg] || '#94a3b8') + '88',
      borderColor: SEG_COLORS[seg] || '#94a3b8',
      borderWidth: 1,
    };
  });

  charts.segScatter = new Chart(ctx, {
    type: 'bubble',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { font: { size: 11 }, padding: 12 } }, tooltip: {
        callbacks: {
          label: ctx => {
            const d = ctx.raw;
            return ` Recency:${d.x}d  Revenue:£${d.y.toLocaleString()}  Freq:${d.r.toFixed(1)}`;
          }
        }
      }},
      scales: {
        x: { title: { display: true, text: 'Recency (days)', color: '#64748b', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { title: { display: true, text: 'Revenue (£)', color: '#64748b', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

function renderSegmentCLVBar(data) {
  destroyChart('segCLVBar');
  const ctx = document.getElementById('chart-seg-clv');
  if (!ctx) return;
  charts.segCLVBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(s => s.Segment),
      datasets: [{
        label: 'Avg CLV (£)',
        data: data.map(s => s.AvgCLV),
        backgroundColor: data.map(s => (SEG_COLORS[s.Segment] || '#3b82f6') + 'bb'),
        borderColor: data.map(s => SEG_COLORS[s.Segment] || '#3b82f6'),
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => '  CLV: £' + ctx.raw.toLocaleString('en-GB', {minimumFractionDigits:2}) }
      }},
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { family: 'Space Mono', size: 11 }, callback: v => '£' + (v/1000).toFixed(0)+'K' } },
      },
    },
  });
}

// ─── Clusters Page ────────────────────────────────────────────────────────────

async function fetchClusters() {
  const res = await fetch('/api/clusters');
  const d = await res.json();

  // Cluster cards
  const grid = document.getElementById('cluster-cards-grid');
  if (grid) {
    grid.innerHTML = d.cluster_summary.map(c => `
      <div class="model-card">
        <div class="model-name" style="color:${CLUSTER_COLORS[c.ClusterLabel] || '#94a3b8'}">${c.ClusterLabel}</div>
        <div class="metric-row"><span class="metric-label">Customers</span><span class="metric-value">${c.Count.toLocaleString()}</span></div>
        <div class="metric-row"><span class="metric-label">Avg Recency</span><span class="metric-value">${c.AvgRecency}d</span></div>
        <div class="metric-row"><span class="metric-label">Avg Revenue</span><span class="metric-value">${fmtCurrency(c.AvgMonetary)}</span></div>
        <div class="metric-row"><span class="metric-label">Avg CLV</span><span class="metric-value">${fmtCurrency(c.AvgCLV)}</span></div>
        <div class="metric-row"><span class="metric-label">Avg Frequency</span><span class="metric-value">${c.AvgFrequency}</span></div>
      </div>
    `).join('');
  }

  // Cluster scatter
  renderClusterScatter(d.scatter_data);

  // Cluster bar
  renderClusterBar(d.cluster_summary);
}

function renderClusterScatter(data) {
  destroyChart('clusterScatter');
  const ctx = document.getElementById('chart-cluster-scatter');
  if (!ctx) return;
  const clusters = [...new Set(data.map(d => d.ClusterLabel))];
  const datasets = clusters.map(cl => ({
    label: cl,
    data: data.filter(d => d.ClusterLabel === cl).map(d => ({ x: d.Recency, y: d.Monetary })),
    backgroundColor: (CLUSTER_COLORS[cl] || '#94a3b8') + '88',
    borderColor: CLUSTER_COLORS[cl] || '#94a3b8',
    borderWidth: 1,
    pointRadius: 4,
  }));
  charts.clusterScatter = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { font: { size: 11 }, padding: 12 } } },
      scales: {
        x: { title: { display: true, text: 'Recency (days)', color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { title: { display: true, text: 'Revenue (£)', color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

function renderClusterBar(data) {
  destroyChart('clusterBar');
  const ctx = document.getElementById('chart-cluster-bar');
  if (!ctx) return;
  charts.clusterBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(c => c.ClusterLabel),
      datasets: [{
        label: 'Customers',
        data: data.map(c => c.Count),
        backgroundColor: data.map(c => (CLUSTER_COLORS[c.ClusterLabel] || '#94a3b8') + 'bb'),
        borderColor: data.map(c => CLUSTER_COLORS[c.ClusterLabel] || '#94a3b8'),
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

// ─── Churn Page ───────────────────────────────────────────────────────────────

async function fetchChurn() {
  const res = await fetch('/api/churn');
  const d = await res.json();

  // Churn bar chart
  renderChurnBarChart(d.churn_by_segment);

  // Revenue at risk
  const rar = document.getElementById('churn-revenue-risk');
  if (rar) rar.textContent = fmtCurrency(d.revenue_at_risk);

  // High risk table
  const tbody = document.getElementById('churn-table-body');
  if (tbody) {
    tbody.innerHTML = d.high_risk_customers.map(c => `
      <tr>
        <td class="td-mono">${c.CustomerID}</td>
        <td>${segBadge(c.Segment)}</td>
        <td class="td-mono">${c.Recency}d</td>
        <td class="td-mono">${c.Frequency}</td>
        <td class="td-mono">${fmtCurrency(c.Monetary)}</td>
        <td class="td-mono">${fmtCurrency(c.CLV)}</td>
        <td>
          <div style="display:flex;align-items:center;gap:8px;">
            <div class="progress-bar" style="width:60px;">
              <div class="progress-fill" style="width:${c.ChurnScore}%;background:var(--danger)"></div>
            </div>
            <span style="font-size:11px;font-family:'Space Mono',monospace">${c.ChurnScore}%</span>
          </div>
        </td>
        <td style="font-size:11px;color:var(--text-muted);max-width:200px">${c.Recommendation}</td>
      </tr>
    `).join('');
  }

  // Country churn bar
  renderChurnCountryBar(d.churn_by_country);
}

function renderChurnBarChart(churnBySegment) {
  destroyChart('churnBar');
  const ctx = document.getElementById('chart-churn-bar');
  if (!ctx) return;
  const segments = Object.keys(churnBySegment['High'] || {});
  charts.churnBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: segments,
      datasets: ['Low','Medium','High'].map(risk => ({
        label: risk + ' Risk',
        data: segments.map(s => (churnBySegment[risk] || {})[s] || 0),
        backgroundColor: CHURN_COLORS[risk] + 'bb',
        borderColor: CHURN_COLORS[risk],
        borderWidth: 1,
        borderRadius: 4,
      })),
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { font: { size: 11 }, padding: 12 } } },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

function renderChurnCountryBar(data) {
  const el = document.getElementById('churn-country-bars');
  if (!el) return;
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(e => e[1]));
  el.innerHTML = entries.map(([country, count]) => `
    <div class="country-bar-row">
      <div class="country-name">${country}</div>
      <div class="country-bar-bg"><div class="country-bar-fill" style="width:${(count/max*100).toFixed(1)}%;background:linear-gradient(90deg,#ef4444,#f97316)"></div></div>
      <div class="country-rev">${count}</div>
    </div>
  `).join('');
}

// ─── Customers Page ────────────────────────────────────────────────────────────

async function fetchCustomers() {
  const url = `/api/customers?page=${state.customerPage}&search=${state.customerSearch}&segment=${state.customerSegment}&churn=${state.customerChurn}&sort=${state.customerSort}&dir=${state.customerDir}`;
  const res = await fetch(url);
  const d = await res.json();

  document.getElementById('customer-count').textContent = `${d.total.toLocaleString()} customers`;

  const tbody = document.getElementById('customer-table-body');
  if (tbody) {
    tbody.innerHTML = d.customers.map(c => `
      <tr>
        <td class="td-mono" style="color:var(--accent)">${c.CustomerID}</td>
        <td>${c.Country || '—'}</td>
        <td>${segBadge(c.Segment)}</td>
        <td>${churnBadge(c.ChurnRisk)}</td>
        <td class="td-mono">${c.Recency}d</td>
        <td class="td-mono">${c.Frequency}</td>
        <td class="td-mono">${fmtCurrency(c.Monetary)}</td>
        <td class="td-mono">${fmtCurrency(c.CLV)}</td>
        <td>
          <span style="font-family:'Space Mono',monospace;font-size:11px;padding:2px 8px;border-radius:4px;background:var(--bg-glass)">${c.RFM_Score}/12</span>
        </td>
      </tr>
    `).join('');
  }

  // Pagination
  renderPagination(d.page, d.pages, d.total);
}

function renderPagination(current, total, totalItems) {
  const el = document.getElementById('pagination');
  if (!el) return;

  let html = `<span class="page-info">${((current-1)*20)+1}–${Math.min(current*20,totalItems)} of ${totalItems.toLocaleString()}</span>`;
  html += `<button class="page-btn" onclick="changePage(${current-1})" ${current<=1?'disabled':''}>‹</button>`;

  const start = Math.max(1, current - 2);
  const end = Math.min(total, start + 4);
  for (let p = start; p <= end; p++) {
    html += `<button class="page-btn ${p===current?'active':''}" onclick="changePage(${p})">${p}</button>`;
  }

  html += `<button class="page-btn" onclick="changePage(${current+1})" ${current>=total?'disabled':''}>›</button>`;
  el.innerHTML = html;
}

function changePage(p) {
  state.customerPage = p;
  fetchCustomers();
}

function customerSearch(val) {
  state.customerSearch = val;
  state.customerPage = 1;
  fetchCustomers();
}

function customerFilter(type, val) {
  if (type === 'segment') state.customerSegment = val;
  if (type === 'churn') state.customerChurn = val;
  state.customerPage = 1;
  fetchCustomers();
}

// ─── Forecast Page ─────────────────────────────────────────────────────────────

async function fetchForecast() {
  const res = await fetch('/api/forecast');
  const d = await res.json();

  // Compute stats
  const histRevenues = d.historical.map(r => r.Revenue);
  const foreRevenues = d.forecast.map(r => r.Revenue);
  const totalHistorical = histRevenues.reduce((a,b)=>a+b,0);
  const totalForecast = foreRevenues.reduce((a,b)=>a+b,0);
  const growth = ((totalForecast/6 - totalHistorical/histRevenues.length) / (totalHistorical/histRevenues.length) * 100).toFixed(1);

  document.getElementById('forecast-total-hist').textContent = fmtCurrency(totalHistorical);
  document.getElementById('forecast-avg-monthly').textContent = fmtCurrency(totalHistorical/histRevenues.length);
  document.getElementById('forecast-projected').textContent = fmtCurrency(totalForecast);
  document.getElementById('forecast-growth').textContent = (growth > 0 ? '+' : '') + growth + '%';

  renderForecastChart(d);
}

function renderForecastChart(d) {
  destroyChart('forecast');
  const ctx = document.getElementById('chart-forecast');
  if (!ctx) return;

  const histLabels = d.historical.map(r => r.Month);
  const histData = d.historical.map(r => r.Revenue);
  const foreLabels = d.forecast.map(r => r.Month);
  const foreData = d.forecast.map(r => r.Revenue);

  charts.forecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [...histLabels, ...foreLabels],
      datasets: [
        {
          label: 'Historical Revenue',
          data: [...histData, ...new Array(foreLabels.length).fill(null)],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          borderWidth: 2.5,
          pointRadius: 3,
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Forecasted Revenue',
          data: [...new Array(histLabels.length - 1).fill(null), histData[histData.length-1], ...foreData],
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.07)',
          borderWidth: 2.5,
          borderDash: [6, 4],
          pointRadius: 4,
          pointBackgroundColor: '#10b981',
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { font: { size: 11 }, padding: 16, usePointStyle: true } },
        tooltip: {
          backgroundColor: '#0d1321',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: { label: ctx => `  ${ctx.dataset.label}: £${ctx.raw?.toLocaleString('en-GB', {minimumFractionDigits:2}) || '—'}` },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { family: 'Space Mono', size: 11 }, callback: v => '£' + (v/1000).toFixed(0)+'K' } },
      },
    },
  });
}

// ─── Model Comparison Page ─────────────────────────────────────────────────────

function metricColor(v) {
  if (v >= 85) return 'var(--success)';
  if (v >= 75) return '#f59e0b';
  return '#ef4444';
}

function generalizationGap(m) {
  if (!m) return 0;
  if (m.generalization_gap != null) return Number(m.generalization_gap);
  if (m.accuracy_gap != null) return Number(m.accuracy_gap);
  const train = Number(m.train_accuracy || 0);
  const test = Number(m.accuracy || 0);
  return Math.max(0, train - test);
}

async function fetchModels() {
  const res = await fetch('/api/models');
  const d = await res.json();

  // ── Dataset info banner ──────────────────────────────────────────────────
  const infoBanner = document.getElementById('ml-dataset-info');
  if (infoBanner && d.split_info) {
    const si = d.split_info;
    infoBanner.innerHTML = `
      <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:12px;color:var(--text-muted);
                  background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.18);
                  border-radius:8px;padding:12px 16px;margin-bottom:18px">
        <span>📊 <b style="color:var(--text-primary)">${si.total_samples?.toLocaleString()}</b> total samples</span>
        <span>🏋️ Train: <b style="color:var(--text-primary)">${si.train_samples?.toLocaleString()}</b></span>
        <span>🧪 Test: <b style="color:var(--text-primary)">${si.test_samples?.toLocaleString()}</b></span>
        <span>📐 CV: <b style="color:var(--text-primary)">${si.cv_folds}-fold stratified</b></span>
        <span>⚖️ SMOTE: <b style="color:var(--text-primary)">${si.smote_applied ? 'Applied' : 'Not needed'}</b></span>
        <span>🚫 Leaky cols removed: <b style="color:#f59e0b">${si.leaky_features_removed?.length || 9}</b></span>
      </div>`;
  }

  // ── Model cards ──────────────────────────────────────────────────────────
  const grid = document.getElementById('model-cards-grid');
  if (grid && d.model_results) {
    const entries = Object.entries(d.model_results).sort((a,b) => b[1].f1 - a[1].f1);
    grid.innerHTML = entries.map(([name, m]) => {
      const isBest = name === d.best_model;
      const gap = generalizationGap(m);
      const isOverfit = gap > 8;
      return `
      <div class="model-card ${isBest ? 'best' : ''}" style="${isOverfit?'border-color:#f59e0b44;':''}" >
        ${isBest ? '<div class="model-best-badge">🏆 BEST</div>' : ''}
        ${isOverfit ? '<div style="position:absolute;top:8px;right:8px;font-size:10px;background:#f59e0b22;color:#f59e0b;padding:2px 6px;border-radius:4px;border:1px solid #f59e0b44">⚠️ Overfitting</div>' : ''}
        <div class="model-name" style="padding-right:${isOverfit?'60px':'0'}">${name}</div>
        ${['accuracy','precision','recall','f1','auc'].map(metric => `
          <div class="metric-row">
            <span class="metric-label">${metric === 'auc' ? 'ROC-AUC' : metric.toUpperCase()}</span>
            <span class="metric-value" style="color:${metricColor(m[metric])}">${m[metric]}%</span>
          </div>
          <div class="progress-bar" style="margin-bottom:5px">
            <div class="progress-fill" style="width:${m[metric]}%;background:${metricColor(m[metric])}"></div>
          </div>
        `).join('')}
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06)">
          <div class="metric-row" style="margin-bottom:2px">
            <span class="metric-label" style="font-size:10px">Train Acc</span>
            <span class="metric-value" style="font-size:11px">${m.train_accuracy != null ? m.train_accuracy+'%' : 'N/A'}</span>
          </div>
          <div class="metric-row" style="margin-bottom:2px">
            <span class="metric-label" style="font-size:10px">Test Acc</span>
            <span class="metric-value" style="font-size:11px;color:${metricColor(m.accuracy)}">${m.accuracy}%</span>
          </div>
          <div class="metric-row" style="margin-bottom:2px">
            <span class="metric-label" style="font-size:10px">Generalization Gap</span>
            <span class="metric-value" style="font-size:11px;color:${gap>8?'#f59e0b':'var(--success)'}">
              ${gap != null ? (gap >= 0 ? '+' : '') + gap.toFixed(1) + '%' : 'N/A'}
            </span>
          </div>
          <div class="metric-row">
            <span class="metric-label" style="font-size:10px">CV Acc</span>
            <span class="metric-value" style="font-size:11px">${m.cv_accuracy != null ? m.cv_accuracy+'%' : 'N/A'}</span>
          </div>
        </div>
      </div>`;
    }).join('');
  }

  // ── Charts ───────────────────────────────────────────────────────────────
  renderModelAccuracyChart(d.model_results);
  renderTrainVsTestChart(d.model_results);
  renderROCChart(d.roc_data);
  renderFeatureImportance(d.feature_importance);
  renderConfusionMatrix(d.confusion_matrix, d.split_info);

  // ── Explanation panel ────────────────────────────────────────────────────
  if (d.explanation) renderExplanation(d.explanation);
}

function renderModelAccuracyChart(results) {
  destroyChart('modelAcc');
  const ctx = document.getElementById('chart-model-accuracy');
  if (!ctx || !results) return;

  const names   = Object.keys(results).sort((a,b) => results[b].f1 - results[a].f1);
  const metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
  const colors  = ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#06b6d4'];

  const minVal = Math.min(...names.flatMap(n => metrics.map(m => results[n][m])));
  const yMin   = Math.max(0, Math.floor(minVal / 10) * 10 - 5);

  charts.modelAcc = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: metrics.map((m, i) => ({
        label: m === 'auc' ? 'ROC-AUC' : m.toUpperCase(),
        data: names.map(n => results[n][m]),
        backgroundColor: colors[i] + 'bb',
        borderColor: colors[i],
        borderWidth: 1.5,
        borderRadius: 4,
      })),
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { font: { size: 11 }, padding: 10, usePointStyle: true } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { min: yMin, max: 100, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } },
      },
    },
  });
}

function renderTrainVsTestChart(results) {
  destroyChart('trainVsTest');
  const ctx = document.getElementById('chart-train-vs-test');
  if (!ctx || !results) return;

  const names = Object.keys(results).sort((a,b) => results[b].f1 - results[a].f1);
  const trainAcc = names.map(n => results[n].train_accuracy || 0);
  const testAcc  = names.map(n => results[n].accuracy || 0);

  charts.trainVsTest = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        { label: 'Train Accuracy', data: trainAcc, backgroundColor: '#3b82f6bb', borderColor: '#3b82f6', borderWidth: 1.5, borderRadius: 4 },
        { label: 'Test Accuracy',  data: testAcc,  backgroundColor: '#10b981bb', borderColor: '#10b981', borderWidth: 1.5, borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { font: { size: 11 }, padding: 10, usePointStyle: true } },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const n = names[items[0].dataIndex];
              const gap = generalizationGap(results[n]);
              return [`Generalization gap: ${gap > 0 ? '+' : ''}${gap.toFixed(1)}%${gap > 8 ? ' ⚠️ Overfitting' : ' ✅ Healthy'}`];
            }
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: {
          min: Math.max(0, Math.min(...[...trainAcc,...testAcc]) - 8),
          max: 100,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { callback: v => v + '%' },
        },
      },
    },
  });
}

function renderROCChart(rocData) {
  destroyChart('roc');
  const ctx = document.getElementById('chart-roc');
  if (!ctx || !rocData) return;

  const colors = ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#06b6d4','#ec4899'];
  const names  = Object.keys(rocData);

  const datasets = names.map((name, i) => ({
    label: `${name} (AUC=${rocData[name].auc}%)`,
    data: rocData[name].fpr.map((x, j) => ({ x, y: rocData[name].tpr[j] })),
    borderColor: colors[i % colors.length],
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1,
  }));

  // Diagonal reference
  datasets.push({
    label: 'Random (AUC=50%)',
    data: [{x:0,y:0},{x:1,y:1}],
    borderColor: 'rgba(255,255,255,0.2)',
    borderDash: [5,5],
    borderWidth: 1,
    pointRadius: 0,
  });

  charts.roc = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      showLine: true,
      plugins: { legend: { labels: { font: { size: 10 }, padding: 8, usePointStyle: true } } },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: 'False Positive Rate', font: { size: 11 } }, ticks: { callback: v => (v*100).toFixed(0)+'%' } },
        y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate',  font: { size: 11 } }, ticks: { callback: v => (v*100).toFixed(0)+'%' } },
      },
    },
  });
}

function renderFeatureImportance(fi) {
  const el = document.getElementById('feature-importance-bars');
  if (!el || !fi || !fi.length) return;
  const max = fi[0][1];
  el.innerHTML = fi.map(([name, val]) => `
    <div class="fi-row">
      <div class="fi-label">${name}</div>
      <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${(val/max*100).toFixed(1)}%"></div></div>
      <div class="fi-pct">${(val*100).toFixed(1)}%</div>
    </div>
  `).join('');
}

function renderConfusionMatrix(cm, splitInfo) {
  const el = document.getElementById('confusion-matrix');
  if (!el || !cm) return;
  const [[tn, fp],[fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const testN = splitInfo?.test_samples || total;
  el.innerHTML = `
    <div class="cm-grid">
      <div class="cm-cell cm-tn">
        <div class="cm-val">${tn}</div>
        <div class="cm-label">True Neg.</div>
        <div style="font-size:10px;color:var(--text-muted)">${(tn/testN*100).toFixed(1)}%</div>
      </div>
      <div class="cm-cell cm-fp">
        <div class="cm-val">${fp}</div>
        <div class="cm-label">False Pos.</div>
        <div style="font-size:10px;color:var(--text-muted)">${(fp/testN*100).toFixed(1)}%</div>
      </div>
      <div class="cm-cell cm-fn">
        <div class="cm-val">${fn}</div>
        <div class="cm-label">False Neg.</div>
        <div style="font-size:10px;color:var(--text-muted)">${(fn/testN*100).toFixed(1)}%</div>
      </div>
      <div class="cm-cell cm-tp">
        <div class="cm-val">${tp}</div>
        <div class="cm-label">True Pos.</div>
        <div style="font-size:10px;color:var(--text-muted)">${(tp/testN*100).toFixed(1)}%</div>
      </div>
    </div>
    <p style="font-size:11px;color:var(--text-muted);margin-top:10px;text-align:center">
      Best model on ${testN} held-out test samples
    </p>`;
}

function renderExplanation(exp) {
  const el = document.getElementById('ml-explanation');
  if (!el || !exp.explanation) return;
  const e = exp.explanation;
  const overfit = exp.overfit_models?.length > 0;
  el.innerHTML = `
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:10px;padding:20px;margin-top:24px">
      <h3 style="font-size:14px;font-weight:600;color:var(--text-primary);margin:0 0 16px">
        🤖 Automatic Model Explanation
      </h3>
      <div style="display:grid;gap:14px">
        <div style="padding:12px;background:rgba(59,130,246,0.08);border-radius:8px;border-left:3px solid #3b82f6">
          <div style="font-size:11px;font-weight:600;color:#3b82f6;margin-bottom:6px">💡 Why ${exp.best_model} Performed Best</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">${e.why_best_performed}</div>
        </div>
        <div style="padding:12px;background:${overfit?'rgba(245,158,11,0.08)':'rgba(16,185,129,0.08)'};border-radius:8px;border-left:3px solid ${overfit?'#f59e0b':'#10b981'}">
          <div style="font-size:11px;font-weight:600;color:${overfit?'#f59e0b':'#10b981'};margin-bottom:6px">⚠️ Overfitting Analysis</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">${e.overfitting_analysis}</div>
        </div>
        <div style="padding:12px;background:rgba(139,92,246,0.08);border-radius:8px;border-left:3px solid #8b5cf6">
          <div style="font-size:11px;font-weight:600;color:#8b5cf6;margin-bottom:6px">📊 Accuracy Range Note</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">${e.accuracy_range_note}</div>
        </div>
        <div style="padding:12px;background:rgba(6,182,212,0.08);border-radius:8px;border-left:3px solid #06b6d4">
          <div style="font-size:11px;font-weight:600;color:#06b6d4;margin-bottom:6px">🏢 Business Interpretation</div>
          <div style="font-size:12px;color:var(--text-secondary);line-height:1.6">${e.business_interpretation}</div>
        </div>
      </div>
    </div>`;
}

// ─── Insights Page ─────────────────────────────────────────────────────────────

async function fetchInsights() {
  const res = await fetch('/api/insights');
  const d = await res.json();

  const el = document.getElementById('insights-list');
  if (!el) return;
  el.innerHTML = d.insights.map(ins => `
    <div class="insight-card">
      <div class="insight-icon">${ins.icon}</div>
      <div>
        <div class="insight-title">${ins.title}</div>
        <div class="insight-body">${ins.body}</div>
        <div class="insight-action" onclick="navigate('${ins.page}')">${ins.action} →</div>
      </div>
    </div>
  `).join('');

  document.getElementById('insights-timestamp').textContent = 'Generated: ' + d.generated_at;
}

// ─── Products Page ─────────────────────────────────────────────────────────────

async function fetchProducts() {
  const res = await fetch('/api/products');
  const d = await res.json();

  document.getElementById('products-total-rev').textContent = fmtCurrency(d.total_revenue);
  document.getElementById('products-top-name').textContent = d.top_product;

  renderProductsChart(d.products);
  renderProductsTable(d.products);
}

function renderProductsChart(products) {
  destroyChart('productsBar');
  const ctx = document.getElementById('chart-products');
  if (!ctx) return;

  const top = products.slice(0, 10);
  const gradient = top.map((_, i) => `hsl(${215 + i * 12}, 80%, ${60 - i*2}%)`);

  charts.productsBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(p => p.Product.length > 28 ? p.Product.substring(0,28)+'…' : p.Product),
      datasets: [{
        label: 'Revenue (£)',
        data: top.map(p => p.Revenue),
        backgroundColor: gradient.map(c => c.replace(')', ', 0.75)').replace('hsl(','hsla(')),
        borderColor: gradient,
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => '  £' + ctx.raw.toLocaleString() }
      }},
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => '£' + (v/1000).toFixed(0)+'K' } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } },
      },
    },
  });
}

function renderProductsTable(products) {
  const tbody = document.getElementById('products-table-body');
  if (!tbody) return;
  const total = products.reduce((a,p) => a + p.Revenue, 0);
  tbody.innerHTML = products.slice(0,15).map((p, i) => `
    <tr>
      <td style="font-family:'Space Mono',monospace;font-size:11px;color:var(--text-muted)">#${i+1}</td>
      <td>${p.Product}</td>
      <td class="td-mono">${fmtCurrency(p.Revenue)}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px;">
          <div class="progress-bar" style="width:80px">
            <div class="progress-fill" style="width:${(p.Revenue/products[0].Revenue*100).toFixed(1)}%;background:var(--accent)"></div>
          </div>
          <span style="font-size:11px;font-family:'Space Mono',monospace">${(p.Revenue/total*100).toFixed(1)}%</span>
        </div>
      </td>
    </tr>
  `).join('');
}

// ─── Predictor Page ────────────────────────────────────────────────────────────

async function runPredictor() {
  const recency = parseFloat(document.getElementById('pred-recency').value) || 30;
  const frequency = parseFloat(document.getElementById('pred-frequency').value) || 5;
  const monetary = parseFloat(document.getElementById('pred-monetary').value) || 500;

  const btn = document.getElementById('predict-btn');
  btn.textContent = 'Predicting…';
  btn.disabled = true;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recency, frequency, monetary }),
    });
    const d = await res.json();

    const riskColor = { High: 'var(--danger)', Medium: 'var(--warning)', Low: 'var(--success)' };

    document.getElementById('pred-result').innerHTML = `
      <div class="result-panel">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
          <h3 style="font-size:14px;font-weight:700">Prediction Results</h3>
          <span style="font-size:11px;color:var(--text-muted)">Model: ${d.model_used}</span>
        </div>
        <div class="result-grid">
          <div class="result-item">
            <div class="result-label">Churn Risk</div>
            <div class="result-value" style="color:${riskColor[d.churn_risk]}">${d.churn_risk}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Churn Score</div>
            <div class="result-value" style="color:${riskColor[d.churn_risk]}">${d.churn_score}%</div>
          </div>
          <div class="result-item">
            <div class="result-label">Predicted CLV</div>
            <div class="result-value" style="color:var(--accent)">£${d.clv.toLocaleString('en-GB',{minimumFractionDigits:2})}</div>
          </div>
          <div class="result-item">
            <div class="result-label">Segment</div>
            <div class="result-value" style="font-size:14px;color:${SEG_COLORS[d.segment]}">${d.segment}</div>
          </div>
        </div>
        <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border)">
          <div style="display:flex;gap:24px;margin-bottom:12px">
            <div style="text-align:center">
              <div class="result-label">R Score</div>
              <div style="font-size:20px;font-weight:700;font-family:'Space Mono',monospace">${d.r_score}/4</div>
            </div>
            <div style="text-align:center">
              <div class="result-label">F Score</div>
              <div style="font-size:20px;font-weight:700;font-family:'Space Mono',monospace">${d.f_score}/4</div>
            </div>
            <div style="text-align:center">
              <div class="result-label">M Score</div>
              <div style="font-size:20px;font-weight:700;font-family:'Space Mono',monospace">${d.m_score}/4</div>
            </div>
            <div style="text-align:center">
              <div class="result-label">RFM Total</div>
              <div style="font-size:20px;font-weight:700;font-family:'Space Mono',monospace">${d.rfm_score}/12</div>
            </div>
          </div>
          <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);border-radius:8px;padding:12px;font-size:13px">
            <strong>💡 Recommendation:</strong> ${d.recommendation}
          </div>
        </div>
      </div>
    `;
  } catch (e) {
    showToast('Prediction failed. Please try again.', 'error');
  }

  btn.textContent = 'Run AI Prediction';
  btn.disabled = false;
}

// ─── Settings Page ──────────────────────────────────────────────────────────────

function toggleTheme() {
  state.darkMode = !state.darkMode;
  document.body.classList.toggle('light-mode', !state.darkMode);
  document.getElementById('theme-icon').textContent = state.darkMode ? '☀️' : '🌙';
  // Update chart defaults
  Chart.defaults.color = state.darkMode ? '#94a3b8' : '#475569';
  Chart.defaults.borderColor = state.darkMode ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  // Redraw all current charts
  pageDataCache = {};
  loadPage(state.currentPage);
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ─── Export ──────────────────────────────────────────────────────────────────────

function exportCSV() {
  window.open('/api/export/csv', '_blank');
  showToast('CSV export started!', 'success');
}

// ─── Init ────────────────────────────────────────────────────────────────────────

function init() {
  // Hide loader
  setTimeout(() => {
    const loader = document.getElementById('page-loader');
    if (loader) { loader.style.opacity = '0'; setTimeout(() => loader.remove(), 400); }
  }, 800);

  // Nav links
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      navigate(el.dataset.page);
    });
  });

  // Search with debounce
  const searchInput = document.getElementById('customer-search-input');
  if (searchInput) {
    let debounce;
    searchInput.addEventListener('input', e => {
      clearTimeout(debounce);
      debounce = setTimeout(() => customerSearch(e.target.value), 350);
    });
  }

  // Load initial page
  loadPage('overview');
}

document.addEventListener('DOMContentLoaded', init);
