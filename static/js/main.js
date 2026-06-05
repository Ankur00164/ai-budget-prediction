// ═══════════════════════════════════════════════════════════
//  AI Budget Prediction — Main JS
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {

  // ── Auto-dismiss alerts ──────────────────────────────────
  document.querySelectorAll('.alert-dismissible').forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      bsAlert && bsAlert.close();
    }, 5000);
  });

  // ── Sidebar mobile toggle ────────────────────────────────
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
    document.addEventListener('click', e => {
      if (sidebar.classList.contains('open') &&
          !sidebar.contains(e.target) &&
          !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // ── Upload drag-and-drop ─────────────────────────────────
  const uploadZone = document.getElementById('uploadZone');
  const fileInput  = document.getElementById('datasetFile');

  if (uploadZone && fileInput) {
    uploadZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(ev => {
      uploadZone.addEventListener(ev, e => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(ev => {
      uploadZone.addEventListener(ev, e => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
      });
    });

    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) {
        fileInput.files = e.dataTransfer.files;
        updateFileName(file.name);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) updateFileName(fileInput.files[0].name);
    });

    function updateFileName(name) {
      const label = document.getElementById('uploadFilename');
      if (label) label.textContent = name;
      const hint = document.getElementById('uploadHint');
      if (hint) hint.style.display = 'none';
    }
  }

  // ── Upload form loading state ────────────────────────────
  const uploadForm = document.getElementById('uploadForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
      showLoading('Uploading and validating dataset...');
    });
  }

  // ── Analyze button loading ────────────────────────────────
  const analyzeBtn = document.getElementById('analyzeBtn');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', () => {
      showLoading('Running ML analysis — please wait...');
    });
  }

  // ── Loading overlay ───────────────────────────────────────
  function showLoading(msg) {
    const overlay = document.getElementById('loadingOverlay');
    const msgEl   = document.getElementById('loadingMsg');
    if (overlay) {
      if (msgEl) msgEl.textContent = msg || 'Processing...';
      overlay.classList.add('show');
    }
  }

  // ── Animate stat counters ─────────────────────────────────
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.getAttribute('data-count'));
    const prefix = el.getAttribute('data-prefix') || '';
    const suffix = el.getAttribute('data-suffix') || '';
    const decimals = el.getAttribute('data-decimals') ? parseInt(el.getAttribute('data-decimals')) : 0;
    animateCount(el, 0, target, 1200, prefix, suffix, decimals);
  });

  function animateCount(el, start, end, duration, prefix, suffix, dec) {
    const startTime = performance.now();
    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = start + (end - start) * eased;
      el.textContent = prefix + formatNum(val, dec) + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  function formatNum(n, dec) {
    if (Math.abs(n) >= 1e7) return (n / 1e7).toFixed(1) + 'Cr';
    if (Math.abs(n) >= 1e5) return (n / 1e5).toFixed(1) + 'L';
    return n.toFixed(dec).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  // ── Render Plotly charts ──────────────────────────────────
  function renderChart(divId) {
    const el = document.getElementById(divId);
    if (!el) return;
    let raw = el.getAttribute('data-chart');
    if (!raw) return;
    try {
      // The attribute is JSON-encoded by Jinja tojson. Parse once to get the
      // plotly JSON string, then parse again to get the figure object.
      let fig = JSON.parse(raw);
      // If fig is still a string (double-encoded), parse again
      if (typeof fig === 'string') fig = JSON.parse(fig);
      Plotly.newPlot(divId, fig.data, {
        ...fig.layout,
        autosize: true,
        margin: { l: 50, r: 30, t: 50, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', size: 12 }
      }, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'pan2d', 'lasso2d']
      });
    } catch (e) {
      console.error('Chart render error for', divId, ':', e);
    }
  }

  ['chartBudgetVsActual', 'chartDeptDeviation', 'chartRegionDeviation',
   'chartMonthlyTrend', 'chartRiskDist', 'chartTop10'].forEach(id => renderChart(id));

  // ── Tooltip init ─────────────────────────────────────────
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
  });

  // ── Number formatting helper for template values ──────────
  window.fmtCurrency = (n) => {
    const num = parseFloat(n);
    if (isNaN(num)) return n;
    return '₹' + num.toLocaleString('en-IN', { maximumFractionDigits: 2 });
  };
});
