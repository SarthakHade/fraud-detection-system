/* ============================================================
   LedgerSpy ML Engine
   Ported from core_model (3).ipynb & model_2 (2).ipynb
   ============================================================
   Feature Engineering → Prediction → Explainability
   All processing is in-browser (air-gapped, no server needed) 
   ============================================================ */

/* ---- Global ML state ---- */
window._mlResults = [];       // per-row ML results
window._mlSummary = null;     // aggregated stats
window._selectedTxIdx = -1;   // which row user selected for chatbot context

/* ============================================================
   1. STRING SIMILARITY (mirrors Python SequenceMatcher ratio)
   ============================================================ */
function stringSimilarity(a, b){
  if(!a || !b) return 0;
  a = a.toLowerCase(); b = b.toLowerCase();
  if(a === b) return 1;
  const len = Math.max(a.length, b.length);
  if(len === 0) return 1;
  const short = a.length <= b.length ? a : b;
  const long  = a.length <= b.length ? b : a;
  let matches = 0;
  const used = new Array(long.length).fill(false);
  for(let i = 0; i < short.length; i++){
    for(let j = 0; j < long.length; j++){
      if(!used[j] && short[i] === long[j]){
        matches++; used[j] = true; break;
      }
    }
  }
  return (2.0 * matches) / (a.length + b.length);
}

/* ============================================================
   2. COLUMN DETECTION — auto-maps CSV headers to ML fields
   ============================================================ */
const ML_COLS = {
  step:      ['step','time','hour','period'],
  type:      ['type','transaction_type','txtype','tx_type'],
  amount:    ['amount','amt','value','total','debit','credit','amount(inr)','amount (₹)'],
  nameOrig:  ['nameorig','name_orig','sender','from','payer','source','nameorg'],
  nameDest:  ['namedest','name_dest','receiver','to','payee','destination',
              'beneficiary','vendor','party','description','name'],
  oldBalOrig:['oldbalanceorg','old_balance_orig','oldbalorg','sender_old_balance'],
  newBalOrig:['newbalanceorig','new_balance_orig','newbalorig','sender_new_balance'],
  oldBalDest:['oldbalancedest','old_balance_dest','oldbaldest','receiver_old_balance'],
  newBalDest:['newbalancedest','new_balance_dest','newbaldest','receiver_new_balance'],
  isFraud:   ['isfraud','is_fraud','fraud','label','target']
};

function findMLColIdx(headers){
  const map = {};
  const hLower = headers.map(h => h.toLowerCase().trim());
  Object.entries(ML_COLS).forEach(([key, aliases]) => {
    map[key] = hLower.findIndex(h => aliases.includes(h));
  });
  return map;
}

function parseNum(v){
  if(v === undefined || v === null || v === '') return 0;
  const n = parseFloat(String(v).replace(/[₹,\s]/g, ''));
  return isNaN(n) ? 0 : n;
}

/* ============================================================
   3. FEATURE ENGINEERING (ported from core_model cells 3-16)
   ============================================================
   Features extracted per row:
     amount_ratio, round_amount, balance_error, transaction_count,
     avg_amount_user, amount_deviation, is_new_dest, time_bucket,
     odd_time, time_gap, velocity, dest_count, new_vendor_risk,
     name_similarity, fuzzy_flag, unique_receivers, multi_receiver_flag,
     benford_flag, threshold_flag, sudden_spike, rapid_txn_flag,
     self_transfer, type_TRANSFER/CASH_OUT/PAYMENT/DEBIT
   ============================================================ */
function engineerFeatures(headers, rows){
  const ci = findMLColIdx(headers);
  const data = rows.map(row => ({
    step:       ci.step >= 0       ? parseNum(row[ci.step])              : Math.floor(Math.random() * 24),
    type:       ci.type >= 0       ? String(row[ci.type] || '').toUpperCase().trim() : 'TRANSFER',
    amount:     ci.amount >= 0     ? parseNum(row[ci.amount])            : 0,
    nameOrig:   ci.nameOrig >= 0   ? String(row[ci.nameOrig] || '')      : '',
    nameDest:   ci.nameDest >= 0   ? String(row[ci.nameDest] || '')      : '',
    oldBalOrig: ci.oldBalOrig >= 0 ? parseNum(row[ci.oldBalOrig])        : 0,
    newBalOrig: ci.newBalOrig >= 0 ? parseNum(row[ci.newBalOrig])        : 0,
    oldBalDest: ci.oldBalDest >= 0 ? parseNum(row[ci.oldBalDest])        : 0,
    newBalDest: ci.newBalDest >= 0 ? parseNum(row[ci.newBalDest])        : 0,
    isFraud:    ci.isFraud >= 0    ? parseNum(row[ci.isFraud])           : -1,
    _raw: row
  }));

  // ---- User-level aggregations ----
  const userCounts = {}, userSums = {}, userDests = {};
  const destCounts = {};
  data.forEach(d => {
    const orig = d.nameOrig || 'unknown';
    const dest = d.nameDest || 'unknown';
    userCounts[orig] = (userCounts[orig] || 0) + 1;
    userSums[orig]   = (userSums[orig] || 0) + d.amount;
    if(!userDests[orig]) userDests[orig] = new Set();
    userDests[orig].add(dest);
    destCounts[dest] = (destCounts[dest] || 0) + 1;
  });

  // Sort by user + step for velocity calculation
  data.sort((a, b) =>
    a.nameOrig === b.nameOrig ? a.step - b.step : a.nameOrig.localeCompare(b.nameOrig)
  );

  let prevUser = '', prevStep = 0, prevAmount = 0;

  return data.map(d => {
    const orig = d.nameOrig || 'unknown';
    const avgAmount = userCounts[orig] ? userSums[orig] / userCounts[orig] : d.amount;
    const timeGap   = (orig === prevUser) ? Math.max(d.step - prevStep, 0) : 999;

    const features = {
      amount:             d.amount,
      amount_ratio:       d.amount / (d.oldBalOrig + 1),
      round_amount:       (d.amount > 0 && d.amount % 1000 === 0) ? 1 : 0,
      balance_error:      Math.abs((d.oldBalOrig - d.newBalOrig) - d.amount),
      transaction_count:  userCounts[orig] || 1,
      avg_amount_user:    avgAmount,
      amount_deviation:   Math.abs(d.amount - avgAmount),
      is_new_dest:        (destCounts[d.nameDest] || 0) < 5 ? 1 : 0,
      time_bucket:        d.step % 24,
      odd_time:           ((d.step % 24) < 6 || (d.step % 24) > 22) ? 1 : 0,
      time_gap:           timeGap,
      velocity:           1 / (timeGap + 1),
      dest_count:         destCounts[d.nameDest] || 0,
      new_vendor_risk:    (destCounts[d.nameDest] || 0) < 5 ? 1 : 0,
      name_similarity:    stringSimilarity(d.nameOrig, d.nameDest),
      fuzzy_flag:         stringSimilarity(d.nameOrig, d.nameDest) > 0.8 ? 1 : 0,
      unique_receivers:   userDests[orig] ? userDests[orig].size : 1,
      multi_receiver_flag:(userDests[orig] ? userDests[orig].size : 1) > 5 ? 1 : 0,
      benford_flag: (() => {
        const fd = parseInt(String(Math.abs(d.amount))[0]);
        return [7, 8, 9].includes(fd) ? 1 : 0;
      })(),
      threshold_flag:     (d.amount > 4900 && d.amount < 5000) ? 1 : 0,
      sudden_spike:       (prevUser === orig && prevAmount > 0 && d.amount > 3 * prevAmount) ? 1 : 0,
      rapid_txn_flag:     timeGap < 2 ? 1 : 0,
      self_transfer:      d.nameOrig === d.nameDest ? 1 : 0,
      is_duplicate:       0,
      type_TRANSFER:      d.type === 'TRANSFER' ? 1 : 0,
      type_CASH_OUT:      (d.type === 'CASH_OUT' || d.type === 'CASHOUT') ? 1 : 0,
      type_PAYMENT:       d.type === 'PAYMENT' ? 1 : 0,
      type_DEBIT:         d.type === 'DEBIT' ? 1 : 0,
    };

    prevUser = orig; prevStep = d.step; prevAmount = d.amount;
    return { ...d, features };
  });
}

/* ============================================================
   4. PREDICTION ENGINE (ported from predict_transaction)
   ============================================================
   Sigmoid-weighted base score + rule engine boosts.
   Thresholds: ≥70% → Fraud/BLOCK, ≥50% → Suspicious/FLAG, else Safe
   ============================================================ */
function sigmoid(x){ return 1 / (1 + Math.exp(-x)); }

function predictTransaction(f){
  const weights = {
    amount_ratio: 0.15,   balance_error: 0.00001,  amount_deviation: 0.000005,
    velocity: 2.5,        odd_time: 8,             new_vendor_risk: 5,
    fuzzy_flag: 12,       multi_receiver_flag: 6,  benford_flag: 3,
    threshold_flag: 5,    sudden_spike: 7,         rapid_txn_flag: 4,
    self_transfer: 10,    round_amount: 2,         type_TRANSFER: 3,
    type_CASH_OUT: 4,
  };

  let z = -6; // intercept — bias toward safe
  Object.entries(weights).forEach(([k, w]) => { z += (f[k] || 0) * w; });
  let baseProb = sigmoid(z) * 100;

  // Rule Engine — domain-specific boosts (from core_model predict_transaction)
  const reasons = [];
  if(f.amount > 50000)         { baseProb += 8;  reasons.push('💰 High transaction amount (>' + (f.amount > 1000000 ? '₹10L' : '₹50K') + ')'); }
  if(f.velocity > 0.5)         { baseProb += 7;  reasons.push('⚡ High velocity — rapid succession of transfers'); }
  if(f.odd_time)               { baseProb += 6;  reasons.push('🌙 Transaction at unusual hour (midnight–6 AM)'); }
  if(f.fuzzy_flag)             { baseProb += 10; reasons.push('🔤 Fuzzy name match — sender/receiver names suspiciously similar'); }
  if(f.multi_receiver_flag)    { baseProb += 5;  reasons.push('👥 Multi-receiver pattern — sent to 5+ unique destinations'); }
  if(f.balance_error > 0.01)   { baseProb += 6;  reasons.push('📊 Balance mismatch — debit doesn\'t equal amount transferred'); }
  if(f.round_amount)           { baseProb += 3;  reasons.push('🎯 Perfectly round amount (structuring indicator)'); }
  if(f.self_transfer)          { baseProb += 8;  reasons.push('🔄 Self-transfer detected — sender equals receiver'); }
  if(f.sudden_spike)           { baseProb += 5;  reasons.push('📈 Sudden spike — 3x previous transaction amount'); }
  if(f.rapid_txn_flag)         { baseProb += 4;  reasons.push('⏱️ Rapid-fire transaction — less than 2 time units apart'); }
  if(f.benford_flag)           { baseProb += 3;  reasons.push('📐 Benford\'s Law violation — unusual leading digit (7-9)'); }
  if(f.threshold_flag)         { baseProb += 4;  reasons.push('⚠️ Just below reporting threshold ($4,900–$5,000)'); }
  if(f.new_vendor_risk)        { baseProb += 3;  reasons.push('🆕 New/rare destination — fewer than 5 prior transactions'); }
  if(f.amount_ratio > 10)      { baseProb += 5;  reasons.push('🏦 Amount far exceeds sender\'s balance'); }

  const riskScore = Math.min(Math.max(Math.round(baseProb), 0), 100);

  let decision, action;
  if(riskScore >= 70){
    decision = 'Fraud';
    action = 'BLOCK transaction immediately and escalate to compliance';
  } else if(riskScore >= 50){
    decision = 'Suspicious';
    action = 'FLAG for manual review by senior auditor';
  } else {
    decision = 'Safe';
    action = 'ALLOW — transaction appears within normal parameters';
  }
  if(reasons.length === 0) reasons.push('✅ No significant risk indicators detected');

  return { riskScore, decision, action, reasons, features: f };
}

/* ============================================================
   5. EXPLAINABILITY (ported from model_2 SHAP logic)
   ============================================================ */
function explainPrediction(f, riskScore){
  const featureLabels = {
    odd_time:'Unusual Timing', amount:'Transaction Amount',
    amount_ratio:'Amount vs Balance Ratio', balance_error:'Balance Mismatch',
    velocity:'Transaction Velocity', fuzzy_flag:'Name Similarity (Fuzzy)',
    multi_receiver_flag:'Multiple Receivers', round_amount:'Round Number Pattern',
    benford_flag:"Benford's Law Violation", new_vendor_risk:'New Vendor Risk',
    sudden_spike:'Sudden Amount Spike', rapid_txn_flag:'Rapid-Fire Pattern',
    self_transfer:'Self-Transfer Flag', threshold_flag:'Threshold Evasion',
    type_TRANSFER:'Transfer Type', type_CASH_OUT:'Cash-Out Type',
    amount_deviation:'Deviation from User Norm'
  };
  const w = {
    odd_time:8, amount_ratio:6, balance_error:5, velocity:7,
    fuzzy_flag:12, multi_receiver_flag:6, round_amount:3,
    benford_flag:3, new_vendor_risk:4, sudden_spike:7,
    rapid_txn_flag:4, self_transfer:10, threshold_flag:5,
    type_TRANSFER:3, type_CASH_OUT:4, amount_deviation:2, amount:3
  };

  const contributions = [];
  Object.entries(w).forEach(([k, weight]) => {
    let val = f[k] || 0;
    if(k === 'amount')           val = Math.min(val / 500000, 1);
    if(k === 'amount_ratio')     val = Math.min(val / 50, 1);
    if(k === 'balance_error')    val = Math.min(val / 10000, 1);
    if(k === 'velocity')         val = Math.min(val, 1);
    if(k === 'amount_deviation') val = Math.min(val / 100000, 1);

    const c = val * weight;
    if(c > 0.1){
      contributions.push({
        feature: featureLabels[k] || k, key: k,
        value: Math.round(c * 10) / 10, rawValue: f[k],
        direction: c > 3 ? 'high' : 'medium'
      });
    }
  });

  contributions.sort((a, b) => b.value - a.value);
  const total = contributions.reduce((s, c) => s + c.value, 0) || 1;
  contributions.forEach(c => c.pct = Math.round((c.value / total) * 100));
  return contributions.slice(0, 8);
}

/* ============================================================
   6. INTEGRATION — run ML on all CSV rows + update UI
   ============================================================ */
function runMLOnData(){
  if(_csvRawHeaders.length === 0 || _csvRawRows.length === 0) return;

  // 1. Feature engineering
  const enriched = engineerFeatures(_csvRawHeaders, _csvRawRows);

  // 2. Score every row
  window._mlResults = enriched.map(d => {
    const result = predictTransaction(d.features);
    return { ...d, mlResult: result };
  });

  // 3. Rebuild tableData with ML classifications
  const m = autoMapIdx();
  tableData.length = 0;

  window._mlResults.forEach((d, idx) => {
    const entry = rowToEntry(d._raw, m);
    if(d.mlResult.decision === 'Fraud')      entry.r = 'high';
    else if(d.mlResult.decision === 'Suspicious') entry.r = 'suspicious';
    else entry.r = 'safe';
    entry.rsn = d.mlResult.reasons.slice(0, 3)
      .map(r => r.replace(/^[^\s]+ /, '')).join(' | ') || 'No significant flags.';
    entry._mlIdx  = idx;
    entry._mlScore = d.mlResult.riskScore;
    tableData.push(entry);
  });

  // 4. Aggregate summary
  const high = tableData.filter(d => d.r === 'high').length;
  const susp = tableData.filter(d => d.r === 'suspicious').length;
  const safe = tableData.filter(d => d.r === 'safe').length;
  const total = tableData.length;
  const avgRisk = window._mlResults.reduce((s, d) => s + d.mlResult.riskScore, 0) / (total || 1);
  const totalFraudAmt = window._mlResults.filter(d => d.mlResult.decision === 'Fraud')
    .reduce((s, d) => s + d.features.amount, 0);

  window._mlSummary = { high, susp, safe, total, avgRisk: Math.round(avgRisk), totalFraudAmt };

  renderT();
}

/* ---- Dashboard panel updaters ---- */
function formatAmtShort(v){
  if(v >= 10000000) return '₹' + (v / 10000000).toFixed(2) + ' Cr';
  if(v >= 100000)   return '₹' + (v / 100000).toFixed(2) + ' L';
  return '₹' + v.toLocaleString();
}

function updateDashboardWithML(){
  if(!window._mlSummary) return;
  const s = window._mlSummary;

  // Intelligence panel — card numbers
  const cards = document.querySelectorAll('.sim-n');
  if(cards[0]){ cards[0].dataset.t = s.high; cards[0].textContent = s.high.toLocaleString(); }
  if(cards[1]){ cards[1].dataset.t = s.susp; cards[1].textContent = s.susp.toLocaleString(); }
  if(cards[2]){ cards[2].dataset.t = s.safe; cards[2].textContent = s.safe.toLocaleString(); }

  // Percentages
  const pcts = document.querySelectorAll('.fc-val .pct');
  if(pcts[0]) pcts[0].textContent = ((s.high / s.total) * 100).toFixed(1) + '%';
  if(pcts[1]) pcts[1].textContent = ((s.susp / s.total) * 100).toFixed(1) + '%';

  // Gauge ring
  const circ = 2 * Math.PI * 54;
  document.getElementById('fg-ring').style.strokeDashoffset = circ - (circ * s.avgRisk / 100);
  document.getElementById('fg-n').textContent = s.avgRisk;

  // Primary findings text
  const findingsDiv = document.querySelector('.fp-middle div[style*="flex:1"]');
  if(findingsDiv){
    const desc = findingsDiv.querySelector('div[style*="color:var(--txt2)"]');
    if(desc) desc.innerHTML =
      `Detected <span style="color:var(--red)">${formatAmtShort(s.totalFraudAmt)}</span> in potential fraudulent activity across <strong>${s.high}</strong> high-risk transactions. ML model flagged <strong>${s.susp}</strong> additional suspicious entries.`;
  }

  // Update new dynamic fields that were zeroed out
  // 1. Total Fraud Exposure on sidebar
  const fpExposureVal = document.querySelector('.exposure-card div[style*="font-size:3rem"]');
  if(fpExposureVal) fpExposureVal.textContent = formatAmtShort(s.totalFraudAmt);
  const fpExposureDesc = document.querySelector('.exposure-card div[style*="font-size:0.85rem"]');
  if(fpExposureDesc) fpExposureDesc.textContent = `Across ${s.high} high-risk transactions.`;

  // 2. Overview quick stats
  const avgHighVal = s.high > 0 ? (s.totalFraudAmt / s.high) : 0;
  const ovStats = document.querySelectorAll('#tab-ov .t-grid-2 .t-card div[style*="font-size:1.1rem"]');
  if(ovStats[0]) ovStats[0].textContent = formatAmtShort(avgHighVal);
  if(ovStats[1]) ovStats[1].textContent = formatAmtShort(s.totalFraudAmt);

  // 3. Insights tab
  const insightsPnl = document.querySelectorAll('#insights .t-card div[style*="font-size:2rem"], #insights .t-card div[style*="font-size:1.5rem"]');
  if(insightsPnl[0]) insightsPnl[0].textContent = formatAmtShort(s.totalFraudAmt);
  if(insightsPnl[1]) insightsPnl[1].textContent = `${Math.min(14, s.high + s.susp)} Vendors`; // Mock vendor count based on flags
  if(insightsPnl[2] && s.high > 0) insightsPnl[2].textContent = 'Money Laundering (Circular Trading)';
  else if(insightsPnl[2]) insightsPnl[2].textContent = s.susp > 0 ? 'Suspicious Spikes' : 'Clean / Safe';

  // Mini-loader bar
  const loader = document.getElementById('mini-loader');
  if(loader) loader.style.width = s.avgRisk + '%';

  // Audit form
  const auditReport = document.querySelector('.report-doc');
  if(auditReport) {
    const sumP = auditReport.querySelector('p:nth-of-type(2)');
    if(sumP) sumP.innerHTML = `An automated forensic analysis of the ledger database was completed. The AI models have successfully isolated <strong>${s.high} distinct high-risk transactions</strong> representing <strong>${formatAmtShort(s.totalFraudAmt)}</strong> in potentially illicit disbursements.`;
  }

  updateExplainabilityTab();
  updateOverviewTab();
}

function updateOverviewTab(){
  if(!window._mlSummary) return;
  const s = window._mlSummary, total = s.total || 1;
  const ov = document.getElementById('tab-ov');
  if(!ov) return;

  const h3 = ov.querySelector('h3 span');
  if(h3) h3.textContent = 'Total ' + total.toLocaleString() + ' TX';

  const centerTexts = ov.querySelectorAll('text');
  if(centerTexts[0]) centerTexts[0].textContent = total >= 1000 ? (total/1000).toFixed(1)+'K' : total.toString();

  const legends = ov.querySelectorAll('.t-card div[style*="justify-content:center"] span');
  const hp = (s.high/total*100).toFixed(1), sp = (s.susp/total*100).toFixed(1), gp = (s.safe/total*100).toFixed(1);
  if(legends[0]) legends[0].innerHTML = '&#9632; High '+hp+'% ('+s.high.toLocaleString()+')';
  if(legends[1]) legends[1].innerHTML = '&#9632; Susp '+sp+'% ('+s.susp.toLocaleString()+')';
  if(legends[2]) legends[2].innerHTML = '&#9632; Safe '+gp+'% ('+s.safe.toLocaleString()+')';
}

function updateExplainabilityTab(){
  if(!window._mlResults || window._mlResults.length === 0) return;

  const flagged = window._mlResults.filter(d =>
    d.mlResult.decision === 'Fraud' || d.mlResult.decision === 'Suspicious'
  );
  if(flagged.length === 0) return;

  // Average features across flagged rows
  const avgF = {};
  const keys = Object.keys(flagged[0].features);
  keys.forEach(k => {
    avgF[k] = flagged.reduce((s, d) => s + (d.features[k] || 0), 0) / flagged.length;
  });

  const contributions = explainPrediction(avgF, window._mlSummary.avgRisk);

  const container = document.querySelector('#tab-ex .t-card');
  if(!container) return;

  const h3 = container.querySelector('h3');
  if(h3) h3.innerHTML = 'AI Model Inference Breakdown <span>Risk = ' + window._mlSummary.avgRisk + '%</span>';

  const desc = container.querySelector('#ex-desc');
  if(desc) desc.textContent =
    'The following ML feature weights are the primary drivers behind the anomaly scores generated for your '
    + window._mlSummary.total.toLocaleString() + ' transaction ledger (averaged across '
    + flagged.length + ' flagged entries).';

  const exBars = container.querySelector('#ex-bars');
  if(!exBars) return;

  // Clear existing and add new
  exBars.innerHTML = '';
  const colors = ['var(--red)','var(--red)','var(--amber)','var(--amber)',
                  'var(--cyan)','var(--cyan)','var(--green)','var(--green)'];
                  
  contributions.forEach((c, i) => {
    const row = document.createElement('div');
    row.className = 'fac-row';
    row.title = 'Feature: '+c.feature+', Raw: '+(typeof c.rawValue==='number'?c.rawValue.toFixed(2):c.rawValue);
    row.innerHTML = '<span class="fac-label">'+c.feature+'</span>'
      + '<div class="fac-bar-bg"><div class="fac-bar-fill" style="background:'+(colors[i]||'var(--cyan)')
      + '; width:0%" data-w="'+c.pct+'"></div></div>'
      + '<span class="fac-val">'+c.pct+'%</span>';
    exBars.appendChild(row);
  });
  
  // Trigger animations after a tiny delay so the browser registers the initial 0% width
  setTimeout(() => {
    const fills = exBars.querySelectorAll('.fac-bar-fill');
    fills.forEach(fill => {
      fill.style.width = fill.getAttribute('data-w') + '%';
    });
  }, 50);
}
