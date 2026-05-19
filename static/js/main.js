/* ── CourtX Main JS ─────────────────────────────────────────────────────── */

let CONFIG = {};
let currentStep = 1;

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch('/api/config');
    CONFIG    = await res.json();
    populateStates();
    populateCaseTypes();
    setupRangeSliders();
    setupPillGroups();
    setupEvidenceCards();
    document.getElementById('case_type').addEventListener('change', onCaseTypeChange);
  } catch (e) {
    console.error('Failed to load config:', e);
  }
}

function populateStates() {
  const sel = document.getElementById('state');
  CONFIG.states.forEach(s => {
    const o = document.createElement('option');
    o.value = o.textContent = s;
    sel.appendChild(o);
  });
}

function populateCaseTypes() {
  const sel = document.getElementById('case_type');
  CONFIG.case_types.forEach(ct => {
    const o = document.createElement('option');
    o.value = o.textContent = ct;
    sel.appendChild(o);
  });
}

function setupRangeSliders() {
  const sliders = [
    { id: 'case_duration_years', display: 'duration_val' },
    { id: 'bench_size',          display: 'bench_val'    },
    { id: 'num_witnesses',       display: 'witness_val'  },
    { id: 'num_cited_cases',     display: 'cited_val'    },
  ];
  sliders.forEach(({ id, display }) => {
    const el = document.getElementById(id);
    const dv = document.getElementById(display);
    if (el && dv) {
      el.addEventListener('input', () => { dv.textContent = el.value; });
    }
  });
}

function setupPillGroups() {
  document.querySelectorAll('.option-pills').forEach(group => {
    group.querySelectorAll('.pill').forEach(pill => {
      pill.addEventListener('click', () => {
        group.querySelectorAll('.pill').forEach(p => p.classList.remove('selected'));
        pill.classList.add('selected');
      });
    });
  });
}

function setupEvidenceCards() {
  ['your-evidence', 'opp-evidence'].forEach(id => {
    const group = document.getElementById(id);
    group.querySelectorAll('.ev-card').forEach(card => {
      card.addEventListener('click', () => {
        group.querySelectorAll('.ev-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        updateEvidenceScore();
      });
    });
  });
}

function onCaseTypeChange() {
  const caseType  = document.getElementById('case_type').value;
  const roles     = CONFIG.roles[caseType] || [];
  const container = document.getElementById('role-options');
  container.innerHTML = '';
  roles.forEach(role => {
    const parts = role.split('(');
    const title = parts[0].trim();
    const desc  = parts.length > 1 ? parts[1].replace(')', '').trim() : '';
    const card  = document.createElement('div');
    card.className = 'role-card';
    card.dataset.val = role;
    card.innerHTML = `<strong>${title}</strong><small>${desc}</small>`;
    card.addEventListener('click', () => {
      container.querySelectorAll('.role-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      // Update doc checklist when role changes
      if (currentStep === 3) buildDocChecklist();
    });
    container.appendChild(card);
  });
}

// ── Step Navigation ────────────────────────────────────────────────────────
function goStep(step) {
  if (!validateStep(currentStep)) return;

  document.getElementById(`step-${currentStep}`).classList.remove('active');
  document.querySelectorAll('.ps').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i + 1 < step) el.classList.add('done');
    if (i + 1 === step) el.classList.add('active');
  });

  currentStep = step;
  document.getElementById(`step-${step}`).classList.add('active');

  if (step === 3) buildDocChecklist();
  document.getElementById('predictor').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function validateStep(step) {
  if (step === 1) {
    if (!document.getElementById('state').value)
      return alert('Please select a state.'), false;
    if (!document.getElementById('case_type').value)
      return alert('Please select a case type.'), false;
    if (!document.getElementById('court_level').value)
      return alert('Please select a court level.'), false;
    if (!document.getElementById('case_complexity').value)
      return alert('Please select case complexity.'), false;
  }
  if (step === 2) {
    const role = document.querySelector('#role-options .role-card.selected');
    if (!role) return alert('Please select your role in the case.'), false;
    const oppLawyer = document.querySelector('#opp_lawyer .pill.selected');
    if (!oppLawyer) return alert('Please select the opposite lawyer experience.'), false;
  }
  return true;
}

// ── Document Checklist ─────────────────────────────────────────────────────
function buildDocChecklist() {
  const caseType = document.getElementById('case_type').value;
  const roleEl   = document.querySelector('#role-options .role-card.selected');
  if (!roleEl) return;

  const role     = roleEl.dataset.val;
  const roles    = CONFIG.roles[caseType] || [];
  const oppRole  = roles.find(r => r !== role) || '';
  const checklist = CONFIG.doc_checklist;

  const yourDocs = checklist?.[caseType]?.[role]?.your  || [];
  const oppDocs  = checklist?.[caseType]?.[role]?.opp   || [];

  buildDocList('your-docs-list', yourDocs, 'your');
  buildDocList('opp-docs-list',  oppDocs,  'opp');

  updateDocCount('your', yourDocs.length, 0);
  updateDocCount('opp',  oppDocs.length,  0);
  updateDocProgress('your', 0);
  updateDocProgress('opp',  0);
}

function buildDocList(containerId, docs, prefix) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  docs.forEach((doc, i) => {
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.id = `${prefix}-item-${i}`;
    item.innerHTML = `
      <input type="checkbox" id="${prefix}-cb-${i}" onchange="onDocChange('${prefix}', ${docs.length})" />
      <label for="${prefix}-cb-${i}">${doc}</label>
    `;
    item.addEventListener('click', (e) => {
      if (e.target.tagName !== 'INPUT') {
        const cb = item.querySelector('input');
        cb.checked = !cb.checked;
        cb.dispatchEvent(new Event('change'));
      }
    });
    container.appendChild(item);
  });
}

function onDocChange(prefix, total) {
  const checked = document.querySelectorAll(`#${prefix}-docs-list input:checked`).length;
  document.querySelectorAll(`#${prefix}-docs-list .doc-item`).forEach((item, i) => {
    const cb = item.querySelector('input');
    item.classList.toggle('checked', cb.checked);
  });
  updateDocCount(prefix, total, checked);
  const pct = total > 0 ? Math.round((checked / total) * 100) : 0;
  updateDocProgress(prefix, pct);
}

function updateDocCount(prefix, total, checked) {
  const el = document.getElementById(`${prefix}-doc-count`);
  if (el) el.textContent = `${checked} / ${total}`;
}

function updateDocProgress(prefix, pct) {
  const bar  = document.getElementById(`${prefix}-progress-bar`);
  const text = document.getElementById(`${prefix}-pct`);
  if (bar)  bar.style.width  = pct + '%';
  if (text) text.textContent = pct + '% document strength';
}

// ── Evidence Score ─────────────────────────────────────────────────────────
function updateEvidenceScore() {
  const extraMap  = { None: 0, Weak: 0.3, Medium: 0.6, Strong: 1.0 };
  const yourEvEl  = document.querySelector('#your-evidence .ev-card.selected');
  const oppEvEl   = document.querySelector('#opp-evidence .ev-card.selected');
  if (!yourEvEl || !oppEvEl) return;

  const yourExtra = extraMap[yourEvEl.dataset.val] || 0;
  const oppExtra  = extraMap[oppEvEl.dataset.val]  || 0;

  const yourDocPct = getDocPct('your');
  const oppDocPct  = getDocPct('opp');
  const finalYour  = (yourDocPct * 2) + yourExtra;
  const finalOpp   = (oppDocPct  * 2) + oppExtra;
  const net        = finalYour - finalOpp;

  let strength, color;
  if      (net >= 1.0) { strength = 'Strong';  color = '#22c55e'; }
  else if (net >= 0)   { strength = 'Medium';  color = '#f59e0b'; }
  else                 { strength = 'Weak';    color = '#ef4444'; }

  const box = document.getElementById('ev-score-detail');
  box.innerHTML = `Evidence Strength: <strong style="color:${color}">${strength}</strong> 
    — Your score: <strong>${finalYour.toFixed(2)}</strong> vs Opposite party: <strong>${finalOpp.toFixed(2)}</strong>`;
}

function getDocPct(prefix) {
  const all     = document.querySelectorAll(`#${prefix}-docs-list input`);
  const checked = document.querySelectorAll(`#${prefix}-docs-list input:checked`);
  return all.length > 0 ? checked.length / all.length : 0;
}

function computeEvidence() {
  const extraMap = { None: 0, Weak: 0.3, Medium: 0.6, Strong: 1.0 };
  const yourEvEl = document.querySelector('#your-evidence .ev-card.selected');
  const oppEvEl  = document.querySelector('#opp-evidence .ev-card.selected');
  const yourExtra = extraMap[yourEvEl?.dataset.val] || 0;
  const oppExtra  = extraMap[oppEvEl?.dataset.val]  || 0;

  const yourDocPct = getDocPct('your');
  const oppDocPct  = getDocPct('opp');
  const finalYour  = (yourDocPct * 2) + yourExtra;
  const finalOpp   = (oppDocPct  * 2) + oppExtra;
  const net        = finalYour - finalOpp;

  if      (net >= 1.0) return 'Strong';
  else if (net >= 0)   return 'Medium';
  else                 return 'Weak';
}

// ── Predict ────────────────────────────────────────────────────────────────
async function submitPrediction() {
  if (!validateStep(2)) return;

  const yourEvEl = document.querySelector('#your-evidence .ev-card.selected');
  const oppEvEl  = document.querySelector('#opp-evidence .ev-card.selected');
  if (!yourEvEl) return alert('Please select your additional evidence level.');
  if (!oppEvEl)  return alert("Please select the opposite party's additional evidence level.");

  const yourDocPct = getDocPct('your');
  const has_documents = yourDocPct > 0 ? 1 : 0;

  const payload = {
    state:                       document.getElementById('state').value,
    case_type:                   document.getElementById('case_type').value,
    court_level:                 document.getElementById('court_level').value,
    case_complexity:             document.getElementById('case_complexity').value,
    case_duration_years:         document.getElementById('case_duration_years').value,
    bench_size:                  document.getElementById('bench_size').value,
    num_witnesses:               document.getElementById('num_witnesses').value,
    num_cited_cases:             document.getElementById('num_cited_cases').value,
    opposite_lawyer_experience:  getPillValue('opp_lawyer'),
    legal_aid:                   getPillValue('legal_aid')          || 'No',
    settlement_attempted:        getPillValue('settlement_attempted') || 'No',
    has_documents:               has_documents,
    evidence_strength:           computeEvidence(),
  };

  goStep(5);
  document.getElementById('result-loading').style.display = 'block';
  document.getElementById('result-content').style.display = 'none';

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { alert('Prediction error: ' + data.error); return; }
    renderResult(data);
  } catch (e) {
    alert('Server error. Make sure Flask is running and models are loaded.');
    console.error(e);
  } finally {
    document.getElementById('result-loading').style.display = 'none';
  }
}

function getPillValue(groupId) {
  const el = document.querySelector(`#${groupId} .pill.selected`);
  return el ? el.dataset.val : null;
}

// ── Render Result ──────────────────────────────────────────────────────────
function renderResult(data) {
  document.getElementById('result-content').style.display = 'block';

  const pct = data.win_probability;

  // Gauge
  animateGauge(pct);
  document.getElementById('gauge-pct').textContent = pct.toFixed(1) + '%';
  const verdictEl = document.getElementById('gauge-verdict');
  verdictEl.textContent = data.verdict;
  verdictEl.style.color = pct >= 70 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#ef4444';

  // Score cards
  document.getElementById('res-prob').textContent    = pct.toFixed(1) + '%';
  document.getElementById('res-xgb').textContent     = data.xgb_prob + '%';
  document.getElementById('res-rf').textContent      = data.rf_prob + '%';
  const vb = document.getElementById('res-verdict');
  vb.textContent   = data.verdict;
  vb.style.color   = pct >= 70 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#ef4444';

  // Factors
  const posList = document.getElementById('pos-factors');
  const negList = document.getElementById('neg-factors');
  posList.innerHTML = '';
  negList.innerHTML = '';
  data.positive_factors.forEach(f => {
    posList.innerHTML += `<li>${f.feature} <span>+${f.value.toFixed(3)}</span></li>`;
  });
  data.negative_factors.forEach(f => {
    negList.innerHTML += `<li>${f.feature} <span>${f.value.toFixed(3)}</span></li>`;
  });

  // SHAP chart
  renderSHAP(data.shap_values);
}

function animateGauge(pct) {
  const totalLen = 408; // arc length for semicircle r=130
  const fillLen  = (pct / 100) * totalLen;
  const fill     = document.getElementById('gauge-fill');
  fill.style.strokeDasharray = `0 ${totalLen}`;
  setTimeout(() => {
    fill.style.transition = 'stroke-dasharray 1.2s ease';
    fill.style.strokeDasharray = `${fillLen} ${totalLen - fillLen}`;
  }, 100);

  // Needle
  const angle   = -180 + (pct / 100) * 180;
  const needle  = document.getElementById('gauge-needle');
  const rad     = (angle * Math.PI) / 180;
  const cx = 150, cy = 160, len = 110;
  const x2 = cx + len * Math.cos(rad);
  const y2 = cy + len * Math.sin(rad);
  setTimeout(() => {
    needle.setAttribute('x2', x2.toFixed(1));
    needle.setAttribute('y2', y2.toFixed(1));
  }, 100);
}

function renderSHAP(shapValues) {
  const container = document.getElementById('shap-chart');
  container.innerHTML = '';

  const maxAbs = Math.max(...shapValues.map(s => Math.abs(s.value)), 0.001);

  shapValues.forEach(item => {
    const pct  = Math.abs(item.value) / maxAbs * 100;
    const pos  = item.value >= 0;
    const color = pos ? '#22c55e' : '#ef4444';

    const row = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <div class="shap-label">${item.feature}</div>
      <div class="shap-bar-wrap">
        <div class="shap-bar" style="width:${pct}%;background:${color}">
          ${pct > 20 ? (pos ? '+' : '') + item.value.toFixed(3) : ''}
        </div>
      </div>
      <div class="shap-val" style="color:${color}">${(pos ? '+' : '') + item.value.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetPredictor() {
  currentStep = 1;
  document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
  document.getElementById('step-1').classList.add('active');
  document.querySelectorAll('.ps').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i === 0) el.classList.add('active');
  });
  // Reset form fields
  document.getElementById('state').value          = '';
  document.getElementById('case_type').value       = '';
  document.getElementById('court_level').value     = '';
  document.getElementById('case_complexity').value = '';
  document.getElementById('case_duration_years').value = 3;
  document.getElementById('duration_val').textContent  = '3';
  document.getElementById('bench_size').value      = 1;
  document.getElementById('bench_val').textContent = '1';
  document.getElementById('num_witnesses').value   = 2;
  document.getElementById('witness_val').textContent = '2';
  document.getElementById('num_cited_cases').value = 3;
  document.getElementById('cited_val').textContent  = '3';
  document.querySelectorAll('.pill').forEach(p    => p.classList.remove('selected'));
  document.querySelectorAll('.ev-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('role-options').innerHTML = '';
  document.getElementById('your-docs-list').innerHTML = '';
  document.getElementById('opp-docs-list').innerHTML  = '';
  document.getElementById('result-content').style.display = 'none';
  document.getElementById('result-loading').style.display = 'none';
  document.getElementById('predictor').scrollIntoView({ behavior: 'smooth' });
}

// ── Start ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);