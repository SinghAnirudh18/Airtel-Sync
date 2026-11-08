// Project Udyam Digital - Frontend Controller

let currentAdvisorBundle = null;
let currentQuoteItems = [];
let voiceAdvisorEnabled = false;

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  // Initialize default Kirana scenario and load backoffice leads
  loadScenario('kirana');
  loadBackofficeLeads();
});

// -----------------------------------------------------------------------------
// Tab Switching
// -----------------------------------------------------------------------------
function switchTab(tabName) {
  const tabs = ['advisor', 'quote', 'kyc', 'tracker', 'backoffice'];
  tabs.forEach(t => {
    const view = document.getElementById(`view-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (view) {
      if (t === tabName) {
        view.classList.remove('hidden');
        view.classList.add('block');
      } else {
        view.classList.remove('block');
        view.classList.add('hidden');
      }
    }
    if (btn) {
      if (t === tabName) {
        btn.classList.add('bg-brand-red', 'text-white', 'shadow');
        btn.classList.remove('text-slate-400');
      } else {
        btn.classList.remove('bg-brand-red', 'text-white', 'shadow');
        btn.classList.add('text-slate-400');
      }
    }
  });

  if (window.lucide) {
    lucide.createIcons();
  }
}

// -----------------------------------------------------------------------------
// Preset Scenarios
// -----------------------------------------------------------------------------
const SCENARIOS = {
  kirana: {
    name: "Saffron Bakers & Confectionery",
    type: "Bakery & Confectionery",
    segment: "Micro retail & services",
    tier: 2,
    emp: 3,
    sites: 1,
    hasGstin: "Y",
    spend: 1350,
    need: "Fast UPI / POS payments and WhatsApp ordering"
  },
  clinic: {
    name: "Harmony Dental Care LLP",
    type: "Dental Clinic",
    segment: "Professional offices",
    tier: 1,
    emp: 8,
    sites: 1,
    hasGstin: "Y",
    spend: 3200,
    need: "High uptime broadband, Cloud Patient Records & Static IP"
  },
  multisite: {
    name: "Star Electronics Retail Chain",
    type: "Consumer Electronics Retailer",
    segment: "Small multi-site businesses",
    tier: 1,
    emp: 45,
    sites: 6,
    hasGstin: "Y",
    spend: 9800,
    need: "Consolidated billing, 4G/5G failover router & Cloud CCTV"
  },
  factory: {
    name: "Precision Auto Components Ltd",
    type: "Auto Components Manufacturing",
    segment: "Small manufacturers",
    tier: 2,
    emp: 30,
    sites: 2,
    hasGstin: "Y",
    spend: 18500,
    need: "Internet Leased Line, Managed Firewall & Machine IoT Sensors"
  }
};

function loadScenario(key) {
  const sc = SCENARIOS[key];
  if (!sc) return;

  document.getElementById('input-biz-name').value = sc.name;
  document.getElementById('input-biz-type').value = sc.type;
  document.getElementById('input-segment').value = sc.segment;
  document.getElementById('input-tier').value = sc.tier;
  document.getElementById('input-emp').value = sc.emp;
  document.getElementById('input-sites').value = sc.sites;
  document.getElementById('input-has-gst').value = sc.hasGstin;
  document.getElementById('input-spend').value = sc.spend;

  requestAdvisorBundle();
}

// -----------------------------------------------------------------------------
// Step 1: Request GenAI Bundle
// -----------------------------------------------------------------------------
async function requestAdvisorBundle() {
  const payload = {
    business_name: document.getElementById('input-biz-name').value,
    business_type: document.getElementById('input-biz-type').value,
    segment: document.getElementById('input-segment').value,
    city_tier: parseInt(document.getElementById('input-tier').value),
    employee_count: parseInt(document.getElementById('input-emp').value),
    num_sites: parseInt(document.getElementById('input-sites').value),
    has_gstin: document.getElementById('input-has-gst').value,
    current_spend_inr: parseFloat(document.getElementById('input-spend').value),
    preferred_language: document.getElementById('lang-select').value
  };

  try {
    const res = await fetch('/api/advisor/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    currentAdvisorBundle = data;

    // Render output
    document.getElementById('rec-biz-title').textContent = data.business_name;
    document.getElementById('rec-monthly-total').textContent = `₹${data.transparent_monthly_total.toLocaleString()}`;
    document.getElementById('rec-setup-fee').textContent = `₹${data.one_time_setup_total.toLocaleString()}`;
    document.getElementById('rec-sla-days').textContent = `${data.estimated_activation_days} Business Days`;
    document.getElementById('rec-contract-term').textContent = `${data.contract_months} Months`;
    document.getElementById('rec-narrative').textContent = data.explanation;

    // Table rows
    const tbody = document.getElementById('rec-items-tbody');
    tbody.innerHTML = '';
    data.items.forEach(item => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/40 transition';
      tr.innerHTML = `
        <td class="py-2.5">
          <div class="font-bold text-white">${item.name}</div>
          <div class="text-[11px] text-slate-400">${item.description}</div>
        </td>
        <td class="py-2.5 text-slate-400">${item.category}</td>
        <td class="py-2.5 text-center font-semibold text-white">${item.quantity} ${item.unit}</td>
        <td class="py-2.5 text-right font-bold text-emerald-400">₹${item.monthly_subtotal.toLocaleString()}</td>
        <td class="py-2.5 text-right font-medium text-sky-400">${item.install_days}d</td>
      `;
      tbody.appendChild(tr);
    });

    // Populate Quote Builder
    setupQuoteBuilder(data);

    if (voiceAdvisorEnabled) {
      speakNarrative();
    }
  } catch (err) {
    console.error("Advisor Error:", err);
  }
}

// -----------------------------------------------------------------------------
// Step 2: Quote Builder & Add-on Management
// -----------------------------------------------------------------------------
function setupQuoteBuilder(bundleData) {
  currentQuoteItems = bundleData.items.map(i => ({
    product_id: i.product_id,
    name: i.name,
    category: i.category,
    quantity: i.quantity,
    unit: i.unit,
    unit_price: i.unit_price,
    monthly_subtotal: i.monthly_subtotal,
    setup_fee: i.setup_fee,
    install_days: i.install_days
  }));

  renderQuoteGrid();
}

function renderQuoteGrid() {
  const container = document.getElementById('quote-products-grid');
  container.innerHTML = '';

  // Standard catalogue of potential add-ons
  const addOnsCatalog = [
    { product_id: 'P07', name: 'Business Postpaid SIM', category: 'Connectivity', unit: 'line', unit_price: 349, setup_fee: 0, install_days: 1 },
    { product_id: 'P23', name: 'Cloud CCTV HD Camera', category: 'IoT & Physical', unit: 'camera', unit_price: 249, setup_fee: 1800, install_days: 4 },
    { product_id: 'P09', name: 'Backup 4G/5G Failover Router', category: 'Connectivity', unit: 'site', unit_price: 549, setup_fee: 1500, install_days: 3 },
    { product_id: 'P11', name: 'Fixed Static Public IP', category: 'Connectivity', unit: 'IP', unit_price: 250, setup_fee: 0, install_days: 1 },
    { product_id: 'P20', name: 'Endpoint Security / Antivirus', category: 'Security', unit: 'device', unit_price: 124, setup_fee: 0, install_days: 1 },
    { product_id: 'P16', name: 'Cloud Email & Productivity', category: 'Cloud', unit: 'user', unit_price: 137, setup_fee: 0, install_days: 1 }
  ];

  // Merge currently selected with addon suggestions
  const allCards = [...currentQuoteItems];
  addOnsCatalog.forEach(addon => {
    if (!allCards.some(c => c.product_id === addon.product_id)) {
      allCards.push({ ...addon, quantity: 0, monthly_subtotal: 0 });
    }
  });

  allCards.forEach(item => {
    const isIncluded = item.quantity > 0;
    const card = document.createElement('div');
    card.className = `p-4 rounded-xl border transition ${
      isIncluded ? 'bg-slate-800/90 border-red-500/40 shadow-lg' : 'bg-slate-900/60 border-slate-800 opacity-60'
    }`;

    card.innerHTML = `
      <div class="flex items-start justify-between">
        <div>
          <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">${item.category}</span>
          <h4 class="font-bold text-white text-sm mt-0.5">${item.name}</h4>
          <span class="text-xs text-slate-400 block mt-1">₹${item.unit_price} / ${item.unit}</span>
        </div>
        <div class="text-right">
          <span class="text-xs font-bold ${isIncluded ? 'text-emerald-400' : 'text-slate-500'}">
            ₹${(item.unit_price * item.quantity).toLocaleString()}/mo
          </span>
        </div>
      </div>

      <div class="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
        <span class="text-xs text-slate-400">Qty:</span>
        <div class="flex items-center space-x-2">
          <button onclick="changeQuoteItemQty('${item.product_id}', -1)" class="w-6 h-6 rounded bg-slate-700 hover:bg-slate-600 text-white font-bold text-xs flex items-center justify-center">-</button>
          <span class="w-6 text-center text-xs font-bold text-white">${item.quantity}</span>
          <button onclick="changeQuoteItemQty('${item.product_id}', 1)" class="w-6 h-6 rounded bg-slate-700 hover:bg-slate-600 text-white font-bold text-xs flex items-center justify-center">+</button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });

  recalculateQuoteTotals();
}

async function changeQuoteItemQty(pid, delta) {
  let found = currentQuoteItems.find(i => i.product_id === pid);
  if (!found && delta > 0) {
    const fallbackAddon = {
      P07: { name: 'Business Postpaid SIM', category: 'Connectivity', unit: 'line', unit_price: 349, setup_fee: 0, install_days: 1 },
      P23: { name: 'Cloud CCTV HD Camera', category: 'IoT & Physical', unit: 'camera', unit_price: 249, setup_fee: 1800, install_days: 4 },
      P09: { name: 'Backup 4G/5G Failover Router', category: 'Connectivity', unit: 'site', unit_price: 549, setup_fee: 1500, install_days: 3 },
      P11: { name: 'Fixed Static Public IP', category: 'Connectivity', unit: 'IP', unit_price: 250, setup_fee: 0, install_days: 1 },
      P20: { name: 'Endpoint Security / Antivirus', category: 'Security', unit: 'device', unit_price: 124, setup_fee: 0, install_days: 1 },
      P16: { name: 'Cloud Email & Productivity', category: 'Cloud', unit: 'user', unit_price: 137, setup_fee: 0, install_days: 1 }
    }[pid];

    if (fallbackAddon) {
      found = { product_id: pid, quantity: 0, ...fallbackAddon };
      currentQuoteItems.push(found);
    }
  }

  if (found) {
    found.quantity = Math.max(0, found.quantity + delta);
    if (found.quantity === 0) {
      currentQuoteItems = currentQuoteItems.filter(i => i.product_id !== pid);
    }
  }

  renderQuoteGrid();
}

async function recalculateQuoteTotals() {
  const payload = {
    items: currentQuoteItems.map(i => ({ product_id: i.product_id, quantity: i.quantity })),
    city_tier: parseInt(document.getElementById('input-tier').value),
    employee_count: parseInt(document.getElementById('input-emp').value),
    num_sites: parseInt(document.getElementById('input-sites').value),
    has_gstin: document.getElementById('input-has-gst').value
  };

  try {
    const res = await fetch('/api/quote/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    document.getElementById('quote-total-monthly').textContent = `₹${data.total_monthly_inr.toLocaleString()}`;
    document.getElementById('quote-total-setup').textContent = `₹${data.total_setup_fee_inr.toLocaleString()}`;
    document.getElementById('quote-total-sla').textContent = `${data.estimated_install_days} Days`;

    const warnBox = document.getElementById('quote-warning-box');
    const warnText = document.getElementById('quote-warning-text');
    if (data.dependency_warnings && data.dependency_warnings.length > 0) {
      warnBox.classList.remove('hidden');
      warnText.textContent = data.dependency_warnings.join(' · ');
    } else {
      warnBox.classList.add('hidden');
    }
  } catch (err) {
    console.error("Quote recalculation error:", err);
  }
}

function acceptQuoteAndProceed() {
  switchTab('kyc');
  runAuditVerification();
}

// -----------------------------------------------------------------------------
// Step 3: Smart Instant KYC & DPDP Compliance
// -----------------------------------------------------------------------------
function onSpecimenSelectChange() {
  const appKey = document.getElementById('specimen-select').value;
  const specimens = {
    K01: { name: "Vindhya Tours & Travels", gstin: "37IHKXV5167L1ZS", udyam: "UDYAM-MH-00-2675962", date: "2024-03-24" },
    K02: { name: "Harmony Dental Care", gstin: "19EOGXH4115Y1ZL", udyam: "UDYAM-WB-00-5053330", date: "2026-05-22" },
    K03: { name: "Ganesh Woodcraft", gstin: "29FGUXG3354T1ZQ", udyam: "UDYAM-KR-00-1115424", date: "2026-06-15" },
    K04: { name: "Saffron Dental Care LLP", gstin: "27P25KS1936012H", udyam: "UDYAM-MH-00-6131836", date: "2026-05-18" },
    K05: { name: "Metro Retail Store", gstin: "", udyam: "", date: "2026-06-11" },
    K06: { name: "Saffron Unisex Salon", gstin: "08YKZXS5189L1ZO", udyam: "UDYAM-RJ-00-6577157", date: "2026-06-01" },
    K07: { name: "Neelgiri Physio Clinic", gstin: "29HIHXN8231E1ZI", udyam: "UDYAM-DL-00-95655547", date: "2026-07-05" },
    K08: { name: "Green Leaf Legal Associates", gstin: "10EGQXG6446F1Z0", udyam: "UDYAM-BR-00-7880010", date: "2026-05-26" },
    K09: { name: "Kaveri Insurance Advisors", gstin: "07HBIXK4506C1ZF", udyam: "UDYAM-TS-00-8542181", date: "2026-07-10" },
    K10: { name: "Kaveri Stays LLP", gstin: "24GMYXK0402E1ZQ", udyam: "UDYAM-GJ-00-6760776", date: "2026-06-18" },
    K11: { name: "Trinity Bake House Pvt Ltd", gstin: "33XGTXT4416F1Z3", udyam: "UDYAM-TN-00-2649689", date: "2025-10-31" },
    K12: { name: "Star Electronics", gstin: "33MFTXS9954Q1Z9", udyam: "UDYAM-TN-00-3866196", date: "2024-02-03" }
  };

  const chosen = specimens[appKey];
  if (chosen) {
    document.getElementById('kyc-input-name').value = chosen.name;
    document.getElementById('kyc-input-gstin').value = chosen.gstin;
    document.getElementById('kyc-input-udyam').value = chosen.udyam;
    document.getElementById('kyc-input-date').value = chosen.date;
  }

  runAuditVerification();
}

async function runAuditVerification() {
  const specimenKey = document.getElementById('specimen-select').value;
  try {
    const res = await fetch('/api/kyc/verify-specimen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ applicant_id: specimenKey })
    });
    const data = await res.json();
    renderKycResults(data);
  } catch (err) {
    console.error("KYC Verification Error:", err);
  }
}

async function onCustomFileUploaded(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('business_name', document.getElementById('kyc-input-name').value);

  const listContainer = document.getElementById('kyc-checks-list');
  listContainer.innerHTML = `
    <div class="p-4 rounded-xl bg-slate-800/80 border border-slate-700 flex items-center space-x-3">
      <div class="w-5 h-5 border-2 border-red-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs text-slate-300 font-semibold">Running OpenRouter Vision OCR document extraction...</span>
    </div>
  `;

  try {
    const res = await fetch('/api/kyc/upload-and-verify', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.extracted_raw) {
      if (data.extracted_raw.gstin) document.getElementById('kyc-input-gstin').value = data.extracted_raw.gstin;
      if (data.extracted_raw.udyam) document.getElementById('kyc-input-udyam').value = data.extracted_raw.udyam;
      if (data.extracted_raw.bill_date) document.getElementById('kyc-input-date').value = data.extracted_raw.bill_date;
    }
    renderKycResults(data);
  } catch (err) {
    console.error("Upload error:", err);
  }
}

function renderKycResults(auditData) {
  document.getElementById('kyc-result-title').textContent = `Applicant: ${auditData.applicant_name || 'Verification Report'}`;

  // Overall badge
  const badgeContainer = document.getElementById('kyc-overall-badge-container');
  const proceedBtn = document.getElementById('btn-proceed-tracker');

  if (auditData.overall_status === 'APPROVED') {
    badgeContainer.innerHTML = `
      <span class="px-3.5 py-1 rounded-full text-xs font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
        ✓ APPROVED
      </span>
    `;
    proceedBtn.classList.remove('opacity-50', 'pointer-events-none');
  } else {
    badgeContainer.innerHTML = `
      <span class="px-3.5 py-1 rounded-full text-xs font-black uppercase tracking-wider bg-red-500/20 text-red-400 border border-red-500/30">
        ✕ REJECTED
      </span>
    `;
    proceedBtn.classList.add('opacity-50', 'pointer-events-none');
  }

  // Check items
  const checksContainer = document.getElementById('kyc-checks-list');
  checksContainer.innerHTML = '';

  auditData.checks.forEach(chk => {
    const isSuccess = chk.status === 'VERIFIED';
    const row = document.createElement('div');
    row.className = `p-4 rounded-xl border flex items-start justify-between space-x-3 transition ${
      isSuccess ? 'bg-slate-800/50 border-emerald-500/30' : 'bg-red-950/20 border-red-500/40'
    }`;

    row.innerHTML = `
      <div class="flex items-start space-x-3">
        <div class="w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold ${
          isSuccess ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
        }">
          ${isSuccess ? '✓' : '✕'}
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xs font-bold text-white">${chk.component}</span>
            <span class="text-[11px] font-mono text-slate-400">${chk.identifier}</span>
          </div>
          <p class="text-xs mt-1 leading-relaxed ${isSuccess ? 'text-slate-300' : 'text-red-300 font-medium'}">
            ${chk.reason}
          </p>
        </div>
      </div>
      <span class="text-[10px] font-black uppercase px-2 py-0.5 rounded shrink-0 ${
        isSuccess ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-400'
      }">
        ${chk.status}
      </span>
    `;
    checksContainer.appendChild(row);
  });

  if (window.lucide) {
    lucide.createIcons();
  }
}

// -----------------------------------------------------------------------------
// Step 5: Sales Intelligence & Lead Router (Backoffice)
// -----------------------------------------------------------------------------
async function loadBackofficeLeads() {
  const seg = document.getElementById('backoffice-segment-filter').value;
  try {
    const res = await fetch(`/api/backoffice/leads?limit=30&segment=${encodeURIComponent(seg)}`);
    const data = await res.json();
    const tbody = document.getElementById('backoffice-leads-tbody');
    tbody.innerHTML = '';

    data.leads.forEach(lead => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/40 transition';

      let badgeClass = 'bg-slate-700 text-slate-300';
      if (lead.route_badge === 'vip') badgeClass = 'bg-red-500/20 text-red-400 border border-red-500/30';
      else if (lead.route_badge === 'partner') badgeClass = 'bg-amber-500/20 text-amber-300 border border-amber-500/30';
      else if (lead.route_badge === 'digital') badgeClass = 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';

      tr.innerHTML = `
        <td class="py-2.5 px-3 font-mono text-slate-400 text-[11px]">${lead.lead_id}</td>
        <td class="py-2.5 px-3">
          <div class="font-bold text-white text-xs">${lead.business_name}</div>
          <div class="text-[10px] text-slate-400">${lead.city} (T${lead.city_tier}) · ${lead.lead_source}</div>
        </td>
        <td class="py-2.5 px-3 text-slate-300 text-xs">${lead.segment}</td>
        <td class="py-2.5 px-3 font-bold ${lead.propensity_score >= 70 ? 'text-emerald-400' : 'text-slate-300'} text-xs">
          ${lead.propensity_score}%
        </td>
        <td class="py-2.5 px-3 text-xs text-slate-300">₹${lead.predicted_mrr.toLocaleString()}</td>
        <td class="py-2.5 px-3 font-bold text-emerald-400 text-xs">₹${lead.expected_annual_value.toLocaleString()}</td>
        <td class="py-2.5 px-3">
          <span class="text-[10px] font-bold px-2.5 py-1 rounded-full ${badgeClass}">
            ${lead.routing}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Backoffice Leads error:", err);
  }
}

// -----------------------------------------------------------------------------
// Voice Assistant & Translation Helpers
// -----------------------------------------------------------------------------
function toggleVoiceAdvisor() {
  voiceAdvisorEnabled = !voiceAdvisorEnabled;
  const label = document.getElementById('voice-label');
  const btn = document.getElementById('voice-btn');
  if (voiceAdvisorEnabled) {
    label.textContent = "Voice AI: On";
    btn.classList.add('border-brand-red', 'text-red-400');
    speakNarrative();
  } else {
    label.textContent = "Voice AI: Off";
    btn.classList.remove('border-brand-red', 'text-red-400');
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }
}

function speakNarrative() {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const text = document.getElementById('rec-narrative').textContent;
  const utterance = new SpeechSynthesisUtterance(text);
  const lang = document.getElementById('lang-select').value;
  utterance.lang = lang === 'Hindi' ? 'hi-IN' : 'en-IN';
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

function onLanguageChange() {
  requestAdvisorBundle();
}

function showConsentModal() {
  document.getElementById('consent-modal').classList.remove('hidden');
}

function hideConsentModal() {
  document.getElementById('consent-modal').classList.add('hidden');
}
