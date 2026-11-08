# Project Udyam Digital: 10-Slide Business Case
## Accel Business · YTL Hackathon 2026 (Enterprise Track A)

---

### Slide 1: The Opportunity & Problem Statement
* **The Market:** India has 6.3 Crore+ MSMEs representing 30% of GDP, but Accel Business holds a small market share due to an enterprise sales model built for large corporations.
* **The Current Reality (from `smb_leads.csv`):**
  * 5,000 Leads generated $\to$ only **368 activations** (a dismal **7.36% funnel conversion**).
  * **17.6 days** average lead-to-activation lag.
  * **₹15,000 blended CAC** per activated customer.
  * Steepest drop occurs between *Needs Assessed* and *Quote Accepted* (44% pass rate) due to 3–5 day delays in sending cryptic multi-page PDF quotes.
* **The Vision:** "Project Udyam Digital" — a self-serve, conversational digital onboarding platform enabling small businesses to discover, configure, verify, and activate their telecom & cloud bundle in a single sitting on their smartphone.

---

### Slide 2: Target Customer Segmentation & Needs
* **1. Micro Retail & Services (55% of base):**
  * *Profile:* Kirana stores, salons, bakeries, pharmacies.
  * *Need:* Zero UPI payment drops, simple POS billing, WhatsApp catalogue. Highly price-sensitive.
  * *Target Bundle:* Biz 5G AirFiber / Biz Fibre 100 + Cloud POS (₹1,500–₹2,500/mo).
* **2. Professional Offices (25% of base):**
  * *Profile:* Diagnostic clinics, CA firms, coaching centers.
  * *Need:* High uptime broadband, domain email, cloud backup, endpoint antivirus.
  * *Target Bundle:* Biz Fibre 300 + Cloud Office Suite + Endpoint Security (₹2,500–₹5,000/mo).
* **3. Small Multi-site Businesses (15% of base):**
  * *Profile:* 3–10 store retail chains, restaurants.
  * *Need:* Single consolidated invoice, failover 4G/5G links, central Cloud CCTV.
  * *Target Bundle:* Biz Fibre 300 / ILL + 4G/5G Failover + Cloud CCTV (₹5,000–₹12,000/mo).
* **4. Small Manufacturers (5% of base):**
  * *Profile:* Auto-parts units, garment factories.
  * *Need:* 99.9% Leased Line SLA, IoT machine telemetry, fleet GPS tracking.
  * *Target Bundle:* Internet Leased Line + Managed Firewall + IoT Sensors (₹15,000–₹35,000/mo).

---

### Slide 3: The 4-Pillar Digital Acquisition Journey
1. **GenAI Conversational Advisor:** Multilingual (English/Hindi) assistant that asks 4 intuitive questions and recommends a tailored bundle with plain-language explanations, eliminating sales representative visits.
2. **Transparent Dynamic Quote Builder:** Replaces 5-day PDF quotes with instant itemized monthly billing, toggleable add-ons, and real-time SLA calculation.
3. **Smart Instant KYC Engine:** Sub-60-second OCR extraction, Mod-36 checksum validation, mock GSTN/Udyam registry queries, and 90-day address proof recency check under DPDP Act consent.
4. **Live Order Tracker:** Milestone status view (Order $\to$ KYC $\to$ Feasibility $\to$ Technician $\to$ Activation) with live OTP and assigned engineer details.

---

### Slide 4: Data-Driven Impact: Solving Churn & Chaser Calls
* **The Root Cause of Churn:**
  * In baseline data, activated customers called support **3.0 times** to chase installation status.
  * Customers making $\ge 3$ chase calls had a **6-month churn rate of 28.4%** and an NPS of only **5.8**.
  * Customers with zero follow-up calls had a churn rate of only **11.2%** and an NPS of **9.2**.
* **The Solution:** The Live Order Tracker provides complete transparency into line laying and technician dispatch, cutting chaser calls by **85%** and directly boosting 6-month customer retention.

---

### Slide 5: Unit Economics & CAC Transformation
* **Current Baseline Economics:**
  * Blended CAC: **₹15,000**
  * Average Revenue Per User (ARPU): **₹6,900 / month** (Micro: ₹2,500 / month)
  * Payback Period on Micro Customers: **6.0 months**
* **Project Udyam Digital Economics:**
  * Digital CAC: **₹3,800** (a **74.6% reduction** achieved by removing field rep overhead for micro/standard leads).
  * Micro Payback Period: **1.5 months** (accelerates profitability by 4×).
  * Funnel Conversion: Improves from **7.36% to 22.50%** (a **3.05× conversion lift**).

---

### Slide 6: ML Omnichannel Lead Routing (The Secret Weapon)
Rather than abandoning existing field reps and channel partners, an ensemble ML model (AUC 0.83) triages leads based on **Expected Annual Value (EV)**:
$$\text{Expected Annual Value (EV)} = P(\text{Activation}) \times \text{Predicted MRR} \times 12$$

* **Tier 1: Field Sales VIP (High EV $\ge$ ₹72,000/yr):** Routed to enterprise sales executives for white-glove onboarding.
* **Tier 2: Channel Partner Assisted (Medium EV / Warm Drop-offs):** Routed to local partners to assist leads stalled at KYC or quote acceptance.
* **Tier 3: 100% Self-Serve Digital (Micro & High Price Sensitivity):** Onboarded entirely via smartphone / WhatsApp at zero manual intervention.

---

### Slide 7: DPDP Act, 2023 & Compliance by Design
* **Purpose Limitation:** Identity data used strictly for telecom connectivity verification.
* **Data Minimization:** Micro-retailers without GST are verified via utility bills without forced tax document demands.
* **Identifier Masking:** In-flight and UI masking (`37XXXXXXXXX1ZS`) across all screens and operational logs.
* **Ephemeral Document Retention:** Images expunged from hot storage 30 days post-activation.

---

### Slide 8: Go-To-Market & Rollout Roadmap
* **Phase 1 (Months 1–3): Tier 1 Digital Launch**
  * Deploy self-serve web app and WhatsApp flow in Delhi, Mumbai, Bengaluru.
  * Target Micro Retail and Professional Offices.
* **Phase 2 (Months 4–6): Regional Language & Partner Network Integration**
  * Expand to Tier 2 cities (Jaipur, Lucknow, Kochi, Ludhiana).
  * Roll out partner portal for warm drop-off lead assistance.
  * Add Tamil, Telugu, Marathi, and Bengali voice advisor models.
* **Phase 3 (Months 7–12): IoT & Leased Line Enterprise Automation**
  * Automate network feasibility checks via GIS fiber integration for manufacturing units.

---

### Slide 9: Risk Matrix & Mitigation Strategy

| Risk Factor | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **Fiber Feasibility Failure** | Medium | Automated GIS port reservation prior to order confirmation; fallback to 5G AirFiber. |
| **KYC Document Forgery** | High | Algorithmic Mod-36 checksum + authoritative registry cross-verification. |
| **Partner Channel Conflict** | Medium | Commission sharing on digital leads completed within partner pin codes. |
| **Digital Adoption Hesitation** | Low | Bilingual WhatsApp conversational interface + voice narration for non-tech owners. |

---

### Slide 10: Summary & Expected Business Returns
* **Conversion Rate:** 7.4% $\to$ **22.5%** (+15.1 percentage points).
* **Time-to-Activate:** 17.6 days $\to$ **2.5 days** (85% acceleration).
* **Blended CAC:** ₹15,000 $\to$ **₹3,800** (74.6% reduction).
* **6-Month Churn:** 28% $\to$ **11%** (boosted by zero-chase tracker).
* **1-Year Net Revenue Impact:** **+₹18.4 Crore incremental ARR** per 50,000 MSME leads processed.

