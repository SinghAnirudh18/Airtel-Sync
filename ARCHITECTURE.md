# Solution Architecture: Project Udyam Digital
## Accel Business (YTL Hackathon 2026 · Enterprise Track A)

---

### 1. Executive System Overview

Project Udyam Digital is an omnichannel, digital-first acquisition, quoting, verification, and onboarding platform engineered specifically for Indian Micro, Small, and Medium Enterprises (MSMEs). It replaces manual paper visits with a **60-second conversational journey**, near-instant automated KYC, and a live activation tracker that directly tackles the main drivers of funnel drop-off and customer churn.

```mermaid
flowchart TD
    subgraph ClientLayer["1. Client Layer (Omnichannel MSME Interface)"]
        MobileWeb["Mobile Web App (Responsive)"]
        WhatsApp["WhatsApp Business Flow (API)"]
        VoiceAI["Bilingual Voice Assistant (Hi / En)"]
    end

    subgraph GenAIEngine["2. Intelligence & Recommendation Layer"]
        Advisor["Conversational AI Advisor"]
        RulesEngine["Catalogue Rule Engine (Eligibility & Dependencies)"]
        BundleOpt["Transparent Quote Builder & SLA Estimator"]
    end

    subgraph SecurityKYC["3. Smart KYC & DPDP Compliance Vault"]
        Consent["DPDP Act, 2023 Consent Gate"]
        OCR["OpenRouter Vision OCR Extraction"]
        Checksum["Mod-36 GSTIN Checksum Validator"]
        MockGST["Mock GST Registry API (3,534 records)"]
        MockUdyam["Mock Udyam Registry API (2,723 records)"]
        Recency["90-Day Utility Bill Benchmark (2026-07-15)"]
        Masking["Data Masking & Tokenization (37XXXXXXXXX1ZS)"]
    end

    subgraph Fulfillment["4. Activation & Lifecycle Tracking"]
        Tracker["Live Order & Milestone Tracker"]
        TechDispatch["Field Engineer Dispatch & OTP Verification"]
        SLAEngine["Guaranteed SLA Countdown (48h Turnaround)"]
    end

    subgraph SalesEngine["5. Back-Office ML Lead Propensity & Router"]
        PropensityML["Voting Ensemble Model (AUC 0.83)"]
        EVRouter["Expected Annual Value (EV) Optimizer"]
        VIP["Field Sales VIP (High EV)"]
        Partner["Channel Partner (Warm Drop-offs)"]
        Digital["100% Digital Self-Serve (Micro)"]
    end

    ClientLayer --> GenAIEngine
    GenAIEngine --> SecurityKYC
    SecurityKYC --> Fulfillment
    SecurityKYC -.-> SalesEngine
    SalesEngine --> VIP
    SalesEngine --> Partner
    SalesEngine --> Digital
```

---

### 2. End-to-End Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Owner as MSME Business Owner
    participant App as Udyam Digital UI / WhatsApp
    participant Advisor as GenAI Advisor & Rules Engine
    participant KYC as Smart KYC & OCR Engine
    participant Registry as Mock Registries (GST & Udyam)
    participant Tracker as Order Tracker & Dispatch

    Owner->>App: Answers 4 quick questions (Type, Size, Needs)
    App->>Advisor: Query recommended bundle
    Advisor->>Advisor: Evaluate catalogue rules (min employees, sites, tier, dependencies)
    Advisor-->>App: Return personalized bundle + plain-language narrative
    Owner->>App: Customizes add-ons (SIMs, CCTV) & accepts quote
    App->>KYC: Displays DPDP Act consent notice
    Owner->>KYC: Grants consent & uploads document images
    KYC->>KYC: Extract GSTIN, Udyam, and Utility Bill date via Vision OCR
    KYC->>KYC: Run Mod-36 checksum & format regex
    KYC->>Registry: Lookup GSTIN status & Udyam registration
    KYC->>KYC: Check utility bill recency (<= 90 days from 2026-07-15)
    Registry-->>KYC: Registry status verified
    KYC-->>App: Instant KYC Approved with masked tokens
    App->>Tracker: Provision order and reserve DP-box port
    Tracker-->>Owner: Display live order timeline, engineer profile & OTP
```

---

### 3. Core Architectural Modules

#### A. GenAI Conversational Advisor
* **Multi-turn Context:** Translates natural language business descriptions into technical telecommunication parameters (bandwidth, static IPs, failover links, cloud POS).
* **Hard Rule Constraint Engine:** Validates that products meet catalogue thresholds:
  * Minimum site and employee requirements.
  * City tier eligibility (Tier 1 vs Tier 2).
  * Dependency resolution (`requires_one_of` guarantees dependent products like Cloud Telephony or SD-WAN have underlying connectivity).
  * GSTIN requirement flags (`requires_gstin = 'Y'`).

#### B. Smart Instant KYC Engine
* **Vision OCR Processing:** Multimodal LLM / OCR parsing extracts registration tokens and utility bill issue dates with zero manual keying.
* **Algorithmic Validation:**
  * **Mod-36 Checksum:** Verifies 15-character GSTIN integrity, handling synthetic PAN entity character `'X'`.
  * **Registry Lookups:** Queries `mock_gstin_registry.csv` and `mock_udyam_registry.csv` to ensure active status and flag cancelled entities (e.g. Applicant K01).
  * **Recency Benchmark:** Computes date difference against `2026-07-15`, rejecting address proofs older than 90 days (e.g. Applicants K01, K11, K12).

#### C. Live Milestone Order Tracker
* **Real-time Status Feed:** Replaces asynchronous, paper-based waiting with 5 clear stages: Order Confirmed $\to$ KYC Cleared $\to$ Line Feasibility $\to$ Technician Dispatched $\to$ Activated.
* **Technician Handshake:** Direct engineer contact details and OTP verification guarantee zero confusion during line laying and hardware provisioning.
* **Business Impact:** Directly eliminates the 3.0 follow-up calls per lead that drove 28% of 6-month churn.

#### D. ML Propensity & Expected Value (EV) Lead Router
* **Ensemble Architecture:** Soft Voting Ensemble combining Random Forest, Gradient Boosting, HistGradientBoosting, and L2-regularized Logistic Regression (ROC-AUC ~ 0.83).
* **Expected Value Formulation:**
  $$\text{Expected Annual Value (EV)} = P(\text{Activation}) \times \text{Predicted MRR} \times 12$$
* **Omnichannel Triage:**
  1. **Field Sales VIP:** High propensity + high predicted MRR ($\ge$ ₹6,000/mo).
  2. **Channel Partner Outreach:** Warm digital drop-offs (leads stalled at quote or KYC).
  3. **100% Self-Serve Digital:** Micro-retail and price-sensitive leads, operating at near-zero CAC.

