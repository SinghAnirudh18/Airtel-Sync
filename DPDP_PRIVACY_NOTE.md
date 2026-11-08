# Data Privacy & Security Architecture Note
## Compliance with the Digital Personal Data Protection (DPDP) Act, 2023
### Accel Business · Project Udyam Digital

---

### 1. Architectural Philosophy: Privacy by Design
Project Udyam Digital transitions Accel Business from legacy physical paperwork and unencrypted email document collection to an automated, cryptographically secured digital onboarding funnel. The system is designed to adhere strictly to India's **Digital Personal Data Protection (DPDP) Act, 2023** and Department of Telecommunications (DoT) KYC guidelines.

---

### 2. Core DPDP Compliance Principles Implemented

#### Principle 1: Purpose Limitation (Section 5 & 6)
* **Specific Telecom Purpose:** Personal data, business registrations (GSTIN/Udyam), and identity proofs are gathered solely for the purpose of verifying subscriber identity, confirming physical installation feasibility, and generating commercial billing invoices.
* **No Secondary Monetization:** Customer credentials are cryptographically siloed and barred from cross-functional data sharing, external marketing, or advertising brokers.

#### Principle 2: Data Minimization (Section 4)
* **Tier-Specific Collection:** Micro-retailers (representing 55% of the MSME base) purchasing base broadband and point-of-sale terminals are not forced to provide GST registration or audited balance sheets. A recent utility bill suffices for address verification.
* **No Excessive Identity Scans:** Aadhaar numbers are never stored in raw form; only the last 4 digits are retained in audit records, accompanied by digital masking.

#### Principle 3: Granular, Multilingual Consent Management (Section 6)
* **Explicit Pre-KYC Consent Gate:** Before any document upload or registry query occurs, an explicit consent notice is presented in the customer's selected language (English or Hindi).
* **Transparent Notice:** The consent notice explicitly details:
  1. Exactly what data points are collected.
  2. How registry APIs (GSTN and Udyam) are queried.
  3. The right to withdraw consent and request data erasure prior to activation.

#### Principle 4: Identifier Masking & Tokenization
* **In-Flight & UI Masking:** Sensitive tax and identity tokens are masked across all client interfaces and operational logs:
  * **GSTIN:** `37XXXXXXXXX1ZS` (First 2 state digits and final 4 checksum digits visible; embedded PAN masked).
  * **Udyam Number:** `UDYAM-WB-XXXX330`.
  * **Signatory Phone:** `+91 98450 XXXXX`.
* **Database Tokenization:** Raw identifiers are hashed using salt-derived SHA-256 tokens for deduplication checks, preventing plain-text data exposure.

#### Principle 5: Storage Limitation & Automated Document Purge
* **Ephemeral Hot Storage:** Uploaded specimen images (utility bills, certificates) reside in an ephemeral processing cache only long enough to perform OCR verification.
* **30-Day Purge SLA:** Statutory telecom audit snapshots are encrypted with AES-256 and scheduled for automatic deletion 30 days after commercial activation, retaining only the cryptographically signed audit log.

---

### 3. Security Engineering Controls

| Threat Vector | Mitigation Control | DPDP Act Alignment |
| :--- | :--- | :--- |
| **Data in Transit Interception** | TLS 1.3 encryption across all API routes, WebSockets, and external OCR calls. | Section 8(5) - Reasonable Security Safeguards |
| **Unauthorized Staff Exposure** | Role-Based Access Control (RBAC) with masked views in the sales/partner back-office. | Section 8(5) - Access Control & Confidentiality |
| **Fraudulent Alteration** | Algorithmic Mod-36 checksum verification + authoritative national registry cross-checks. | Section 8(2) - Accuracy of Personal Data |
| **Data Breach Notification** | Automated audit logging with immediate alert pipelines to the Data Protection Officer (DPO). | Section 8(6) - Personal Data Breach Reporting |

---

### 4. Summary for Regulators & Auditors
By eliminating field reps taking unencrypted photos of PAN and electricity bills on personal mobile devices, Project Udyam Digital achieves a **100% auditable, consent-bound, and masked digital verification workflow** that sets a new industry standard for enterprise telecom onboarding.

