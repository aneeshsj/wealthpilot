# Document Verification and Fraud Screening Procedure

# ASHVA CAPITAL INTERNAL CREDIT POLICY MANUAL
**SECTION 4: DOCUMENT VERIFICATION AND FRAUD DETECTION PROCEDURES**
**Document Ref:** ASHVA-CP-2024-402  
**Effective Date:** January 15, 2024  
**Applies To:** All Credit Underwriters, Fraud Operations Unit, and Risk Management  

---

## 4.1 PURPOSE AND MANDATORY DOCUMENT SET

1.1 This section establishes the mandatory document collection and verification protocols for all Small and Medium Enterprise (SME) term loan and working capital applications processed by Ashva Capital.

1.2 Underwriters are strictly prohibited from submitting any credit memo to the Credit Committee without acquiring and validating the complete mandatory document set specified below.

1.3 **Mandatory Document Set for SME Applications:**
1. **Primary Bank Statements:** Twelve (12) consecutive months of primary operating account statements in native PDF format. Scanned paper copies or images are rejected automatically by the intake system.
2. **Tax Returns:** Filed Goods and Services Tax (GST) returns (Form GSTR-3B and GSTR-1) for the preceding twelve (12) months.
3. **Statutory Financials:** Audited balance sheets, profit and loss statements, and cash flow statements for the last two (2) completed financial years, bearing the UDIN (Unique Document Identification Number) where applicable.
4. **Legal Entity Documentation:** Certificate of Incorporation, Memorandum and Articles of Association, and the latest shareholding pattern certified by a company secretary or director.
5. **Identity and Address Proof:** Permanent Account Number (PAN) and verified physical address proof of the entity and all primary guarantors holding greater than twenty percent (20%) equity.

---

## 4.2 VERIFICATION STEPS FOR PRIMARY FINANCIAL SOURCES

### A. Bank Statements
2.1 Underwriters must upload native PDF bank statements directly into the AshvaVerify automated extraction engine. 
2.2 The underwriter must reconcile total monthly credits against reported revenue, identifying and discounting internal transfers, director infusions, and loan disbursements.
2.3 Any instance of balance-matching across consecutive months, recurring round-figure debits to related parties, or an end-of-month balance falling below zero for more than three (3) consecutive days requires manual investigation and a mandatory comment in the underwriting file.

### B. GST Returns
2.4 Underwriters must cross-reference GSTR-3B filings against GSTR-1 filings for all twelve (12) months.
2.5 Total taxable turnover declared in GSTR-3B must be reconciled with the sales ledger and the primary bank statement credit summation. 
2.6 A variance exceeding five percent (5%) between GSTR-3B turnover and bank credits must be documented with an explanatory CA certificate uploaded to the loan origination portal prior to underwriting progression.

### C. Audited Financials
2.7 Audited financials must be verified against the corporate registry database to confirm the filing status of the applicant entity.
2.8 Underwriters must perform ratio analysis on the audited statements, specifically validating that the Debt-to-Equity ratio does not exceed 3.5:1 and the Current Ratio is maintained at a minimum of 1.25:1, unless an explicit exception is signed off by Chief Risk Officer Marcus Vance.

---

## 4.3 REVENUE DISCREPANCY TOLERANCE THRESHOLDS

3.1 The applicant's declared annual revenue from the application form shall be compared against the verified annual revenue derived from the synthesis of bank statements, GST filings, and audited financials.

3.2 **Tolerance Thresholds:**
1. **Accepted Variance:** If the declared revenue is within ten percent (10%) of the verified revenue (inclusive of both positive and negative variances), the revenue figure is accepted without penalty.
2. **Moderate Variance (Reason Code `DOC_MISMATCH`):** If the difference between declared and verified revenue exceeds ten percent (10%) but is less than or equal to twenty-five percent (25%), the system shall automatically assign reason code `DOC_MISMATCH`. The application is halted from automated processing and routed to a human reviewer for justification.
3. **Severe Variance (Reason Code `FRAUD_HOLD`):** If the difference between declared and verified revenue exceeds twenty-five percent (25%), the system shall assign reason code `FRAUD_HOLD` and immediately trigger a fraud hold, suspending all processing and locking the file.

---

## 4.4 FRAUD HOLD TRIGGERS AND RED FLAGS

4.1 The occurrence of any single red flag listed below mandates the immediate application of a `FRAUD_HOLD` status, regardless of whether the revenue discrepancy threshold has been breached.

4.2 **Mandatory Fraud Hold Triggers:**
1. **Metadata Manipulation:** PDF bank statements or GST filings exhibiting creation or modification timestamps within 48 hours of application submission, or displaying evidence of digital tampering via PDF editing software.
2. **Shell Company Indicators:** Operating accounts showing zero or nominal operational utility (utility bills, payroll debits, supplier payments) combined with large, immediate-pass-through tranches of funds.
3. **Director Disqualification:** Any primary guarantor or director flagged on the national corporate fraud registry or insolvency database.
4. **GST Counterparty Mismatch:** Greater than thirty percent (30%) of reported B2B GST transactions executed with shell entities or non-existent GSTINs identified via the tax intelligence module.
5. **Revenue Discrepancy:** Any revenue variance exceeding twenty-five percent (25%), as defined in Section 3.2, Item 3.

---

## 4.5 ESCALATION PATH AND EXCEPTION HANDLING

5.1 **Standard Discrepancy Escalation (`DOC_MISMATCH`):**
1. Files flagged with `DOC_MISMATCH` are routed to the Senior Underwriter queue.
2. The Senior Underwriter has four (4) business hours to request clarifying documentation from the applicant via the secure portal.
3. The applicant has forty-eight (48) hours to provide documentation. Failure to do so results in auto-declination of the application under policy rule `ERR-REV-09`.

5.2 **Fraud Hold Escalation (`FRAUD_HOLD`):**
1. Files placed on `FRAUD_HOLD` are automatically locked and transferred to the Fraud Operations Unit overseen by Fraud Operations Manager Sarah Lin.
2. The Fraud Operations Unit must conduct an investigative review within twenty-four (24) hours of the hold being triggered.
3. If the fraud triggers are validated as intentional misrepresentation, the file must be formally declined, the applicant added to the internal Ashva Capital Negative Database, and a Suspicious Transaction Report (STR) filed with the relevant financial intelligence unit within forty-eight (48) hours.
4. **Exception Authority:** A `FRAUD_HOLD` can only be overridden and cleared if Sarah Lin (Fraud Operations Manager) and Marcus Vance (Chief Risk Officer) jointly execute a written clearance waiver confirming the anomaly was the result of a documented accounting error rather than intentional fraud. No single individual possesses the authority to clear a `FRAUD_HOLD`.
