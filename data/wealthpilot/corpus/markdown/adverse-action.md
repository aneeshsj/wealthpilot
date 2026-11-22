# Adverse Action and Applicant Communication Standard

# ASHVA CAPITAL INTERNAL CREDIT POLICY MANUAL
## CHAPTER 14: CREDIT DECLINE, REASON CODING, AND ADVERSE ACTION GOVERNANCE
**Document ID:** AC-POL-CR-2024-14  
**Classification:** Internal - Restricted  
**Effective Date:** 01 November 2024  
**Applies To:** All Credit Underwriters, Risk Officers, and Automated Decision Systems  

---

### 14.1 PURPOSE AND SCOPE
This chapter establishes the binding standards for declining credit applications submitted to Ashva Capital. Adherence to these provisions is mandatory for all manual and automated underwriting decisions. Deviation from these rules constitutes a severe compliance violation subject to internal audit review.

### 14.2 MAXIMUM PERMITTED TURN turnaround
14.2.1 Every credit application must reach a terminal status (approval, formal decline, or information hold) within exactly **forty-eight (48) hours** of initial digital submission through the Ashva Capital Origination Portal.  
14.2.2 For applications flagged under Section 14.7 (`FRAUD_HOLD`), the maximum turnaround for investigative clearance or formal decline is extended to **one hundred twenty (120) hours**.  
14.2.3 Failure by an assigned underwriter to issue a decision within these timelines results in automatic routing of the file to the Deputy Chief Credit Officer for immediate disposition.

### 14.3 MANDATORY CONTENT OF A DECLINE NOTICE
14.3.1 Every formal decline notice issued to an applicant must be generated via the Ashva Capital automated document engine and contain exactly the following fields:
1. **Ashva Capital Reference ID:** The unique alphanumeric identifier assigned at intake.
2. **Date of Decision:** Formats as DD-MM-YYYY.
3. **Primary Decline Reason Code:** Exactly one code selected from the list in Section 14.4.
4. **Secondary Decline Reason Code (Optional):** At most one additional code from Section 14.4, applicable only if multiple independent underwriting thresholds were breached.
5. **Appraisal and Bureau Summary:** Key numerical metrics that triggered the decline (e.g., verified DSCR, bureau score, current ratio).
6. **Appeal Window:** Explicit notice of the appeal deadline pursuant to Section 14.5.

### 14.4 APPROVED REASON CODES AND DEFINITIONS
Underwriters are strictly prohibited from citing any reason, text, or rationale that does not appear in the following exhaustive list. 

| Reason Code | Technical Definition & Trigger Threshold |
| :--- | :--- |
| **APPROVED_STANDARD** | Meets every underwriting standard. *(Note: Must never be used on a decline notice; reserved solely for conditional system logging).* |
| **DSCR_BELOW_FLOOR** | Debt-service coverage ratio is below the sector floor (calculated as Net Operating Income divided by Total Debt Service). |
| **THIN_FILE** | Fewer than 24 months operating history, or no active commercial trade line on bureau file. |
| **ADVERSE_BUREAU** | Primary commercial bureau score is strictly below 450. |
| **PAST_DELINQUENCY** | Existence of any 90-day delinquency within the past 36 months, or 500,000 rupees or more written off across any commercial or promoter credit facility. |
| **DOC_MISMATCH** | Declared revenue differs from verified revenue by more than 10 percent. |
| **SECTOR_CAP** | The applicant's industry sector has reached its maximum permitted share of the total Ashva Capital loan book. |
| **LEVERAGE_ABOVE_LIMIT** | Total debt after the requested loan exceeds 6.0 times EBITDA. |
| **LIQUIDITY_BELOW_FLOOR** | Current ratio (Current Assets divided by Current Liabilities) is below 1.10. |
| **TENOR_ABOVE_LIMIT** | Requested loan tenor is longer than the assigned credit grade or sector policy allows. |
| **COLLATERAL_SHORTFALL** | Pledged collateral is valued at less than the minimum cover percentage required for the assigned credit grade. |
| **FRAUD_HOLD** | Declared revenue differs from verified revenue by more than 25 percent, or material document falsification is detected. |
| **GRADE_REQUIRES_REVIEW** | Assigned credit grade is C2 or D1; policy mandates that these grades are never approved automatically. |
| **INCOMPLETE_APPLICATION** | A figure, document, or data point needed to test a mandatory credit standard was not supplied within the cure window. |
| **WITHDRAWN_BY_APPLICANT** | The applicant formally withdrew the application prior to final underwriting completion. |

14.4.1 **Prohibition on Unlisted Reasons:** Underwriters, automated systems, and customer service personnel are strictly prohibited from inventing, paraphrasing, or citing any reason not codified in this section. Any communication containing an unauthorized reason string must be retracted and re-issued within **four (24) hours**.

### 14.5 APPEAL PROCESS AND DEADLINE
14.5.1 An applicant may appeal a decline decision by submitting a formal appeal package through the Ashva Capital Portal.  
14.5.2 The strict deadline for submitting an appeal is **fourteen (14) calendar days** from the date stamped on the decline notice. Appeals received after the 14-day mark shall be rejected automatically without review.  
14.5.3 Appeals shall be adjudicated by the Credit Appeals Committee, comprising the Chief Credit Officer, Head of Risk, and a senior regional credit manager who was not involved in the original underwriting decision.  
14.5.4 The Credit Appeals Committee must render a final binding decision within **seven (7) business days** of receiving a complete appeal package.

### 14.6 RECORD RETENTION PERIOD
14.6.1 All decline notices, underlying credit memorandums, financial statements, bureau reports, communication logs, and appeal documentation must be retained in the Ashva Capital secure digital archive for a mandatory period of **seven (7) years** from the date of the decline notice.  
14.6.2 Under no circumstances may records associated with a decline coded as `FRAUD_HOLD` be purged or destroyed prior to the expiration of **ten (10) years**.  
14.6.3 Automated deletion routines must be configured by the Information Security Department to prevent premature purging of records governed by this section.

---
**Approved by:**  
*V. R. Swaminathan*  
Chief Credit Officer, Ashva Capital  
01 November 2024
