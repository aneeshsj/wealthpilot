# Internal Risk Grading and Pricing Grid

# ASHVA CAPITAL INTERNAL CREDIT POLICY MANUAL
**Document Ref:** ACP-2024-RM-04  
**Effective Date:** 01 November 2024  
**Author:** Chief Credit Officer  
**Subject:** Risk Grading Methodology and Underwriting Parameters  

---

## SECTION 4. RISK GRADING METHODOLOGY AND PRICING MATRIX

### 4.1 Purpose
This section establishes the mandatory risk grading methodology for all commercial and retail credit facilities originated by Ashva Capital. Underwriters and automated underwriting systems (AUS) are strictly bound by the quantitative thresholds, pricing spreads, maximum tenors, and collateral coverage ratios defined herein. 

### 4.2 Grading Assignment Logic
4.2.1 The underwriting system shall evaluate every credit application against the Debt Service Coverage Ratio (DSCR) and the primary bureau score.
4.2.2 An application shall be assigned the highest risk grade (from A1 descending to D2) for which it simultaneously satisfies both the minimum DSCR threshold and the minimum bureau score threshold.
4.2.3 Grade D2 shall act as the catch-all category for any application failing to meet the minimum criteria of Grade D1.

### 4.3 Risk Grade Matrix
The eight internal credit grades (A1 through D2) shall adhere strictly to the parameters set forth in Table 4.1 below.

#### Table 4.1: Ashva Capital Risk Grade Matrix
| Grade | Minimum DSCR | Minimum Bureau Score | Interest Rate Spread (over Base Rate) | Indicative Interest Rate (Base = 9.50%) | Maximum Permitted Tenor | Minimum Collateral Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | 2.00 | 750 | 150 bps | 11.00% | 84 months | 0% |
| **A2** | 1.75 | 700 | 200 bps | 11.50% | 72 months | 25% |
| **B1** | 1.50 | 650 | 275 bps | 12.25% | 60 months | 50% |
| **B2** | 1.35 | 600 | 350 bps | 13.00% | 60 months | 75% |
| **C1** | 1.25 | 550 | 450 bps | 14.00% | 48 months | 100% |
| **C2** | 1.15 | 500 | 575 bps | 15.25% | 36 months | 125% |
| **D1** | 1.00 | 450 | 725 bps | 16.75% | 24 months | 150% |
| **D2** | N/A (Below D1) | N/A (Below D1) | 900 bps | 18.50% | 12 months | 150% |

### 4.4 Interest Rate Calculation
4.4.1 The Ashva Capital lending base rate is fixed at 9.50 percent per annum.
4.4.2 The indicative interest rate charged to the borrower shall be calculated as the sum of the lending base rate (9.50%) and the applicable risk grade spread specified in Table 4.1.

### 4.5 Tenor Constraints
4.5.1 The final tenor offered to a borrower shall not exceed the maximum tenor permitted for the assigned risk grade under Table 4.1.
4.5.2 The final tenor shall also be constrained by the maximum tenor permitted for the borrower's specific industry sector as defined in Appendix B of this manual, whichever is shorter.
4.5.3 Where a borrower requests a tenor exceeding either the grade limit or the sector limit, the underwriting system shall generate reason code **TENOR_ABOVE_LIMIT** and route the file to a designated human reviewer.

### 4.6 Collateral Coverage Requirements
4.6.1 Collateral cover is expressed as a mandatory percentage of the total approved loan amount. Valuations shall be determined exclusively by Ashva Capital-empaneled independent valuers.
4.6.2 If the verified market value of acceptable collateral pledged is less than the required percentage specified in Table 4.1 for the assigned grade, the system shall generate reason code **COLLATERAL_SHORTFALL**.
4.6.3 Any application triggering reason code **COLLATERAL_SHORTFALL** shall be removed from the automated approval queue and referred to a human reviewer for exception processing or credit structuring.

### 4.7 Automated Approval (Auto-Approval) Parameters
4.7.1 Applications qualifying for grades A1, A2, B1, B2, and C1 are eligible for auto-approval by the underwriting system, provided the requested loan amount does not exceed 1,000,000 rupees (INR 10,00,000).
4.7.2 Grades C2, D1, and D2 are strictly ineligible for auto-approval regardless of the requested loan size. Any application falling into Grade C2 or lower must be manually reviewed and signed off by a human credit officer holding the requisite delegation authority.
