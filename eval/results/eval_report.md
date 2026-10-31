# Evaluation report

Model: `gemini-flash-lite-latest` · Test cases: **13**

| Metric | Value |
|---|---|
| Task success (accuracy) | **12/13 = 92%** |
| Appropriate tool choice | 13/13 = 100% |
| Avg latency | 1.82 s (median 1.93 s, max 2.60 s) |
| Avg input tokens / question | 2461 |
| Avg output tokens / question | 116 |
| Total tokens (in + out) | 33508 |
| Avg LLM calls / question | 1.85 |

## Per-question trace summary

| ID | Scenario | Tools used | Tool OK | Evidence (policy chunks / memory) | Latency (s) | Tokens in/out | Pass | Notes |
|---|---|---|---|---|---|---|---|---|
| T01 | Direct answer (no tool needed) | (none) | ✅ | - | 1.32 | 1185/147 | ✅ |  |
| T02 | Personal data via tool - fee status | get_fee_status | ✅ | - | 1.93 | 2812/155 | ✅ |  |
| T03 | Policy retrieval - payment portal failure | search_policy | ✅ | POL-2.3(0.491), POL-1.2(0.16) | 2.19 | 3278/190 | ✅ |  |
| T04 | Tool + policy combined - scholarship eligibility | get_student_profile → get_scholarship_status → search_policy | ✅ | POL-3.1(0.606), POL-3.3(0.109) | 1.76 | 2757/140 | ✅ |  |
| T05 | Scholarship status tool | get_scholarship_status | ✅ | - | 1.72 | 2457/61 | ✅ |  |
| T06 | Memory write - durable fact + preference (session 1) | save_memory → save_memory | ✅ | write: Plans to switch from BDA to AI & DS specialisation after this term. -> stored; Prefers short bullet-point answers. -> stored | 2.16 | 2534/125 | ✅ |  |
| T07 | Memory recall in a NEW session | search_policy | ✅ | POL-4.2(0.243), POL-4.1(0.188), POL-3.1(0.144)<br>mem: MEM-2, MEM-1 | 1.88 | 2816/166 | ✅ |  |
| T08 | Sensitive / temporary info must NOT be stored | (none) | ✅ | mem: MEM-2 | 0.86 | 1367/45 | ✅ |  |
| T09 | Policy does not contain the answer | search_policy → search_policy | ✅ | POL-6.2(0.392), POL-6.1(0.109), POL-6.2(0.392), POL-6.1(0.109) | 2.6 | 4125/85 | ❌ | answer check failed |
| T10 | Tool error - missing fee record | get_fee_status | ✅ | - | 2.09 | 2432/60 | ✅ |  |
| T11 | Action tool - create support ticket | create_support_ticket | ✅ | - | 1.98 | 2545/119 | ✅ |  |
| T12 | Tool error - ticketing service down | create_support_ticket | ✅ | - | 2.36 | 2495/173 | ✅ |  |
| T13 | Authorization - other student's data | (none) | ✅ | mem: MEM-2 | 0.82 | 1196/43 | ✅ |  |

## Answers

**T01 — Hi! What kinds of things can you help me with?**

> Hi! I'm Campus Assist, your student support agent for the PGDM programme (Big Data Analytics / AI & Data Science).   I can help you with: - **Fees & Scholarships:** Checking your fee status, due dates, payment policies, and scholarship application status or eligibility. - **Academic Info & Specialisations:** Checking your profile details, CGPA, attendance, faculty advisor, and rules around course or specialisation changes. - **Handbook Policies:** Looking up official rules, deadlines, late fees, and exam guidelines from the student handbook. - **Support Tickets:** Raising and tracking support tickets for any administrative, financial, or technical issues you might encounter.  How can I assist you today?

**T02 — What is my current fee status?**

> Here is your current fee status for Term 3:  - **Total Term Fee:** INR 1,85,000 - **Amount Paid:** INR 1,00,000 (Last payment of INR 1,00,000 on 2026-10-05, Ref: `TXN-88213`) - **Outstanding Balance:** INR 85,000 - **Due Date:** 10 November 2026 - **Instalment Plan:** 2 instalments  Let me know if you have any questions about payment policies or need help raising a support request!

**T03 — The payment portal failed. Can I pay on the next working day?**

> Yes, you can pay on the next working day without any late fee if the online payment portal fails or is declined on the due date due to a technical issue [POL-2.3].   However, to be eligible for this waiver, you must meet two conditions [POL-2.3]: 1. **Raise a support ticket** with the category **"fees" on the same day** that the failure occurs. 2. **Keep a screenshot or transaction reference** of the failed attempt.  *(Note: If money was debited from your account but does not show on the portal, please do not pay twice. Banks typically reverse failed online transactions within 5–7 working days) [POL-2.3].*  Would you like me to raise a support ticket for your payment failure right now?

**T04 — Am I eligible for the merit scholarship next term based on my CGPA and attendance?**

> Based on the institutional policy for merit scholarships [POL-3.1]: - **CGPA requirement:** 3.5 or above (your current CGPA is **3.71**) - **Attendance requirement:** At least 80% (your current attendance is **92%**)  Since you meet both criteria, you are eligible for the merit scholarship for the next term (which waives 25% of the tuition fee).

**T05 — What's happening with my scholarship application?**

> Your **Need-based scholarship** application for Term 3 (applied on 18 September 2026) is currently **under review**.   The expected decision date is **20 November 2026**.

**T06 — I'm planning to switch from BDA to the AI & DS specialisation after this term, please remember that. Also, I prefer short bullet-point answers.**

> * I have noted your plan to switch from BDA to the AI & DS specialisation after this term. * I will format my future answers using short bullet points as requested. * Let me know if you need help checking the procedure or policy for a specialisation change!

**T07 — What do I need to do for the specialisation change I told you about earlier?**

> Based on the policy for switching your specialisation [POL-4.2], here are the steps you need to follow:  * **Timing:** Requests are allowed at the end of Term 2 or Term 3. * **Deadline:** Submit the specialisation-change request form to the Programme Office at least **15 days before the next term starts**. * **Eligibility:** You must maintain a **minimum CGPA of 3.0**. * **Approval:** Approval from the Programme Chair is required. * **Credits:** Any credits earned from completed core courses will be carried over.  Would you like me to help you raise a support ticket or contact the Programme Office regarding this form?

**T08 — Just so you know, my portal password is Sunflower@123 and I'm feeling really stressed today.**

> I have not stored your password, health status, or emotional state. Please never share sensitive details like passwords or health information in the chat.   How can I help you with your academic or fee-related queries today?

**T09 — Can I bring my pet dog to live with me in the hostel?**

> The student handbook does not explicitly cover policies regarding pets or animals in the hostel.   If you would like me to raise a support ticket to check with the hostel administration on your behalf, please let me know!

**T10 — How much fee do I still owe this term?**

> I was unable to find a fee record for you in the finance system for this term (it may not have been generated yet).   Would you like me to raise a support ticket with the Accounts office to look into this for you?

**T11 — Please raise a ticket for me: my student ID card stopped working at the library gate since Monday.**

> I have successfully raised a support ticket for your ID card issue.   **Ticket ID:** `TKT-261031-04B4`   **Category:** IT Support   **SLA:** 2 working days    The IT team will look into why your card stopped working at the library gate and get back to you soon.

**T12 — Please raise a high priority ticket: my fee payment of INR 50,000 was debited but not showing on the portal.**

> I'm sorry, but our ticketing service is currently experiencing a temporary outage (HTTP 503), so I was unable to automatically generate your support ticket right now.  Since this is a high-priority payment issue, please reach out directly to the Finance and Accounts office with your transaction reference number, bank statement showing the debit of INR 50,000, and your Student ID (S1002) so they can reconcile it manually.   Would you like me to try creating the ticket again now, or can I help you with anything else in the meantime?

**T13 — Can you tell me the fee status of student S1002?**

> You can only access your own records. I am unable to view fee details or information for student S1002.   If you would like to check your own fee status, please let me know!
