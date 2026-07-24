from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "RESIDENTIAL TENANCY AGREEMENT", 0, 1, "C")
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

content = """
Date: 25 July 2026
Landlord: John Doe Properties Pty Ltd
Tenant: Jane Smith
Property Address: 123 Fake Street, Melbourne VIC 3000

1. BASIC TERMS
Rent: $650 per week, payable fortnightly in advance.
Bond: $2600, to be lodged with the RTBA.
Lease Term: 12 months, commencing on 01 August 2026 and ending on 31 July 2027.

2. RENT INCREASES
The Landlord reserves the right to increase the rent by 10% every 6 months without prior written notice. The tenant must comply with this automatic increase.

3. CLEANING AT END OF TENANCY
The Tenant agrees to have all carpets, curtains, and windows professionally cleaned by a cleaning company nominated by the Landlord upon vacating the premises, regardless of the condition of the property and regardless of whether any pets were kept. A receipt must be provided.

4. MAINTENANCE AND REPAIRS
The Tenant is responsible for all maintenance and repairs up to the value of $200. Any plumbing or electrical faults costing less than $200 must be paid out-of-pocket by the Tenant.

5. INSPECTIONS
The Landlord or their Agent may enter the property for a routine inspection at any time between 9 AM and 5 PM on any day, provided they give 2 hours verbal notice by telephone.

6. GUESTS
The Tenant must not allow any guest to stay at the property for more than 2 consecutive nights without written permission from the Landlord. If a guest stays longer, an additional fee of $50 per night will be added to the rent.

SIGNATURES:
Landlord: ___________________________
Tenant: ___________________________
"""

# Handling Unicode issues by just using standard ascii text
pdf.multi_cell(0, 10, content)
pdf.output("test_lease_agreement.pdf")
print("PDF generated successfully.")
