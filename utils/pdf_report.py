from fpdf import FPDF

def create_pdf(data, result, prob):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Diabetes Medical Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)

    for key, value in data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    pdf.cell(200, 10, f"Prediction Result: {result}", ln=True)
    pdf.cell(200, 10, f"Diabetes Probability: {prob:.2f}%", ln=True)

    file_name = "report.pdf"
    pdf.output(file_name)

    return file_name