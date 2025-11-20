from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime
from io import BytesIO
import csv
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from model import classify_tone_rich  # <-- Import upgraded model

app = Flask(__name__)

# In-memory history
history = []

# Ensure export folder exists
os.makedirs("exports", exist_ok=True)


# INDEX (MAIN PAGE)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("email_text", "").strip()

        if text:
            result = classify_tone_rich(text)  # <-- uses new upgraded threat model

            # Save to history
            history.append({
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text,
                **result
            })

    return render_template(
        "index.html",
        title="Analyze Email",
        email_text=text,
        result=result
    )


# HISTORY PAGE

@app.route("/history")
def history_view():
    q = request.args.get("q", "").lower()
    filter_label = request.args.get("label", "").lower()

    filtered = history[:]

    # Apply search
    if q:
        filtered = [h for h in filtered if q in h["text"].lower()]

    # Apply label filter (aggressive / neutral / polite / friendly)
    if filter_label:
        filtered = [h for h in filtered if h["label"].lower() == filter_label]

    # Newest first
    filtered = list(reversed(filtered))

    return render_template(
        "history.html",
        title="History",
        history=filtered,
        search=q,
        active_filter=filter_label
    )



# EXPORT CSV

@app.route("/export/csv")
def export_csv():
    filepath = os.path.join("exports", "history.csv")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Time UTC", "Label", "Confidence", "Severity",
            "ThreatScore", "PolitenessScore", "FriendlyScore",
            "HasThreat", "HasProfanity", "HasSarcasm",
            "Text"
        ])

        for e in history:
            writer.writerow([
                e["created_at"],
                e["label"],
                f"{e['confidence']:.1f}",
                e["severity"],
                e["threat_score"],
                e["politeness_score"],
                e["friendly_score"],
                int(e["has_threat"]),
                int(e["has_profanity"]),
                int(e["has_sarcasm"]),
                e["text"]
            ])

    return send_file(filepath, as_attachment=True)



 # EXPORT PDF

@app.route("/export/pdf")
def export_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header bar
    c.setFillColorRGB(0.12, 0.15, 0.20)
    c.rect(0, height - 60, width, 60, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 35, "Tone Classifier – History Report")

    y = height - 80

    for h in reversed(history):
        if y < 90:
            c.showPage()
            y = height - 60

        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(40, y, f"{h['created_at']} | {h['label']} | Severity {h['severity']}")
        y -= 12

        meta = f"Threat:{h['threat_score']}  Polite:{h['politeness_score']}  Friendly:{h['friendly_score']}"
        c.setFont("Helvetica", 9)
        c.drawString(40, y, meta)
        y -= 12

        text = h["text"]
        while len(text) > 90:
            idx = text.rfind(" ", 0, 90)
            if idx == -1:
                idx = 90
            c.drawString(50, y, text[:idx])
            text = text[idx:].strip()
            y -= 11

        c.drawString(50, y, text)
        y -= 20

    c.showPage()
    c.save()

    buffer.seek(0)
    filepath = os.path.join("exports", "history.pdf")
    with open(filepath, "wb") as f:
        f.write(buffer.getvalue())

    return send_file(filepath, as_attachment=True)



# CLEAR HISTORY

@app.route("/history/clear", methods=["POST"])
def clear_history():
    history.clear()
    return redirect(url_for("history_view"))



# RUN APP

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)











