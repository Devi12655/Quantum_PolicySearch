from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import numpy as np
import io
import os
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import openpyxl

# ------------------------------------------------------
# App setup
# ------------------------------------------------------
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Static mount
static_path = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# ------------------------------------------------------
# Load quantum model
# ------------------------------------------------------
_model_file = None
for p in [
    os.path.join(BASE_DIR, "quantum_model.pkl"),
    "/mnt/data/quantum_model.pkl"
]:
    if os.path.exists(p):
        _model_file = p
        break

if not _model_file:
    print("⚠️ Warning: quantum_model.pkl not found.")
    model = None
else:
    model = joblib.load(_model_file)
    vectorizer = model["vectorizer"]
    scaler = model["scaler"]
    pca = model["pca"]
    embeddings = model["embeddings"]
    df = model["df"]

# ------------------------------------------------------
# Embedding generation
# ------------------------------------------------------
def make_query_embedding(query: str):
    if model is None:
        return None
    qv = vectorizer.transform([query]).toarray()
    qs = scaler.transform(qv)
    qp = pca.transform(qs)
    qe = np.tanh(qp * np.pi)
    qq = np.concatenate([qe, np.sin(qe)], axis=1)
    return qq

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), k: int = Form(6)):
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": "Model missing. Place quantum_model.pkl in project root."
            },
        )

    q_emb = make_query_embedding(query)
    sims = cosine_similarity(q_emb, embeddings).ravel()

    df2 = df.copy()
    df2["score"] = sims
    results = df2.sort_values("score", ascending=False).head(int(k)).fillna("N/A")

    # Chart data
    scores_data = {
        str(i + 1): float(round(s, 4))
        for i, s in enumerate(results["score"].tolist())
    }

    region_data = {}
    year_data = {}

    for _, r in results.iterrows():
        region = r.get("region") or "Unknown"
        region_data[region] = region_data.get(region, 0) + 1

        year = str(r.get("year") or "Unknown")
        year_data[year] = year_data.get(year, 0) + 1

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": query,
            "results": results.to_dict(orient="records"),
            "scores_data": scores_data,
            "region_data": region_data,
            "year_data": year_data,
        },
    )

# ------------------------------------------------------
# File Generators
# ------------------------------------------------------
def make_pdf_bytes(title, region, year, status, summary):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 24

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Region: {region}    Year: {year}    Status: {status}")
    y -= 20

    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(summary, "Helvetica", 11, width - 2 * margin)

    for line in lines:
        if y < margin + 20:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)

        c.drawString(margin, y, line)
        y -= 14

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ------------------------------------------------------
# Download: PDF
# ------------------------------------------------------
@app.get("/download/pdf/{idx}")
def download_pdf(idx: int):
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    try:
        r = df.iloc[idx]
    except Exception:
        raise HTTPException(status_code=404, detail="Index out of bounds")

    title = str(r.get("title", "Policy")).replace("/", "_")[:120]

    buf = make_pdf_bytes(
        title,
        r.get("region", ""),
        r.get("year", ""),
        r.get("status", ""),
        r.get("summary", ""),
    )

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{title}.pdf"'},
    )

# ------------------------------------------------------
# Download: TXT
# ------------------------------------------------------
@app.get("/download/txt/{idx}")
def download_txt(idx: int):
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    try:
        r = df.iloc[idx]
    except Exception:
        raise HTTPException(status_code=404, detail="Index out of bounds")

    content = (
        f"Title: {r.get('title','')}\n"
        f"Region: {r.get('region','')}\n"
        f"Year: {r.get('year','')}\n"
        f"Status: {r.get('status','')}\n\n"
        f"{r.get('summary','')}\n"
    )

    filename = str(r.get("title", "policy")).replace("/", "_")[:80]

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}.txt"'},
    )

# ------------------------------------------------------
# Download: XLSX
# ------------------------------------------------------
@app.get("/download/xlsx/{idx}")
def download_xlsx(idx: int):
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    try:
        r = df.iloc[idx]
    except Exception:
        raise HTTPException(status_code=404, detail="Index out of bounds")

    wb = openpyxl.Workbook()
    ws = wb.active

    ws.append(["Title", "Region", "Year", "Status", "Summary"])
    ws.append([
        r.get("title", ""),
        r.get("region", ""),
        r.get("year", ""),
        r.get("status", ""),
        r.get("summary", "")
    ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = str(r.get("title", "policy")).replace("/", "_")[:80]

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}.xlsx"'},
    )
