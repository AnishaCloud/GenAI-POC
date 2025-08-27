import fitz  # PyMuPDF
import numpy as np
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from openai import OpenAI
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
ASTRA_BUNDLE_PATH = "secure-connect-db.zip"  # Your Astra secure bundle
KEYSPACE = "your_keyspace"
TABLE = "fg_items_pdf"
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold

# -------------------------------
# Connect to Astra DB
# -------------------------------
cloud_config = {'secure_connect_bundle': ASTRA_BUNDLE_PATH}
auth_provider = PlainTextAuthProvider(**cloud_config)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect(KEYSPACE)

# -------------------------------
# Create table if not exists
# -------------------------------
session.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    fgitem TEXT PRIMARY KEY,
    batch_number TEXT,
    quantity INT,
    site TEXT,
    modified_date TIMESTAMP,
    pdf_content BLOB,
    embedding VECTOR<float, 1536>
)
""")
print(f"Table '{TABLE}' is ready ✅")

# -------------------------------
# OpenAI Client
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Helper Functions
# -------------------------------
def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Upsert with Vector Search
# -------------------------------
def upsert_or_search_pdf(fgitem, batch_number, quantity, site, pdf_path):
    text = pdf_to_text(pdf_path)
    new_emb = get_embedding(text)
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Step 1: Check by FG item
    row = session.execute(
        f"SELECT pdf_content, embedding FROM {TABLE} WHERE fgitem = %s",
        (fgitem,)
    ).one()
    
    if row:
        existing_emb = row.embedding
        similarity = cosine_similarity(existing_emb, new_emb)
        if similarity > SIMILARITY_THRESHOLD:
            print(f"FG item '{fgitem}' found and PDF matches ✅")
            return fgitem
        else:
            # Update PDF and embedding
            stmt = session.prepare(f"""
                UPDATE {TABLE}
                SET batch_number=?, quantity=?, site=?, modified_date=?, pdf_content=?, embedding=?
                WHERE fgitem=?
            """)
            session.execute(stmt, (batch_number, quantity, site, datetime.utcnow(), pdf_bytes, new_emb, fgitem))
            print(f"FG item '{fgitem}' exists but PDF updated 🔄")
            return fgitem
    else:
        # Step 2: Search by vector similarity across all FG items
        rows = session.execute(f"SELECT fgitem, embedding FROM {TABLE}")
        best_match = None
        best_sim = 0
        for r in rows:
            sim = cosine_similarity(new_emb, r.embedding)
            if sim > best_sim:
                best_sim = sim
                best_match = r.fgitem
        
        if best_sim >= SIMILARITY_THRESHOLD:
            print(f"No exact FG item match, but PDF similar to FG item '{best_match}' (sim={best_sim:.2f})")
            return best_match
        else:
            # Step 3: Insert as new FG item
            stmt = session.prepare(f"""
                INSERT INTO {TABLE} (fgitem, batch_number, quantity, site, modified_date, pdf_content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """)
            session.execute(stmt, (fgitem, batch_number, quantity, site, datetime.utcnow(), pdf_bytes, new_emb))
            print(f"FG item '{fgitem}' not found. New PDF inserted ➕")
            return fgitem

# -------------------------------
# Add Sample Data
# -------------------------------
sample_data = [
    ("FG001", "B001", 100, "SiteA", "sample1.pdf"),
    ("FG002", "B002", 50, "SiteB", "sample2.pdf"),
    ("FG003", "B003", 200, "SiteC", "sample3.pdf")
]

for fgitem, batch, qty, site, pdf_file in sample_data:
    upsert_or_search_pdf(fgitem, batch, qty, site, pdf_file)
