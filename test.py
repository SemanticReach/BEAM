import json, requests, tempfile
from pathlib import Path

SERVER_URL = "http://18.220.128.24:8000"
API_KEY    = "hb_h97e95G1aOWJ5mNs5sdzs5iR1OJQ7nfpa_3aXtUx7YE"

# Load and flatten turns
with open("test_chats/100K/1/chat.json", encoding="utf-8") as f:
    data = json.load(f)

turns = []
if isinstance(data, list):
    for batch in data:
        for pair in batch.get("turns", []):
            if isinstance(pair, list):
                turns.extend(pair)
            elif isinstance(pair, dict):
                turns.append(pair)

print(f"Total turns: {len(turns)}")

# Format as text
lines = []
for turn in turns[:50]:   # just first 50 turns to test
    role    = turn.get("role", "unknown").upper()
    content = turn.get("content", "").strip()
    tid     = turn.get("id", "?")
    time_anchor = turn.get("time_anchor")
    if not content:
        continue
    prefix = f"[TURN {tid}]"
    if time_anchor:
        prefix += f" [TIME: {time_anchor}]"
    lines.append(f"{prefix} {role}: {content}")

doc_text = "\n\n".join(lines)
print(f"Text length: {len(doc_text)} chars")

# Upload
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
    tmp.write(doc_text)
    tmp_path = tmp.name

with open(tmp_path, "rb") as f:
    resp = requests.post(
        f"{SERVER_URL}/upload_document/",
        headers={"X-API-Key": API_KEY},
        files={"file": ("chat_test.txt", f)},
        data={"dim": 512, "seed": 42, "depth": 3},
        timeout=120,
    )

print(resp.status_code)
print(resp.json())
Path(tmp_path).unlink()