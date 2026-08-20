import json
import urllib.request

BOARD = "https://aidc.nadir.sh/model"

TEAM = "11"
BY = "Amal Alharbi"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
IMAGE = "ghcr.io/layankhalid0/aidc-team11-warmup:latest"


def request(url, body):
    data = json.dumps(body).encode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as r:
        print(r.status)
        print(r.read().decode())


with urllib.request.urlopen(
    urllib.request.Request(
        "http://localhost:8000/generate",
        headers={"User-Agent": "Mozilla/5.0"},
    )
) as r:
    result = json.loads(r.read().decode())


body = {
    "team": TEAM,
    "by": BY,
    "model": MODEL,
    "image": IMAGE,
    "tokens_per_sec": result["tokens_per_sec"],
    "sample": result["sample"].replace("\n", " ").strip(),
}

request(BOARD, body)
