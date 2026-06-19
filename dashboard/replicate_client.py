import os, time, requests

_MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"

def generate_image(prompt, *, token=None, aspect_ratio="1:1", timeout=120):
    token = token or os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait"}
    body = {"input": {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png"}}
    r = requests.post(_MODEL_URL, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    pred = r.json()
    get_url = (pred.get("urls") or {}).get("get")
    deadline = time.time() + timeout
    while pred.get("status") not in ("succeeded", "failed", "canceled"):
        if time.time() > deadline:
            raise TimeoutError("replicate prediction timed out")
        time.sleep(2)
        pred = requests.get(get_url, headers=headers, timeout=30).json()
    if pred.get("status") != "succeeded":
        raise RuntimeError(f"replicate prediction {pred.get('status')}")
    out = pred.get("output")
    url = out[0] if isinstance(out, list) else out
    if not url:
        raise RuntimeError("replicate returned no output")
    img = requests.get(url, timeout=60)
    img.raise_for_status()
    return img.content
