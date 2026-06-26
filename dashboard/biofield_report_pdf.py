"""Render a presentation HTML doc to a real PDF via headless Chromium (Playwright,
already installed in the local app's python3). Used for the printable/shippable
Biofield report. Local only; no network."""


def report_pdf_bytes(html: str) -> bytes:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            return page.pdf(
                format="Letter",
                print_background=True,
                margin={"top": "0.6in", "bottom": "0.6in", "left": "0.6in", "right": "0.6in"},
            )
        finally:
            browser.close()


def save_report_pdf(html: str, out_path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(report_pdf_bytes(html))
    return out_path
