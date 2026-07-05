import requests
from pathlib import Path

ACTS = {
    "IT_Act_2000": "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
    "IT_Rules_2021": "https://www.meity.gov.in/static/uploads/2024/02/Information-Technology-Intermediary-Guidelines-and-Digital-Media-Ethics-Code-Rules-2021-updated-06.04.2023-.pdf",
    "BNS_2023": "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf",
    "BNSS_2023": "https://www.indiacode.nic.in/bitstream/123456789/21544/1/the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf",
    "Consumer_Protection_Act_2019": "https://www.indiacode.nic.in/bitstream/123456789/16939/1/a2019-35.pdf",
}

def download_acts(output_dir: str = "data/raw_pdfs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
        "Referer": "https://www.indiacode.nic.in/",
    })

    for name, url in ACTS.items():
        out_path = Path(output_dir) / f"{name}.pdf"
        if out_path.exists():
            print(f"[skip] {name} already downloaded ({out_path.stat().st_size // 1024}KB)")
            continue
        print(f"[downloading] {name}...")
        try:
            r = session.get(url, timeout=60, allow_redirects=True)
            content_type = r.headers.get("content-type", "")
            if r.status_code == 200 and ("pdf" in content_type or r.content[:4] == b'%PDF'):
                out_path.write_bytes(r.content)
                print(f"[done] {name} -> {out_path} ({len(r.content)//1024}KB)")
            else:
                print(f"[failed] {name}: HTTP {r.status_code} | content-type: {content_type}")
                print(f"         Final URL: {r.url}")
        except Exception as e:
            print(f"[error] {name}: {e}")

if __name__ == "__main__":
    download_acts()