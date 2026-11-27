"""Fetch open primary sources for the staged reader and teacher design."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from subprocess import run

ROOT = Path(__file__).resolve().parent
PAPERS = {
    "1999_Ng_Potential_Based_Shaping": "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf",
    "2015_Chang_LOLS": "https://proceedings.mlr.press/v37/changb15.pdf",
    "2017_Guo_Confidence_Calibration": "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf",
    "2018_Jayaraman_Learning_to_Look_Around": "https://openaccess.thecvf.com/content_cvpr_2018/papers/Jayaraman_Learning_to_Look_CVPR_2018_paper.pdf",
    "2021_Chaplot_SEAL": "https://proceedings.neurips.cc/paper_files/paper/2021/file/6d0c932802f6953f70eb20931645fa40-Paper.pdf",
    "2022_Yan_SAM": "https://arxiv.org/pdf/2012.02383",
    "2022_Baisero_Unbiased_Asymmetric_RL": "https://arxiv.org/pdf/2105.11674",
    "2024_Scarpellini_Look_Around_and_Learn": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07259.pdf",
    "2024_Raina_Coaching_a_Robotic_Sonographer": "https://arxiv.org/pdf/2409.02337",
}


def fetch(item):
    name, url = item
    file = ROOT / "papers" / f"{name}.pdf"
    if not file.is_file():
        pending = file.with_suffix(".part")
        result = run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "1",
                      "--connect-timeout", "12", "--max-time", "55", "--output", str(pending), url],
                     capture_output=True, text=True)
        if result.returncode:
            pending.unlink(missing_ok=True)
            return {"status": "failed", "source": url, "error": result.stderr.strip()}
        pending.replace(file)
    metadata = run(["pdfinfo", str(file)], check=True, capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", metadata, re.M)[1])
    return {"status": "verified_pdf", "source": url, "file": str(file.relative_to(ROOT)),
            "pages": pages, "bytes": file.stat().st_size, "sha256": sha256(file.read_bytes()).hexdigest()}


def main():
    (ROOT / "papers").mkdir(exist_ok=True)
    with ThreadPoolExecutor(6) as executor:
        entries = dict(zip(PAPERS, executor.map(fetch, PAPERS.items()), strict=True))
    (ROOT / "papers.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    for name, result in entries.items():
        print(name, result["status"], result.get("pages", result.get("error")))


if __name__ == "__main__":
    main()
