"""보고서 PNG용 공통: 한글 폰트, 출력 폴더."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

REPORT_FIGURES_OUT = Path("report_figures")


def setup_korean_matplotlib_font() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunsl.ttf"),
    ]
    for font_path in candidates:
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
        except Exception:
            pass
        fp = font_manager.FontProperties(fname=str(font_path))
        name = fp.get_name()
        plt.rcParams["font.family"] = name
        plt.rcParams["font.sans-serif"] = [name, "Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
