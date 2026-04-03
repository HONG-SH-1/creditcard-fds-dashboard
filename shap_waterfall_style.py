"""SHAP waterfall matplotlib fixes shared by Streamlit app and report notebooks."""

from __future__ import annotations

from contextlib import contextmanager
import re

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.text as mtext
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


def shap_display_feature_names(columns) -> np.ndarray:
    """워터폴 축 라벨용(모델 입력 컬럼명은 변경하지 않음)."""
    out: list[str] = []
    fixed = {
        "Time_sin": "시간 주기(sin)",
        "Time_cos": "시간 주기(cos)",
        "Amount": "거래 금액(RobustScaler)",
    }
    for c in columns:
        if c in fixed:
            out.append(fixed[c])
        elif len(c) >= 2 and c[0] == "V" and c[1:].isdigit():
            out.append(f"{c}(비식별·PCA)")
        else:
            out.append(c)
    return np.array(out, dtype=object)


def configure_matplotlib_for_shap() -> None:
    """SHAP 워터폴: 한글 폰트 + ASCII 마이너스 + '$'·'$$'를 수식이 아니라 글자로만 다루지 않게 방지.

    Matplotlib 3.10+ 기본값 text.parse_math=True 이면 SHAP가 넣는 `$E[f(X)]$` 등이
    mathtext로 파싱되다 꼬여 화면에 `$$` 잔상이 남을 수 있음 → 전역으로 끔.
    """
    mpl.rcParams["axes.unicode_minus"] = False
    try:
        mpl.rcParams["text.parse_math"] = False
    except (KeyError, ValueError):
        pass
    names = {f.name for f in fm.fontManager.ttflist}
    for fam in (
        "Malgun Gothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Noto Sans CJK KR",
    ):
        if fam in names:
            mpl.rcParams["font.family"] = fam
            return
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Malgun Gothic",
        "NanumGothic",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]


_SHAP_MINUS_DISPLAY = "-"


def _brutal_shap_label(s: str) -> str:
    """SHAP 워터폴 문자열 최종 정리: $ 제거·ㅁ□·U+2212 등 → ASCII '-' 음수·E[f(X)] 접두."""
    if not s:
        return s
    t = re.sub(r"\$+", "", s).replace("＄", "")
    t = t.replace("\\,", " ").replace("{", "").replace("}", "")
    for ch in (
        "\u2212",
        "\u3161",
        "\u2013",
        "\u2014",
        "\u2012",
        "\u2015",
        "\uFE63",
        "\uFF0D",
        "\u207B",
        "\u208B",
    ):
        t = t.replace(ch, _SHAP_MINUS_DISPLAY)
    for bad in ("\u3141", "\u25a1", "\u25a0", "\ufffd", "\u2751"):
        t = t.replace(bad, _SHAP_MINUS_DISPLAY)
    t = re.sub(r"^-(?=[\d.])", _SHAP_MINUS_DISPLAY, t)
    t = re.sub(r"(?<=[\s=])-(?=[\d.])", _SHAP_MINUS_DISPLAY, t)
    lines = [" ".join(line.split()) for line in t.splitlines()] or [""]
    adj: list[str] = []
    for line in lines:
        if not line:
            continue
        if "E[f(X)]" in line and "기준(기대값)" not in line:
            line = "기준(기대값) " + line
        line = re.sub(rf"{re.escape(_SHAP_MINUS_DISPLAY)}{{2,}}", _SHAP_MINUS_DISPLAY, line)
        adj.append(line)
    return "\n".join(adj)


def _shap_minus_for_display(s: str) -> str:
    return _brutal_shap_label(s)


def _strip_dollar_marks(s: str) -> str:
    return re.sub(r"\$+", "", s).replace("＄", "")


def _plain_text_from_mathtext(s: str) -> str:
    return _brutal_shap_label(s.replace("\\,", " "))


def _stack_expectation_value_lines(merged: str) -> str:
    if " = " not in merged:
        return merged
    left, right = merged.split(" = ", 1)
    left_s, right_s = left.strip(), right.strip()
    if not right_s:
        return merged
    if "E[f(X)]" not in left_s and "f(x)" not in left_s.lower():
        return merged
    return f"{left_s}\n= {right_s}"


def _shorten_shap_expectation_display(merged: str) -> str:
    lines = merged.split("\n")
    if not lines:
        return merged
    head = lines[0]
    if "E[f(X)]" in head or "기준(기대값)" in head or "기대값" in head:
        lines[0] = "기대값"
    elif "f(x)" in head.lower():
        lines[0] = "이번 예측"
    return "\n".join(lines)


def bump_shap_main_axis_x_tickpad(fig) -> None:
    try:
        main = fig.axes[0]
    except IndexError:
        return
    main.tick_params(axis="x", which="major", pad=8)


def clear_shap_twiny_x_tick_labels(fig) -> None:
    if len(fig.axes) < 2:
        return
    for ax in fig.axes[1:]:
        for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
            for lab in (tick.label1, tick.label2):
                lab.set_text("")
                lab.set_visible(False)
                try:
                    lab.set_parse_math(False)
                except (AttributeError, ValueError):
                    pass


def set_shap_waterfall_footer_xlabel(fig, base_val: float, fx_val: float) -> None:
    try:
        main = fig.axes[0]
    except IndexError:
        return
    txt = f"기준(기대값) E[f(X)] = {base_val:.3f}\n이번 예측 f(x) = {fx_val:.3f}"
    main.set_xlabel(txt, fontsize=10, labelpad=12)
    try:
        main.xaxis.label.set_parse_math(False)
    except (AttributeError, ValueError):
        pass


@contextmanager
def shap_ascii_minus_format_value():
    import shap.utils._general as shap_general
    import shap.plots._waterfall as shap_waterfall

    orig = shap_general.format_value

    def _patched(s, format_str):
        return _brutal_shap_label(orig(s, format_str))

    prev_wf = shap_waterfall.format_value
    shap_general.format_value = _patched
    shap_waterfall.format_value = _patched
    try:
        yield
    finally:
        shap_general.format_value = orig
        shap_waterfall.format_value = prev_wf


def _sanitize_tick_label_sequence(labels):
    if labels is None:
        return None
    if isinstance(labels, str):
        return _brutal_shap_label(labels)
    try:
        return [_brutal_shap_label(x) if isinstance(x, str) else x for x in labels]
    except TypeError:
        return labels


@contextmanager
def shap_sanitize_mpl_ticklabels():
    orig_x = Axes.set_xticklabels
    orig_y = Axes.set_yticklabels

    def set_xticklabels(self, labels, *args, **kwargs):
        return orig_x(self, _sanitize_tick_label_sequence(labels), *args, **kwargs)

    def set_yticklabels(self, labels, *args, **kwargs):
        return orig_y(self, _sanitize_tick_label_sequence(labels), *args, **kwargs)

    Axes.set_xticklabels = set_xticklabels  # type: ignore[method-assign]
    Axes.set_yticklabels = set_yticklabels  # type: ignore[method-assign]
    try:
        yield
    finally:
        Axes.set_xticklabels = orig_x  # type: ignore[method-assign]
        Axes.set_yticklabels = orig_y  # type: ignore[method-assign]


@contextmanager
def shap_waterfall_render_patches():
    with shap_ascii_minus_format_value(), shap_sanitize_mpl_ticklabels():
        yield


def merge_shap_twiny_duplicate_labels(fig) -> None:
    try:
        fig.canvas.draw()
    except Exception:
        pass

    def _tick_xtext(tick) -> str:
        return (tick.label1.get_text() or tick.label2.get_text() or "").strip()

    for ax in fig.get_axes():
        ticks = [t for t in ax.xaxis.get_major_ticks() if _tick_xtext(t)]
        if len(ticks) == 1:
            s0 = ticks[0].label1.get_text() or ticks[0].label2.get_text()
            if "$" in s0 or "＄" in s0 or "E[f(X)]" in s0 or "f(x)" in s0.lower():
                merged = _shorten_shap_expectation_display(
                    _stack_expectation_value_lines(_brutal_shap_label(s0))
                )
                ticks[0].label1.set_text(merged)
                ticks[0].label2.set_text(merged)
                for lab in (ticks[0].label1, ticks[0].label2):
                    try:
                        lab.set_parse_math(False)
                    except (AttributeError, ValueError):
                        pass
            continue
        if len(ticks) < 2:
            continue
        parts = [(t.label1.get_text() or t.label2.get_text()) for t in ticks]
        combo = " ".join(_strip_dollar_marks(p) for p in parts)
        if "E[f(X)]" not in combo and "f(x)" not in combo.lower():
            continue
        merged = _shorten_shap_expectation_display(
            _stack_expectation_value_lines(_plain_text_from_mathtext(" ".join(parts)))
        )
        ticks[0].label1.set_text(merged)
        ticks[0].label2.set_text(merged)
        for lab in (ticks[0].label1, ticks[0].label2):
            try:
                lab.set_parse_math(False)
            except (AttributeError, ValueError):
                pass
        for t in ticks[1:]:
            for lab in (t.label1, t.label2):
                lab.set_text("")
                lab.set_visible(False)
                try:
                    lab.set_parse_math(False)
                except (AttributeError, ValueError):
                    pass


def merge_shap_split_value_labels(fig) -> None:
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    for ax in fig.axes:
        cand = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        for t_left in list(cand):
            if not t_left.get_visible():
                continue
            s_l = t_left.get_text()
            bb_l = t_left.get_window_extent(renderer)
            cy_l = bb_l.y0 + bb_l.height / 2
            best: mtext.Text | None = None
            best_dx = 1e9
            for t_right in cand:
                if t_right is t_left or not t_right.get_visible():
                    continue
                sr0 = t_right.get_text().strip()
                sr = _strip_dollar_marks(sr0).strip()
                if not (sr.startswith("=") or sr.startswith("= ")):
                    continue
                bb_r = t_right.get_window_extent(renderer)
                cy_r = bb_r.y0 + bb_r.height / 2
                if abs(cy_l - cy_r) > 14:
                    continue
                dx = bb_r.x0 - (bb_l.x0 + bb_l.width)
                if dx < -22:
                    continue
                if dx < best_dx:
                    best_dx = dx
                    best = t_right
            if best is not None:
                sr_final = best.get_text()
                glue = "" if s_l.endswith((" ", "$")) else " "
                merged = _plain_text_from_mathtext(s_l + glue + sr_final)
                t_left.set_text(merged)
                try:
                    t_left.set_parse_math(False)
                except (AttributeError, ValueError):
                    pass
                best.set_visible(False)


def sanitize_shap_figure_mathtext(fig) -> None:
    for artist in fig.findobj(lambda o: isinstance(o, mtext.Text)):
        s = artist.get_text()
        if not s:
            continue
        had_math_marker = "$" in s or "\\" in s or "＄" in s
        if had_math_marker:
            artist.set_text(_plain_text_from_mathtext(s))
            try:
                artist.set_parse_math(False)
            except (AttributeError, ValueError):
                pass
        else:
            artist.set_text(_shap_minus_for_display(_strip_dollar_marks(s)))


def polish_shap_figure_after_layout(fig) -> None:
    try:
        fig.canvas.draw()
    except Exception:
        pass
    for ax in fig.axes:
        _scrub_axis_ticks_inplace(ax)
        for t in ax.texts:
            if not t.get_visible():
                continue
            s = t.get_text()
            if not s:
                continue
            if "$" in s or "＄" in s or "\\" in s:
                s2 = _plain_text_from_mathtext(s)
            else:
                s2 = _shap_minus_for_display(_strip_dollar_marks(s))
            t.set_text(s2)
            try:
                t.set_parse_math(False)
            except (AttributeError, ValueError):
                pass
    merge_shap_twiny_duplicate_labels(fig)
    scrub_every_text_in_figure(fig)


def _scrub_axis_ticks_inplace(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        for minor in (False, True):
            ticks = axis.get_minor_ticks() if minor else axis.get_major_ticks()
            for tick in ticks:
                for lab in (tick.label1, tick.label2):
                    if lab is None:
                        continue
                    s = lab.get_text()
                    if not s:
                        continue
                    lab.set_text(_brutal_shap_label(s))
                    try:
                        lab.set_parse_math(False)
                    except (AttributeError, ValueError):
                        pass


def scrub_every_text_in_figure(fig) -> None:
    for ax in fig.get_axes():
        _scrub_axis_ticks_inplace(ax)
    for o in fig.findobj(lambda x: isinstance(x, mtext.Text)):
        s = o.get_text()
        if not s:
            try:
                o.set_parse_math(False)
            except (AttributeError, ValueError):
                pass
            continue
        o.set_text(_brutal_shap_label(s))
        try:
            o.set_parse_math(False)
        except (AttributeError, ValueError):
            pass
    for ax in fig.get_axes():
        _scrub_axis_ticks_inplace(ax)


def shap_waterfall_fig_height_inches(n_values: int, max_display: int = 10) -> float:
    n_val = int(n_values)
    max_disp = int(max_display)
    n_rows = min(max_disp, n_val) + (1 if n_val > max_disp else 0)
    fig_h = 2.35 + n_rows * 0.62
    return float(max(6.2, min(fig_h, 12.0)))


def finalize_shap_waterfall_figure(
    fig,
    explanation_row,
    fraud_prob: float | None = None,
    *,
    width_inches: float = 11.0,
    max_display: int = 10,
) -> None:
    """Post-process current figure after ``shap.plots.waterfall(..., show=False)`` (same as app)."""
    merge_shap_twiny_duplicate_labels(fig)
    merge_shap_split_value_labels(fig)
    sanitize_shap_figure_mathtext(fig)
    merge_shap_split_value_labels(fig)

    n_val = len(np.asarray(explanation_row.values).ravel())
    fig_h = shap_waterfall_fig_height_inches(n_val, max_display=max_display)
    fig.set_size_inches(width_inches, float(fig_h))

    bump_shap_main_axis_x_tickpad(fig)
    plt.tight_layout(rect=(0.02, 0.26, 0.98, 0.9))
    polish_shap_figure_after_layout(fig)
    try:
        fig.canvas.draw()
    except Exception:
        pass
    merge_shap_twiny_duplicate_labels(fig)
    scrub_every_text_in_figure(fig)
    clear_shap_twiny_x_tick_labels(fig)

    bv = float(np.asarray(explanation_row.base_values).reshape(-1)[0])
    if fraud_prob is None:
        fx = float(bv + np.asarray(explanation_row.values).sum())
    else:
        fx = float(fraud_prob)

    set_shap_waterfall_footer_xlabel(fig, bv, fx)
    plt.tight_layout(rect=(0.02, 0.26, 0.98, 0.9))
    try:
        fig.canvas.draw()
    except Exception:
        pass
