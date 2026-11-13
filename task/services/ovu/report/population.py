"""Population statistics as a Word document, built from report.json only.

    python -m ovu.report.population [--report outputs/ovu/report.json] [--out outputs/ovu/人群统计分析.docx]

report.json is written by ``python -m ovu.pipeline`` and holds aggregates only, so this step
never opens patient rows and can run on any machine. Run it again after the pipeline to refresh
every number, table and figure. Needs python-docx and matplotlib (not needed by anything else).

The document reports each field on its own (distributions, missing values); it does not relate
fields to each other or to the oocyte count.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6b68", "#e4e4e1"
LATIN_FONT, CJK_FONT = "Times New Roman", "宋体"
CJK_CANDIDATES = ("Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Source Han Sans CN", "Microsoft YaHei",
                  "SimHei", "PingFang SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Serif CJK SC", "SimSun")
AGE_BANDS = ((None, 29, "<30"), (30, 34, "30–34"), (35, 37, "35–37"), (38, 40, "38–40"), (41, None, "≥41"))
# Rows of the baseline table and of the medication table: field -> name shown (unit left out: not in the data)
BASELINE = (("bmi", "体重指数（BMI）"), ("height_cm", "身高（cm）"), ("weight_kg", "体重（kg）"), ("bp_systolic", "收缩压"),
            ("bp_diastolic", "舒张压"), ("pulse", "脉搏"), ("infertility_years", "不孕年限（年）"), ("fsh", "FSH"), ("lh", "LH"),
            ("prl", "PRL"), ("e2", "E2"), ("testosterone", "T"), ("amh", "AMH"), ("afc_downreg", "降调日窦卵泡数"))
SHOWN_ELSEWHERE = ("age",)                                          # has its own section
MEDICATION = (("gn_start_dose", "启动 Gn 用量"), ("gn_start_day", "启动 Gn 时机（周期第几天）"), ("gn_days", "Gn 总天数"),
              ("gn_total_dose", "Gn 总用量"), ("gnrha_start_dose", "GnRHa 开始量"), ("gnrha_days", "GnRHa 总天数"),
              ("gnrha_total_dose", "GnRHa 总用量"))


# --- figures ------------------------------------------------------------------------------------
def chart_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in CJK_CANDIDATES if name in available), None)


def _axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.set_axisbelow(True)
    return fig, ax


def _png(fig) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def histogram_figure(histogram: dict[str, Any], xlabel: str, ylabel: str) -> io.BytesIO:
    edges, counts = histogram["edges"], histogram["counts"]
    fig, ax = _axes(6.2, 2.7)
    widths = [b - a for a, b in zip(edges, edges[1:])]
    ax.bar(edges[:-1], counts, width=[w * 0.86 for w in widths], align="edge", color=BLUE, linewidth=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_xlim(edges[0], edges[-1])
    return _png(fig)


def bar_figure(labels: list[str], values: list[float], xlabel: str, *, percent: bool = True) -> io.BytesIO:
    """Horizontal bars, largest first as given, each with its value at the end."""
    fig, ax = _axes(6.2, 0.34 * len(labels) + 0.8)
    positions = range(len(labels))[::-1]
    ax.barh(list(positions), values, height=0.62, color=BLUE, linewidth=0)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, color=INK, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.spines["bottom"].set_visible(False)
    top = max(values) if values else 1
    for y, v in zip(positions, values):
        ax.text(v + top * 0.012, y, f"{v:.1%}" if percent else f"{v:g}", va="center", ha="left", color=INK, fontsize=8)
    ax.set_xlim(0, top * 1.13)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if percent:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    return _png(fig)


# --- document helpers ---------------------------------------------------------------------------
def _font(run, size: float | None = None, bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = LATIN_FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), CJK_FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#").upper())


def _shade(cell, fill: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    cell._element.get_or_add_tcPr().append(shading)


class Builder:
    def __init__(self) -> None:
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = section.right_margin = Cm(2.5)
        section.top_margin = section.bottom_margin = Cm(2.5)
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = LATIN_FONT, Pt(10.5)
        normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), CJK_FONT)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.3
        self.tables = self.figures = 0
        self.width = section.page_width - section.left_margin - section.right_margin

    def title(self, text: str, subtitle: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        _font(p.add_run(text), 18, True)
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(14)
        _font(p.add_run(subtitle), 10, color=MUTED)

    def heading(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 1")
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(14), Pt(6)
        p.paragraph_format.keep_with_next = True
        _font(p.add_run(text), 13, True, INK)

    def text(self, text: str) -> None:
        _font(self.doc.add_paragraph().add_run(text))

    def bullets(self, items: list[str]) -> None:
        for item in items:
            _font(self.doc.add_paragraph(style="List Bullet").add_run(item))

    def _caption(self, text: str, *, before: bool) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = (Pt(8), Pt(3)) if before else (Pt(2), Pt(10))
        p.paragraph_format.keep_with_next = before
        _font(p.add_run(text), 9.5, True)

    TAILS = "（两端各 0.5% 的极端值并入最外侧的柱）"

    def table(self, caption: str, header: list[str], rows: list[list[str]], widths: list[float], note: str | None = None) -> None:
        self.tables += 1
        self._caption(f"表 {self.tables}　{caption}", before=True)
        table = self.doc.add_table(rows=1, cols=len(header))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Cm(width)                                 # the grid; some readers ignore cell widths
        lines = [(table.rows[0].cells, header, True)] + [(table.add_row().cells, r, False) for r in rows]
        for index, (cells, values, is_header) in enumerate(lines):
            for i, (cell, value) in enumerate(zip(cells, values)):
                cell.width = Cm(widths[i])
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.keep_with_next = index < len(lines) - 1     # keep the table on one page
                paragraph.paragraph_format.line_spacing = 1.15
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                _font(paragraph.add_run(str(value)), 9.5, is_header)
                if is_header:
                    _shade(cell, "EEF2F7")
        p = self.doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        if note:
            _font(p.add_run(note), 9, color=MUTED)

    def figure(self, caption: str, image: io.BytesIO) -> None:
        self.figures += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(image, width=min(Cm(15), self.width))
        self._caption(f"图 {self.figures}　{caption}", before=False)


# --- content ------------------------------------------------------------------------------------
def _n(value: int | float) -> str:
    return f"{int(value):,}"


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _g(value: float) -> str:
    return f"{value:g}"


def age_bands(histogram: dict[str, Any]) -> list[tuple[str, int]]:
    """Counts per age band from the one-bar-per-year histogram."""
    ages = [round((a + b) / 2) for a, b in zip(histogram["edges"], histogram["edges"][1:])]
    return [(name, sum(c for age, c in zip(ages, histogram["counts"])
                       if (low is None or age >= low) and (high is None or age <= high))) for low, high, name in AGE_BANDS]


def _table_fields(report: dict[str, Any], preferred: tuple[tuple[str, str], ...], stages: tuple[str, ...]) -> list[tuple[str, str]]:
    """The preferred fields in their order, then any other numeric field of these stages that data.yaml added."""
    listed = {n for group in (BASELINE, MEDICATION) for n, _ in group} | set(SHOWN_ELSEWHERE)
    extra = [(n, f.get("title") or n) for n, f in report["fields"].items()
             if n not in listed and f.get("stage") in stages and "summary" in f]
    return list(preferred) + extra


def _summary_rows(report: dict[str, Any], fields: list[tuple[str, str]]) -> list[list[str]]:
    rows = []
    for name, fallback in fields:
        field = report["fields"].get(name)
        if not field or "summary" not in field or not field["summary"].get("n"):
            continue                                                 # removed from the config, or empty
        s = field["summary"]
        title = field.get("title") or fallback                       # the name given in data.yaml wins
        rows.append([title, _n(s["n"]), f"{_g(s['median'])}（{_g(s['p25'])}–{_g(s['p75'])}）", f"{_g(s['p5'])}–{_g(s['p95'])}",
                     _pct(field["missing_rate"])])
    return rows


def build(report: dict[str, Any], cjk: bool) -> Document:
    def label(chinese: str, english: str) -> str:
        return chinese if cjk else english

    b = Builder()
    rows, split, lab, fields = report["rows"], report["split"], report["label"], report["fields"]
    total = rows["rows_clean"]
    groups = report["bin_names"]
    b.title("促排周期人群统计分析", f"获卵数预测项目 · 清洗后 {_n(total)} 个促排周期")

    b.heading("一、研究人群与数据清洗")
    b.text(f"原始数据共 {_n(rows['rows_raw'])} 个促排周期。清洗时去掉周期号重复的 {_n(rows['rows_duplicate_id'])} 行、"
           f"获卵数缺失或超出合理范围的 {_n(rows['rows_label_missing'])} 行，最终纳入 {_n(total)} 个周期。"
           "字段名称、数据类型、合理范围和缺失的判定规则都写在配置文件中，清洗过程不参考获卵数。")
    b.text("数据按 7:1:2 划分为训练集、验证集和测试集，按获卵数分组分层；测试集单独存放，只用于最终评估。")
    b.table("数据集划分及各集合的获卵数分组构成", ["集合", "周期数", "平均获卵数"] + groups,
            [[{"train": "训练集", "val": "验证集", "test": "测试集"}[part], _n(s["n"]), _g(s["label_mean"])] + [_pct(x) for x in s["bin_share"]]
             for part, s in split.items()], [2.2, 2.0, 2.4] + [9.4 / len(groups)] * len(groups))

    if "age" in fields:
        age = fields["age"]
        s = age["summary"]
        b.heading("二、年龄分布")
        b.text(f"女方年龄中位数 {_g(s['median'])} 岁（P25–P75 {_g(s['p25'])}–{_g(s['p75'])}，P5–P95 {_g(s['p5'])}–{_g(s['p95'])}），"
               f"均值 {_g(s['mean'])} 岁，标准差 {_g(s['std'])}。")
        bands = age_bands(age["histogram"])
        counted = sum(c for _, c in bands)
        b.table("年龄段构成", ["年龄段（岁）", "周期数", "占比"], [[name, _n(c), _pct(c / counted)] for name, c in bands], [4.0, 3.5, 3.5])
        b.figure("女方年龄分布" + b.TAILS, histogram_figure(age["histogram"], label("女方年龄（岁）", "Female age (years)"), label("周期数", "Cycles")))

    s = lab["summary"]
    b.heading("三、获卵数分布")
    zero = lab["value_count"].get("0", 0)
    b.text(f"获卵数均值 {_g(s['mean'])}，标准差 {_g(s['std'])}，中位数 {_g(s['median'])}（P25–P75 {_g(s['p25'])}–{_g(s['p75'])}，"
           f"P5–P95 {_g(s['p5'])}–{_g(s['p95'])}）。分布右偏：多数周期集中在中低水平，少数周期获卵数很高。"
           + (f"获卵数为 0 的周期有 {_n(zero)} 个（{_pct(zero / total)}）。" if zero else ""))
    b.table("获卵数分组构成", ["获卵数分组", "周期数", "占比"],
            [[g, _n(c), _pct(c / total)] for g, c in zip(groups, lab["bin_count"])], [4.0, 3.5, 3.5])
    b.figure("获卵数分布" + b.TAILS, histogram_figure(lab["histogram"], label("获卵数（个）", "Oocytes retrieved"), label("周期数", "Cycles")))

    b.heading("四、基础情况与内分泌指标")
    b.text("各指标以中位数和四分位数描述。AMH 和降调日窦卵泡数的缺失比例较高，使用时需要注意。")
    b.table("基础情况与内分泌指标", ["指标", "有效例数", "中位数（P25–P75）", "P5–P95", "缺失率"], _summary_rows(report, _table_fields(report, BASELINE, ("baseline",))),
            [3.6, 2.0, 4.8, 3.2, 2.2], "数据文件未提供化验指标的单位，表中不标注单位。")
    if "infertility_years" in fields:
        b.figure("不孕年限分布" + b.TAILS, histogram_figure(fields["infertility_years"]["histogram"], label("不孕年限（年）", "Years of infertility"),
                                              label("周期数", "Cycles")))

    b.heading("五、用药方案与用药情况")
    if "protocol" in fields:
        counts = {k: v for k, v in fields["protocol"]["counts"].items() if not k.startswith("（")}
        known = sum(counts.values())
        top = list(counts.items())
        b.text(f"共 {len(top)} 类用药方案，最常用的是{top[0][0]}（{_pct(top[0][1] / known)}）"
               + (f"和{top[1][0]}（{_pct(top[1][1] / known)}）" if len(top) > 1 else "") + "。")
        b.table("用药方案构成", ["用药方案", "周期数", "占比"], [[k, _n(v), _pct(v / known)] for k, v in top], [5.0, 3.0, 3.0])
        b.figure("用药方案构成", bar_figure(list(counts) if cjk else [f"Protocol {i + 1}" for i in range(len(counts))],
                                         [v / known for v in counts.values()], label("占全部周期的比例", "Share of cycles")))
    b.table("促排用药情况", ["指标", "有效例数", "中位数（P25–P75）", "P5–P95", "缺失率"], _summary_rows(report, _table_fields(report, MEDICATION, ("plan", "during"))),
            [4.4, 2.0, 4.8, 2.8, 1.8], "数据文件未提供用量的单位，表中不标注单位。")
    if "gnrha_drug" in fields:
        counts = {k: v for k, v in fields["gnrha_drug"]["counts"].items() if not k.startswith("（")}
        known = sum(counts.values())
        b.table("GnRH 类药物构成", ["药物", "周期数", "占比"], [[k, _n(v), _pct(v / known)] for k, v in counts.items()], [5.0, 3.0, 3.0],
                f"占比按有记录的 {_n(known)} 个周期计算；缺失率 {_pct(fields['gnrha_drug']['missing_rate'])}。")

    flags = sorted(((n, f) for n, f in fields.items() if f["type"] == "flag"), key=lambda kv: -kv[1]["positive"])
    if flags:
        d = report["diagnosis"]
        b.heading("六、不孕因素与诊断")
        b.text(f"诊断为自由文本，共 {_n(d['distinct_texts'])} 种不同写法。按关键词把同义写法归并为 {len(flags)} 个类别"
               "（如“瘢痕子宫”与“疤痕子宫”归为同一类），一个周期可以同时属于多个类别；"
               f"没有匹配到任何已定义类别的 {_n(d['unmatched'])} 个周期（{_pct(d['unmatched'] / total)}）归入“其他”。"
               "男方因素只统计精液异常，男方的染色体异常和地贫归入遗传学指征。")
        b.table("不孕因素与诊断类别构成", ["类别", "周期数", "占比"],
                [[f["title"], _n(f["positive"]), _pct(f["positive_rate"])] for _, f in flags], [8.0, 2.6, 2.6],
                "一个周期可属于多个类别，占比之和大于 100%。")
        shown = [(n, f) for n, f in flags if f["positive"] > 0]
        b.figure("不孕因素与诊断类别构成", bar_figure([f["title"].split("（")[0] if cjk else n for n, f in shown],
                                               [f["positive_rate"] for _, f in shown], label("占全部周期的比例", "Share of cycles")))
        per = d.get("factors_per_cycle")
        if per:
            b.table("每个周期的不孕因素数目", ["因素数目", "周期数", "占比"], [[k, _n(v), _pct(v / total)] for k, v in per.items()],
                    [4.0, 3.5, 3.5], "不含“原发不孕”“继发不孕”这两个分类。")

    b.heading("七、数据处理说明" if flags else "六、数据处理说明")
    zero_fields = [(fields[n]["source"], fields[n]["cleaning"].get("zero_as_missing", 0)) for n in fields
                   if fields[n].get("cleaning", {}).get("zero_as_missing")]
    notes = ["超出合理范围或无法识别为数字的取值记为缺失，不删除整行；只有周期号重复、获卵数缺失或超范围的周期被剔除。"]
    if zero_fields:
        notes.append("以下指标的 0 不是有效测量值，按缺失处理：" + "、".join(f"{src}（{_n(c)} 例）" for src, c in zero_fields) + "。")
    if "infertility_years" in fields:
        notes.append("不孕年限的 0 保留为 0，不按缺失处理。")
    notes.append("血压原始记录为“收缩压/舒张压”，已拆成两个字段。")
    notes.append("周期号只用于去重和数据集划分，不进入任何统计、特征或输出文件。")
    b.bullets(notes)
    return b.doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default="outputs/ovu/report.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    font = chart_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        print("提示：这台机器上没有中文字体，图中的文字改用英文（表格和正文不受影响）")
    out = Path(args.out or Path(args.report).with_name("人群统计分析.docx"))
    build(report, cjk=font is not None).save(out)
    print(f"已生成：{out}")


if __name__ == "__main__":
    main()
