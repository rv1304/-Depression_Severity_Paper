"""Build an IEEE-conference-style Word version of main_final.tex that follows the PDF layout.

    python paper/final/make_docx.py --pandoc <path-to-pandoc> --csl <ieee.csl> --aux <main_final.aux> --work <dir>

Steps: (1) rewrite the LaTeX so pandoc sees explicit IEEE numbering (I., A., TABLE I, Fig. 1) and figure widths,
(2) convert with pandoc against a reference document carrying IEEE styles (Times New Roman 10 pt, small-caps
section heads, bordered tables), (3) patch the Word XML for the page layout: a one-column title block with a
three-column author table, a two-column body, and one-column sections around the wide tables and figures.
Figures must already exist as PNG files in <work>/figures (the PDFs converted with pdftoppm)."""
import argparse
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
TITLE = ("Benchmarking Classical Machine Learning for Four-Level Depression Severity Classification from Text: "
         "A Leakage-Controlled Study with Clinical-Lexicon and Label-Noise Analyses")
AUTHORS = ["Raghav Vashisht", "Aanandit Kanwar", "Apurav Gautam"]
AFFIL = ["Department of Computer Science and Engineering", "Chitkara Institute of Engineering (CUIET)", "Punjab, India"]
WIDE = {"tab:scen", "tab:metrics", "fig:cm", "fig:sym"}
COL_IN, WIDE_IN = 3.45, 7.0


def read_labels(aux: Path) -> dict:
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{", aux.read_text()):
        labels[m.group(1)] = m.group(2).replace("\\mbox  {", "").replace("}", "").strip()
    return labels


def prepare_tex(src: str, lab: dict) -> str:
    s = src[src.index("\\begin{abstract}"):src.index("\\bibliographystyle")]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1).strip()
    s = s.replace(re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", s, re.S).group(0),
                  "ABSTRACTMARK " + abstract)
    s = s.replace("\\begin{IEEEkeywords}", "KEYWORDSMARK ").replace("\\end{IEEEkeywords}", "")

    # explicit IEEE numbering: I. II. ... for sections, A. B. ... for subsections
    out, sec, sub = [], 0, 0
    for line in s.split("\n"):
        m = re.match(r"\\section\{(.*?)\}(.*)", line)
        if m:
            sec += 1
            sub = 0
            line = f"\\section*{{{ROMAN[sec - 1]}. {m.group(1)}}}"
        m = re.match(r"\\subsection\{(.*?)\}(\\label\{[^}]*\})?(.*)", line)
        if m:
            sub += 1
            line = f"\\subsection*{{{chr(64 + sub)}. {m.group(1)}}}{m.group(3)}"
        out.append(line)
    s = "\n".join(out)

    # the pipeline drawing is a pre-rendered PNG; figure widths follow the column layout
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", r"\\includegraphics{fig_pipeline}", s, flags=re.S)

    def float_env(m):
        env, body = m.group(1), m.group(2)
        l = re.search(r"\\label\{([^}]*)\}", body)
        key = l.group(1) if l else ""
        wide = key in WIDE
        n = lab.get(key, "")
        pre = f"TABLE {n}. " if env.startswith("table") else f"Fig. {n}. "
        body = re.sub(r"\\caption\{", lambda _: "\\caption{" + pre, body, count=1)
        imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body)
        w = WIDE_IN / max(len(imgs), 1) - (0.1 if len(imgs) > 1 else 0) if wide else COL_IN
        body = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^}.]*)\}",
                      lambda mm: f"\\includegraphics[width={w:.2f}in]{{{mm.group(1)}.png}}", body)
        base = env.rstrip("*")
        block = f"\\begin{{{base}}}{body}\\end{{{base}}}"
        return f"\n\nWIDESTARTMARK\n\n{block}\n\nWIDEENDMARK\n\n" if wide else block
    s = re.sub(r"\\begin\{(table\*?|figure\*?)\}(.*?)\\end\{\1\}", float_env, s, flags=re.S)

    s = re.sub(r"\\num\{(\d+)\}", lambda m: f"{int(m.group(1)):,}", s)
    s = s.replace("\\begin{split}", "\\begin{aligned}").replace("\\end{split}", "\\end{aligned}")

    def eq(m):
        body = m.group(1)
        l = re.search(r"\\label\{([^}]*)\}", body)
        n = lab.get(l.group(1), "") if l else ""
        body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
        return f"\\[ {body} \\qquad ({n}) \\]"
    s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq, s, flags=re.S)
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({lab[m.group(1)]})", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab[m.group(1)], s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = s.replace("\\footnotesize", "").replace("\\hfill", " ").replace("\\section*{Acknowledgment}", "\\section*{Acknowledgment}")
    return ("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\nTITLEMARK\n\n" + s
            + "\n\\section*{References}\n\\end{document}\n")


# ----------------------------------------------------------------------------------------- styles
RPR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'


def pstyle(sid, name, ppr, rpr, based="Normal"):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/><w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{RPR}{rpr}</w:rPr></w:style>')


STYLES = {
    "Normal": ('<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
               '<w:pPr><w:spacing w:before="0" w:after="0" w:line="228" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
               f'<w:rPr>{RPR}<w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-US"/></w:rPr></w:style>'),
    "BodyText": pstyle("BodyText", "Body Text", '<w:spacing w:before="0" w:after="0"/><w:ind w:firstLine="202"/><w:jc w:val="both"/>', ""),
    "FirstParagraph": pstyle("FirstParagraph", "First Paragraph", '<w:spacing w:before="0" w:after="0"/><w:ind w:firstLine="202"/><w:jc w:val="both"/>', "", "BodyText"),
    "Compact": pstyle("Compact", "Compact", '<w:spacing w:before="0" w:after="0"/><w:jc w:val="left"/>', ""),
    "Title": pstyle("Title", "Title", '<w:spacing w:before="0" w:after="240"/><w:jc w:val="center"/>', '<w:sz w:val="48"/><w:szCs w:val="48"/>'),
    "Abstract": pstyle("Abstract", "Abstract", '<w:spacing w:before="0" w:after="120"/><w:ind w:firstLine="202"/><w:jc w:val="both"/>', '<w:b/><w:sz w:val="18"/><w:szCs w:val="18"/>'),
    "Heading1": pstyle("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="160" w:after="80"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/>', '<w:smallCaps/><w:sz w:val="20"/><w:szCs w:val="20"/>'),
    "Heading2": pstyle("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/>', '<w:i/><w:sz w:val="20"/><w:szCs w:val="20"/>'),
    "TableCaption": pstyle("TableCaption", "Table Caption", '<w:keepNext/><w:spacing w:before="120" w:after="80"/><w:jc w:val="center"/>', '<w:smallCaps/><w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "ImageCaption": pstyle("ImageCaption", "Image Caption", '<w:spacing w:before="60" w:after="160"/><w:jc w:val="both"/>', '<w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "CaptionedFigure": pstyle("CaptionedFigure", "Captioned Figure", '<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>', ""),
    "Figure": pstyle("Figure", "Figure", '<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>', ""),
    "TableText": pstyle("TableText", "Table Text", '<w:spacing w:before="20" w:after="20"/><w:jc w:val="left"/>', '<w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "Bibliography": pstyle("Bibliography", "Bibliography", '<w:spacing w:before="0" w:after="40"/><w:ind w:left="360" w:hanging="360"/><w:jc w:val="both"/>', '<w:sz w:val="16"/><w:szCs w:val="16"/>'),
    "Table": ('<w:style w:type="table" w:customStyle="1" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
              '<w:pPr><w:spacing w:before="0" w:after="0"/><w:jc w:val="left"/></w:pPr>'
              f'<w:rPr>{RPR}<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
              '<w:tblPr><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
              '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'
              '<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr>'
              '<w:tblStylePr w:type="firstRow"><w:rPr><w:b w:val="0"/></w:rPr><w:tcPr><w:tcBorders>'
              '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders></w:tcPr></w:tblStylePr></w:style>'),
}


def write_reference(default_ref: Path, out: Path) -> None:
    zin = zipfile.ZipFile(default_ref)
    styles = zin.read("word/styles.xml").decode()
    for sid, xml in STYLES.items():
        pat = re.compile(r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % sid, re.S)
        styles = pat.sub(lambda _: xml, styles, count=1) if pat.search(styles) else styles.replace("</w:styles>", xml + "</w:styles>")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = styles.encode() if item.filename == "word/styles.xml" else zin.read(item.filename)
            zout.writestr(item, data)


# ----------------------------------------------------------------------------------------- layout patch
def sect(cols: int, final: bool = False) -> str:
    page = ('<w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="720" w:gutter="0"/>')
    c = f'<w:cols w:num="{cols}" w:space="360"/>' if cols > 1 else '<w:cols w:space="720"/>'
    return f'<w:sectPr><w:type w:val="continuous"/>{page}{c}</w:sectPr>'


def para_with_text(xml: str, text: str) -> re.Match:
    return re.search(r'<w:p>(?:(?!<w:p>).)*?' + re.escape(text) + r'.*?</w:p>', xml, re.S)


def run(text, size=None, italic=False, bold=False):
    rpr = RPR + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + (f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else "")
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'


def title_block() -> str:
    """Title, then the authors as in the PDF: two side by side, the third centred below."""
    title = f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>{run(TITLE, 48)}</w:p>'

    def cell(name, width, span=1):
        lines = f'<w:p><w:pPr><w:spacing w:before="120"/><w:jc w:val="center"/></w:pPr>{run(name, 22)}</w:p>'
        lines += "".join(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{run(a, 20)}</w:p>' for a in AFFIL)
        grid = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
        return f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{grid}</w:tcPr>{lines}</w:tc>'
    nil = "".join(f'<w:{b} w:val="nil"/>' for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
    table = (f'<w:tbl><w:tblPr><w:tblW w:w="10440" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{nil}</w:tblBorders>'
             '<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid><w:gridCol w:w="5220"/><w:gridCol w:w="5220"/></w:tblGrid>'
             f'<w:tr>{cell(AUTHORS[0], 5220)}{cell(AUTHORS[1], 5220)}</w:tr>'
             f'<w:tr>{cell(AUTHORS[2], 10440, span=2)}</w:tr></w:tbl>')
    end = f'<w:p><w:pPr><w:spacing w:before="0" w:after="200"/>{sect(1)}</w:pPr></w:p>'
    return title + table + end


COL_TW, WIDE_TW, MIN_TW = 5040, 10440, 560


def fit_tables(xml: str) -> str:
    """Give every data table a fixed grid that fills its column (or the page for wide tables), with each
    column's share set by its longest cell text, so no column collapses to one letter per line."""
    def fix(m):
        t = m.group(0)
        if "<w:tblStyle" not in t:
            return t  # the author block keeps its own layout
        rows = re.findall(r"<w:tr[ >].*?</w:tr>", t, re.S)
        cells = [[re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)] for r in rows]
        ncol = max(len(r) for r in cells)
        header = "".join(cells[0])
        wide = ncol >= 8 or header.startswith("IDData")
        total = WIDE_TW if wide else COL_TW
        need = [max([len(r[i]) for r in cells if i < len(r)] + [1]) for i in range(ncol)]
        weight = [min(n, 46) + 3 for n in need]
        widths = [max(MIN_TW, round(total * w / sum(weight))) for w in weight]
        widths[weight.index(max(weight))] -= sum(widths) - total
        t = re.sub(r"<w:tblW [^>]*/>", f'<w:tblW w:w="{total}" w:type="dxa"/>', t, count=1)
        t = re.sub(r"<w:tblLayout [^>]*/>", "", t)
        # schema order inside tblPr: tblLayout comes before tblCellMar, tblLook, tblCaption and tblDescription
        pr = re.search(r"<w:tblPr>.*?</w:tblPr>", t, re.S).group(0)
        anchor = re.search(r"<w:tblCellMar|<w:tblLook|<w:tblCaption|<w:tblDescription|</w:tblPr>", pr)
        pr2 = pr[:anchor.start()] + '<w:tblLayout w:type="fixed"/>' + pr[anchor.start():]
        t = t.replace(pr, pr2, 1)
        t = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
                   + "</w:tblGrid>", t, count=1, flags=re.S)
        t = re.sub(r"<w:tcW [^>]*/>", "", t)
        # 8 pt table text with tight rows, as in the PDF: cell paragraphs get their own style
        t = re.sub(r'<w:pStyle w:val="Compact"\s*/>', '<w:pStyle w:val="TableText" />', t)
        return t
    return re.sub(r"<w:tbl>.*?</w:tbl>", fix, xml, flags=re.S)


def lead_para(match_xml: str, label: str) -> str:
    """Abstract and Index Terms: IEEE style, bold, with an italic run-in label."""
    body = match_xml.replace(label, "", 1)
    body = re.sub(r'(<w:t xml:space="preserve">)\s+', r"\1", body, count=1)
    body = re.sub(r'<w:r><w:t xml:space="preserve"></w:t></w:r>', "", body)
    body = re.sub(r"<w:pPr>.*?</w:pPr>", "", body, count=1, flags=re.S)
    inner = body[len("<w:p>"):-len("</w:p>")]
    inner = re.sub(r"<w:rPr>", "<w:rPr><w:b/>", inner)
    inner = re.sub(r"<w:r>(?!<w:rPr>)", "<w:r><w:rPr><w:b/></w:rPr>", inner)
    name = label.replace("MARK", "").strip().title().replace("Keywords", "Index Terms") + "\u2014"
    return f'<w:p><w:pPr><w:pStyle w:val="Abstract"/></w:pPr>{run(name, 18, italic=True, bold=True)}{inner}</w:p>'


def patch_layout(docx: Path) -> None:
    zin = zipfile.ZipFile(docx)
    xml = zin.read("word/document.xml").decode()
    xml = xml.replace(para_with_text(xml, "TITLEMARK").group(0), title_block(), 1)
    for label in ("ABSTRACTMARK", "KEYWORDSMARK"):
        m = para_with_text(xml, label)
        xml = xml.replace(m.group(0), lead_para(m.group(0), label), 1)
    # one-column sections around wide floats: a sectPr ends the section before it
    while (m := para_with_text(xml, "WIDESTARTMARK")):
        xml = xml.replace(m.group(0), f'<w:p><w:pPr>{sect(2)}</w:pPr></w:p>', 1)
    while (m := para_with_text(xml, "WIDEENDMARK")):
        xml = xml.replace(m.group(0), f'<w:p><w:pPr>{sect(1)}</w:pPr></w:p>', 1)
    # body ends in two columns, on US Letter with IEEE margins
    xml = re.sub(r"<w:sectPr(?: [^>]*)?>(?:(?!<w:sectPr).)*?</w:sectPr>\s*</w:body>",
                 lambda _: sect(2).replace('<w:type w:val="continuous"/>', '<w:type w:val="continuous"/>') + "</w:body>", xml, count=1, flags=re.S)
    xml = fit_tables(xml)
    out = docx.with_suffix(".tmp")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename.endswith("/"):
                continue
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = xml.encode()
            elif item.filename == "[Content_Types].xml" and b'Extension="png"' not in data:
                data = data.replace(b"<Types ", b"<Types ", 1).replace(
                    b"</Types>", b'<Default Extension="png" ContentType="image/png"/></Types>', 1)
            zout.writestr(item, data)
    out.replace(docx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pandoc", required=True)
    ap.add_argument("--csl", required=True)
    ap.add_argument("--aux", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "Depression_Severity_Paper_revised.docx")
    a = ap.parse_args()
    tex = prepare_tex((HERE / "main_final.tex").read_text(), read_labels(a.aux))
    (a.work / "word.tex").write_text(tex)
    subprocess.run([a.pandoc, "-o", str(a.work / "ref_default.docx"), "--print-default-data-file", "reference.docx"], check=True)
    write_reference(a.work / "ref_default.docx", a.work / "ieee_ref.docx")
    subprocess.run([a.pandoc, str(a.work / "word.tex"), "-f", "latex", "-t", "docx", "--citeproc",
                    "--bibliography", str(HERE.parent / "refs.bib"), "--csl", a.csl,
                    "--reference-doc", str(a.work / "ieee_ref.docx"), "--resource-path", str(a.work / "figures"),
                    "-o", str(a.out)], check=True)
    patch_layout(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
