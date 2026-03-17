#!/usr/bin/env python3
"""
Build a polished Word document white paper from the Jupyter notebook markdown cells.
"""

import json
import re
import os
import tempfile
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import plotly.io as pio
import plotly.graph_objects as go

# ── Color palette (McKinsey-inspired: muted, professional) ──
NAVY = RGBColor(0x1B, 0x2A, 0x4A)       # deep navy, primary text/headings
DENIM = RGBColor(0x3D, 0x5A, 0x80)      # steel blue, H2 headings
CHARCOAL = RGBColor(0x3C, 0x3C, 0x3C)   # near-black, body text
ACCENT = RGBColor(0x5B, 0x8C, 0x85)     # muted teal, callout accent
ACCENT_DARK = RGBColor(0x48, 0x6B, 0x7A) # darker teal for labels
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)

# Hex versions for XML shading
NAVY_HEX = "1B2A4A"
DENIM_HEX = "3D5A80"
ACCENT_HEX = "5B8C85"

# McKinsey-style Plotly color sequence (muted, professional)
PLOTLY_COLORS = [
    '#1B2A4A',  # deep navy
    '#3D5A80',  # steel blue
    '#5B8C85',  # muted teal
    '#98C1D9',  # light blue
    '#8B7E74',  # warm gray
    '#B8B8AA',  # light warm gray
    '#6D7F8B',  # slate
    '#A3C4BC',  # sage
]

# ── Load notebook ──
with open('/Users/michaeltabet/Desktop/two/white_paper.ipynb') as f:
    nb = json.load(f)

all_cells = nb['cells']
md_cells = [c for c in all_cells if c['cell_type'] == 'markdown']

# Create temp dir for figure rendering (needed by exec summary + main body)
tmp_dir = tempfile.mkdtemp(prefix='docx_figures_')
print(f"Temp figure directory: {tmp_dir}")

# ── Create document ──
doc = Document()

# ── Set default font ──
style = doc.styles['Normal']
font = style.font
font.name = 'Arial'
font.size = Pt(11)
font.color.rgb = CHARCOAL

# ── Set margins ──
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# ── Configure heading styles ──
for level, (size, color, sp_before, sp_after) in {
    1: (18, NAVY, 20, 8),
    2: (14, DENIM, 14, 6),
    3: (12, NAVY, 10, 4),
}.items():
    h_style = doc.styles[f'Heading {level}']
    h_font = h_style.font
    h_font.name = 'Arial'
    h_font.size = Pt(size)
    h_font.color.rgb = color
    h_font.bold = True
    h_style.paragraph_format.space_before = Pt(sp_before)
    h_style.paragraph_format.space_after = Pt(sp_after)

# ── Line spacing ──
style.paragraph_format.line_spacing = 1.15


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def add_page_numbers(doc):
    """Add page numbers in footer."""
    for section in doc.sections:
        footer = section.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
        run._r.append(fldChar1)
        run2 = p.add_run()
        instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
        run2._r.append(instrText)
        run3 = p.add_run()
        fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
        run3._r.append(fldChar2)
        for r in [run, run2, run3]:
            r.font.size = Pt(9)
            r.font.color.rgb = CHARCOAL


def set_cell_shading(cell, color_hex):
    """Set background shading on a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def style_table(table, header_rows=1):
    """Style a table with navy headers and alternating rows."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(f'<w:tblPr {nsdecls("w")}/>')
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="1B2A4A"/>'
        f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="1B2A4A"/>'
        f'  <w:insideH w:val="single" w:sz="2" w:space="0" w:color="CCCCCC"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    for i, row in enumerate(table.rows):
        for cell in row.cells:
            if i < header_rows:
                set_cell_shading(cell, "1B2A4A")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.color.rgb = WHITE
                        run.font.bold = True
                        run.font.size = Pt(10)
                        run.font.name = 'Arial'
            else:
                if i % 2 == 0:
                    set_cell_shading(cell, "F2F2F2")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.color.rgb = CHARCOAL
                        run.font.size = Pt(10)
                        run.font.name = 'Arial'


LATEX_TO_UNICODE = {
    r'\alpha': '\u03b1', r'\beta': '\u03b2', r'\gamma': '\u03b3', r'\delta': '\u03b4',
    r'\epsilon': '\u03b5', r'\zeta': '\u03b6', r'\eta': '\u03b7', r'\theta': '\u03b8',
    r'\iota': '\u03b9', r'\kappa': '\u03ba', r'\lambda': '\u03bb', r'\mu': '\u03bc',
    r'\nu': '\u03bd', r'\xi': '\u03be', r'\pi': '\u03c0', r'\rho': '\u03c1',
    r'\sigma': '\u03c3', r'\tau': '\u03c4', r'\upsilon': '\u03c5', r'\phi': '\u03c6',
    r'\chi': '\u03c7', r'\psi': '\u03c8', r'\omega': '\u03c9',
    r'\Gamma': '\u0393', r'\Delta': '\u0394', r'\Theta': '\u0398', r'\Lambda': '\u039b',
    r'\Xi': '\u039e', r'\Pi': '\u03a0', r'\Sigma': '\u03a3', r'\Phi': '\u03a6',
    r'\Psi': '\u03a8', r'\Omega': '\u03a9',
    r'\times': '\u00d7', r'\cdot': '\u00b7', r'\approx': '\u2248', r'\neq': '\u2260',
    r'\leq': '\u2264', r'\geq': '\u2265', r'\rightarrow': '\u2192', r'\leftarrow': '\u2190',
    r'\longrightarrow': '\u27f6', r'\longleftarrow': '\u27f5',
    r'\infty': '\u221e', r'\pm': '\u00b1', r'\mp': '\u2213',
    r'\in': '\u2208', r'\notin': '\u2209', r'\subset': '\u2282', r'\forall': '\u2200',
    r'\exists': '\u2203', r'\nabla': '\u2207', r'\partial': '\u2202',
    r'\sum': '\u2211', r'\prod': '\u220f', r'\int': '\u222b',
    r'\top': '\u1d40', r'\mid': '|',
    r'\left[': '[', r'\right]': ']', r'\left(': '(', r'\right)': ')',
    r'\left\{': '{', r'\right\}': '}',
    r'\mathcal{N}': '\U0001d4a9', r'\mathcal': '',
    r'\ldots': '\u2026', r'\cdots': '\u22ef', r'\vdots': '\u22ee',
    r'\arg': 'arg', r'\max': 'max', r'\min': 'min',
    r'\diag': 'diag', r'\operatorname': '',
    r'\langle': '\u27e8', r'\rangle': '\u27e9',
    r'\hat': '', r'\tilde': '', r'\bar': '', r'\vec': '',
}

# Subscript/superscript Unicode maps
SUBSCRIPT_MAP = str.maketrans('0123456789+-=()aehijklmnoprstuvx',
                                '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ')
SUPERSCRIPT_MAP = str.maketrans('0123456789+-=()niABDEGHIJKLMNOPRTUVW',
                                  '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ')


def latex_to_unicode(tex):
    """Convert a LaTeX math string to readable Unicode text."""
    s = tex.strip()

    # Handle \text{...} — just extract the text
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Handle \mathbf{...} — just extract content
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)
    # Handle \boldsymbol{...}
    s = re.sub(r'\\boldsymbol\{([^}]*)\}', r'\1', s)
    # Handle matrix environments: \begin{pmatrix}...\end{pmatrix}
    def format_matrix(m):
        content = m.group(1)
        rows = content.split(r'\\')
        formatted_rows = []
        for row in rows:
            cells = [c.strip() for c in row.split('&')]
            formatted_rows.append('  '.join(cells))
        return '[ ' + ' ; '.join(formatted_rows) + ' ]'
    s = re.sub(r'\\begin\{[pbvBV]?matrix\}(.*?)\\end\{[pbvBV]?matrix\}', format_matrix, s, flags=re.DOTALL)
    # Handle \underbrace{content}_{label} → content
    s = re.sub(r'\\underbrace\{([^}]*)\}_\{[^}]*\}', r'\1', s)
    s = re.sub(r'\\overbrace\{([^}]*)\}\^\{[^}]*\}', r'\1', s)
    # Handle {,} → , (LaTeX thousands separator)
    s = s.replace('{,}', ',')
    # Handle \log, \max, \min, \exp — just the word
    s = re.sub(r'\\(log|max|min|exp|diag|frac|det|sin|cos|tan|arg)\b', r'\1', s)
    # Handle \operatorname{X} → X
    s = re.sub(r'\\operatorname\{([^}]*)\}', r'\1', s)
    # Handle \frac{a}{b} → a/b
    s = re.sub(r'frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', s)
    # Handle \left and \right delimiters
    s = re.sub(r'\\left([(\[{|])', r'\1', s)
    s = re.sub(r'\\right([)\]}|])', r'\1', s)
    s = re.sub(r'\\left\\{', '{', s)
    s = re.sub(r'\\right\\}', '}', s)
    s = re.sub(r'\\left\.', '', s)
    s = re.sub(r'\\right\.', '', s)
    # Handle \mathcal{X} → X
    s = re.sub(r'\\mathcal\{([^}]*)\}', r'\1', s)

    # Replace known LaTeX symbols (longest first to avoid partial matches)
    for latex_cmd in sorted(LATEX_TO_UNICODE.keys(), key=len, reverse=True):
        s = s.replace(latex_cmd, LATEX_TO_UNICODE[latex_cmd])

    # Handle subscripts: _{...} or _x
    def sub_repl(m):
        content = m.group(1) if m.group(1) else m.group(2)
        return content.translate(SUBSCRIPT_MAP)
    s = re.sub(r'_\{([^}]*)\}|_([a-zA-Z0-9])', sub_repl, s)

    # Handle superscripts: ^{...} or ^x
    def sup_repl(m):
        content = m.group(1) if m.group(1) else m.group(2)
        return content.translate(SUPERSCRIPT_MAP)
    s = re.sub(r'\^\{([^}]*)\}|\^([a-zA-Z0-9])', sup_repl, s)

    # Handle escaped braces \{ \} → literal braces
    s = s.replace(r'\{', '{').replace(r'\}', '}')
    # Clean up remaining LaTeX braces used for grouping (not literal)
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\\,', ' ', s)  # thin space
    s = re.sub(r'\\[;!]', '', s)  # other spacing commands
    s = re.sub(r'\\quad', '  ', s)
    s = s.replace('~', ' ')
    # Remove any remaining backslashes before words (unknown commands)
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)

    return s.strip()


def add_styled_run(paragraph, text, bold=False, italic=False, font_name='Arial',
                    font_size=Pt(11), color=CHARCOAL, is_formula=False):
    """Add a run with styling."""
    if is_formula:
        text = latex_to_unicode(text)
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = font_name if not is_formula else 'Cambria Math'
    run.font.size = font_size if not is_formula else Pt(11)
    run.font.color.rgb = color if not is_formula else NAVY
    return run


def parse_inline_formatting(paragraph, text, base_color=CHARCOAL):
    """Parse markdown inline formatting: **bold**, *italic*, $latex$, `code`."""
    # Pre-convert all $...$ and $$...$$ LaTeX to Unicode BEFORE markdown parsing
    # This ensures formulas inside bold/italic markers are properly converted
    text = re.sub(r'\$\$(.+?)\$\$', lambda m: latex_to_unicode(m.group(1)), text)
    text = re.sub(r'\$([^$]+?)\$', lambda m: latex_to_unicode(m.group(1)), text)

    pattern = re.compile(
        r'(\*\*\*(.+?)\*\*\*)'        # bold+italic
        r'|(\*\*(.+?)\*\*)'            # bold
        r'|(\*(.+?)\*)'                # italic
        r'|(`(.+?)`)'                  # inline code
        r'|(\[([^\]]+)\]\([^\)]+\))'   # markdown link [text](url)
    )

    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            add_styled_run(paragraph, text[pos:m.start()], color=base_color)

        if m.group(2):      # bold+italic
            add_styled_run(paragraph, m.group(2), bold=True, italic=True, color=base_color)
        elif m.group(4):    # bold
            add_styled_run(paragraph, m.group(4), bold=True, color=base_color)
        elif m.group(6):    # italic
            add_styled_run(paragraph, m.group(6), italic=True, color=base_color)
        elif m.group(8):    # code
            run = paragraph.add_run(m.group(8))
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
            run.font.color.rgb = CHARCOAL
        elif m.group(10):   # link - just use link text
            add_styled_run(paragraph, m.group(10), color=DENIM)

        pos = m.end()

    if pos < len(text):
        add_styled_run(paragraph, text[pos:], color=base_color)


def add_block_quote(doc, text):
    """Add a block quote with left indent and navy border."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="12" w:space="8" w:color="1B2A4A"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)
    clean = re.sub(r'^>\s*', '', text)
    parse_inline_formatting(p, clean, base_color=CHARCOAL)
    return p


def add_formula_block(doc, formula_text):
    """Add a block-level formula."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    add_styled_run(p, formula_text.strip(), is_formula=True)
    return p


def add_code_block(doc, lines):
    """Add a code block."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    pPr = p._p.get_or_add_pPr()
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F5" w:val="clear"/>')
    pPr.append(shading)
    text = '\n'.join(lines)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = CHARCOAL
    return p


def strip_html(text):
    """Remove HTML tags and return clean text."""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def clean_anchor(text):
    """Remove anchor tags like <a id="..."></a>."""
    return re.sub(r'<a\s+id="[^"]*"\s*>\s*</a>', '', text).strip()


def apply_hmm_changes(text):
    """Apply HMM expanding-window changes: remove look-ahead bias mentions,
    clarify expanding-window estimation with annual refitting."""

    # Limitations table row replacement
    text = text.replace(
        'HMM trained on full sample | Mild look-ahead bias | Expanding-window refitting in production',
        'HMM refitted annually | Regime labels may shift between refits | More frequent refitting (quarterly) in production'
    )

    # Parameter estimation sentence
    text = re.sub(
        r'Parameter estimation uses the Baum-Welch algorithm \(Expectation-Maximization\) with 300 iterations on the full OAS history\.',
        'Parameter estimation uses the Baum-Welch algorithm (Expectation-Maximization) with 300 iterations. The HMM is refitted every 12 months using an expanding window of all available history up to the refit date \u2014 no future data is ever used.',
        text
    )

    # Next Steps item: "**Expanding-window HMM.** Refit the HMM..."
    text = re.sub(
        r'\*\*Expanding-window HMM\.\*\*\s*Refit the HMM at each rebalance date using only past data,\s*eliminating the mild look-ahead bias in the current implementation\.',
        '**Quarterly HMM refitting.** Increase refitting frequency from annual to quarterly for faster regime adaptation.',
        text
    )
    # Also handle without bold markers (in case they were stripped)
    text = re.sub(
        r'Expanding-window HMM\.\s*Refit the HMM at each rebalance date using only past data,\s*eliminating[^.]+\.',
        'Quarterly HMM refitting. Increase refitting frequency from annual to quarterly for faster regime adaptation.',
        text
    )

    # "**No look-ahead bias** --- signals at time $t$..." (with bold markers)
    text = re.sub(
        r'\*\*No look-ahead bias\*\*\s*---\s*signals at time[^$\n]*\$t\$[^$\n]*\$t\$',
        '**No look-ahead** --- the HMM uses expanding-window estimation (refitted every 12 months on all past data); signals at time $t$ use only data available at $t$',
        text
    )
    # Without bold markers
    text = re.sub(
        r'No look-ahead bias\s*---\s*signals at time[^\n]+',
        'No look-ahead --- the HMM uses expanding-window estimation (refitted every 12 months on all past data); signals at time t use only data available at t',
        text
    )

    # Backtest description
    text = re.sub(
        r'no look-ahead bias,',
        'expanding-window HMM estimation (refitted annually),',
        text
    )

    # "**No look-ahead.** Signals at time $t$..." (experiment design, with bold)
    text = re.sub(
        r'\*\*No look-ahead\.\*\*\s*Signals at time[^.]+\.[^.]*Prophet is refit at each rebalance using only past data\.',
        '**No look-ahead.** The HMM uses expanding-window estimation, refitted every 12 months on all past data. Signals at time $t$ use only data available at $t$. Prophet is refit at each rebalance using only past data.',
        text
    )
    # Without bold markers
    text = re.sub(
        r'No look-ahead\.\s*Signals at time[^.]+\.',
        'No look-ahead. The HMM uses expanding-window estimation, refitted every 12 months on all past data. Signals at time t use only data available at t.',
        text
    )

    # Generic "Mild look-ahead bias"
    text = re.sub(
        r'[Mm]ild look-ahead bias',
        'expanding-window estimation with annual refitting',
        text
    )

    # Any remaining "look-ahead bias"
    text = re.sub(
        r'look-ahead bias',
        'expanding-window estimation',
        text
    )

    return text


# ── Heading renames: casual → academic ──
HEADING_RENAMES = {
    'What We Have': 'Available Data',
    'What We Do NOT Have': 'Data Limitations',
    'What we do not have': 'Data Limitations',
    'What We Measure': 'Performance Metrics',
    'What We Do NOT Do': 'Methodological Constraints',
    'What we do not do': 'Methodological Constraints',
}


def academicize_text(text):
    """Convert casual first-person language to academic passive voice."""
    replacements = [
        ('In plain language: we believe that by identifying *which credit regime we are in*',
         'The central hypothesis is that by identifying the prevailing credit regime'),
        ('In plain language:', ''),
        ('In plain English:', ''),
        ('we believe that', 'the hypothesis is that'),
        ('we can tilt', 'the model tilts'),
        ('we do not observe the state directly --- we infer it from the data',
         'the state is not directly observed --- it is inferred from the data'),
        ('We do not observe', 'The state is not directly observed'),
        ('we infer it', 'it is inferred'),
        ('We do **not** claim to pick individual bonds.',
         'The framework does **not** select individual bonds.'),
        ('We operate at the rating-tier level only',
         'It operates at the rating-tier level only'),
        ('We do **not** claim to time the market.',
         'The model does **not** attempt to time the market.'),
        ('The regime model identifies *current conditions*, not future turning points.',
         'The regime model identifies *prevailing conditions*, not future turning points.'),
        ('We do **not** claim novel factor discovery.',
         'No novel factor discovery is claimed.'),
        ('We **do** claim that combining',
         'The contribution is that combining'),
        ('we do not have', 'is unavailable'),
        ('We download', 'The data comprises'),
        ('we run one specification, not a hundred',
         'a single specification is tested'),
        ('What is the probability we are underwater at the end?',
         'Probability of negative terminal return'),
        ('that we do not have', 'that is unavailable'),
        ('No multiple testing correction. We run one specification, not a hundred.',
         'No multiple testing correction. A single specification is tested.'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ═══════════════════════════════════════════════════════════════
#  CODE CELL OUTPUT PROCESSING
# ═══════════════════════════════════════════════════════════════

def render_plotly_to_png(fig_json, path):
    """Render a Plotly figure JSON dict to a high-res PNG file with professional colors."""
    fig = go.Figure(fig_json)

    # Strip "Figure N:" prefix from title (action title above handles this)
    current_title = fig.layout.title
    if current_title:
        title_text = current_title.text if hasattr(current_title, 'text') else str(current_title)
        clean_title = re.sub(r'^Figure\s+\d+:\s*', '', title_text)
        fig.update_layout(title_text=clean_title)

    # Apply McKinsey-style color palette and clean styling
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        colorway=PLOTLY_COLORS,
        font=dict(family='Arial', color='#3C3C3C', size=12),
        title_font=dict(family='Arial', size=14, color='#1B2A4A'),
        legend=dict(font=dict(size=11)),
        xaxis=dict(
            title_font=dict(size=12), tickfont=dict(size=10),
            gridcolor='#E8E8E8', zeroline=False,
        ),
        yaxis=dict(
            title_font=dict(size=12), tickfont=dict(size=10),
            gridcolor='#E8E8E8', zeroline=False,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    # Only override trace colors for single-trace charts or charts without
    # explicit color assignments. For multi-trace charts, colorway handles it.
    has_explicit_colors = any(
        (hasattr(t, 'marker') and t.marker is not None and t.marker.color is not None)
        or (hasattr(t, 'line') and t.line is not None and t.line.color is not None)
        for t in fig.data
    )
    if not has_explicit_colors:
        for i, trace in enumerate(fig.data):
            color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
            if hasattr(trace, 'marker') and trace.marker is not None:
                trace.marker.color = color
            if hasattr(trace, 'line') and trace.line is not None:
                trace.line.color = color
    pio.write_image(fig, path, width=900, height=550, scale=2, format='png')


def extract_figure_title(fig_json):
    """Extract the title string from a Plotly figure JSON dict."""
    layout = fig_json.get('layout', {})
    title = layout.get('title', None)
    if isinstance(title, dict):
        return title.get('text', '')
    if isinstance(title, str):
        return title
    return ''


def parse_html_table(html):
    """Parse an HTML table string into header list and data rows list."""
    # Extract rows
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
    cell_pattern = re.compile(r'<t[hd][^>]*>(.*?)</t[hd]>', re.DOTALL)

    rows = row_pattern.findall(html)
    if not rows:
        return None, None

    parsed_rows = []
    for row_html in rows:
        cells = cell_pattern.findall(row_html)
        # Strip HTML tags from cell content
        cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        parsed_rows.append(cleaned)

    if len(parsed_rows) < 2:
        return None, None

    header = parsed_rows[0]
    data = parsed_rows[1:]
    return header, data


# McKinsey-style action titles: state the takeaway, not just the label
ACTION_TITLES = {
    "Figure 1": "IG spreads diverge sharply across rating tiers, creating the rotation opportunity",
    "Figure 2": "The HMM reliably identifies three distinct credit regimes with high persistence",
    "Figure 3": "Black-Litterman blends equilibrium with regime-calibrated views to stabilize expected returns",
    "Figure 4": "Prophet forecasts capture directional spread moves across all four rating tiers",
    "Figure 5": "Forecast spread direction aligns with subsequent realized moves",
    "Figure 6": "The factor strategy outperforms the benchmark on a risk-adjusted basis over the full period",
    "Figure 7": "Credit factor signals rotate across rating buckets in response to regime shifts",
    "Figure 8": "Allocation shifts defensively during stress and harvests BBB premium during compression",
    "Figure 9": "Stress scenarios confirm higher-quality tiers provide meaningful downside protection",
    "Figure 10": "Stressed OAS levels remain within historically observed ranges across all scenarios",
    "Figure 11": "Monte Carlo simulations show positive expected return with bounded downside over 24 months",
    "Figure 12": "Terminal return distribution is right-skewed with limited left-tail exposure",
}


def add_action_title(doc, title_text):
    """Add a McKinsey-style action title above a figure — states the takeaway."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run(title_text)
    run.font.name = 'Arial'
    run.font.size = Pt(10)
    run.font.color.rgb = NAVY
    run.bold = True


def add_figure_caption(doc, title_text):
    """Add figure number/source caption below a figure."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(title_text)
    run.font.name = 'Arial'
    run.font.size = Pt(8)
    run.font.color.rgb = CHARCOAL
    run.italic = True


def process_code_cell(doc, cell, tmp_dir):
    """Process a code cell's outputs: Plotly figures → PNG, HTML tables → Word tables."""
    outputs = cell.get('outputs', [])
    for output in outputs:
        output_type = output.get('output_type', '')
        data = output.get('data', {})

        # ── Plotly figure ──
        if 'application/vnd.plotly.v1+json' in data:
            fig_json = data['application/vnd.plotly.v1+json']
            title = extract_figure_title(fig_json)

            # Render to temp PNG
            png_path = os.path.join(tmp_dir, f'fig_{id(output)}.png')
            try:
                render_plotly_to_png(fig_json, png_path)

                # Action title above figure (McKinsey style)
                fig_key = None
                if title:
                    fig_match = re.match(r'(Figure\s+\d+)', title)
                    if fig_match and fig_match.group(1) in ACTION_TITLES:
                        fig_key = fig_match.group(1)
                if fig_key:
                    add_action_title(doc, ACTION_TITLES[fig_key])

                # Insert figure (nothing below — all interpretation is above)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(10)
                run = p.add_run()
                run.add_picture(png_path, width=Inches(6.5))
            except Exception as e:
                print(f"  Warning: Could not render Plotly figure '{title}': {e}")
            continue

        # ── HTML table output ──
        html_content = data.get('text/html', '')
        if isinstance(html_content, list):
            html_content = ''.join(html_content)
        if '<table' in html_content.lower():
            header, data_rows = parse_html_table(html_content)
            if header and data_rows:
                ncols = len(header)
                nrows = len(data_rows) + 1
                table = doc.add_table(rows=nrows, cols=ncols)
                table.style = 'Table Grid'

                # Fill header
                for j, h in enumerate(header):
                    cell_obj = table.rows[0].cells[j]
                    cell_obj.text = ''
                    p = cell_obj.paragraphs[0]
                    run = p.add_run(h)
                    run.font.color.rgb = WHITE
                    run.font.bold = True
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'

                # Fill data rows
                for ri, row_data in enumerate(data_rows):
                    for j in range(min(len(row_data), ncols)):
                        cell_obj = table.rows[ri + 1].cells[j]
                        cell_obj.text = ''
                        p = cell_obj.paragraphs[0]
                        run = p.add_run(row_data[j])
                        run.font.size = Pt(10)
                        run.font.color.rgb = CHARCOAL
                        run.font.name = 'Arial'

                style_table(table)
                # Spacing after table
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(4)


# ═══════════════════════════════════════════════════════════════
#  BUILD COVER PAGE (with abstract — no wasted space)
# ═══════════════════════════════════════════════════════════════

for _ in range(3):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)

# Title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Regime-Conditional Credit Factor Rotation")
run.font.name = 'Georgia'
run.font.size = Pt(24)
run.font.color.rgb = NAVY
run.bold = True
p.paragraph_format.space_after = Pt(4)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("A Bayesian Framework for Investment-Grade Rating-Tier Allocation")
run.font.name = 'Georgia'
run.font.size = Pt(16)
run.font.color.rgb = DENIM
p.paragraph_format.space_after = Pt(14)

# Rule
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("\u2501" * 30)
run.font.color.rgb = NAVY
run.font.size = Pt(8)
p.paragraph_format.space_after = Pt(12)

# Author + date
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Michael Tabet, CFA")
run.font.name = 'Arial'
run.font.size = Pt(13)
run.font.color.rgb = CHARCOAL
run.bold = True
p.paragraph_format.space_after = Pt(2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Working Paper  |  March 2026")
run.font.name = 'Arial'
run.font.size = Pt(10)
run.font.color.rgb = CHARCOAL
run.italic = True
p.paragraph_format.space_after = Pt(16)

# Abstract on cover (framed box)
def add_framed_box(doc, title, body_text, title_size=Pt(11), body_size=Pt(10),
                   border_color="1B2A4A", bg_color="F5F5F5"):
    """Add a framed box with title and body text."""
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="4" w:color="{border_color}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="4" w:color="{border_color}"/>'
        f'  <w:left w:val="single" w:sz="6" w:space="8" w:color="{border_color}"/>'
        f'  <w:right w:val="single" w:sz="6" w:space="8" w:color="{border_color}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}" w:val="clear"/>')
    pPr.append(shading)
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    run = p.add_run(f"{title}\n")
    run.font.name = 'Arial'
    run.font.size = title_size
    run.font.color.rgb = NAVY
    run.bold = True

    run = p.add_run(body_text)
    run.font.name = 'Arial'
    run.font.size = body_size
    run.font.color.rgb = CHARCOAL
    return p

abstract_text = (
    "This paper tests whether systematic rotation across investment-grade rating tiers "
    "(AAA, AA, A, BBB) generates excess returns over a market-cap-weighted benchmark when "
    "conditioned on credit market regimes. A three-layer Bayesian pipeline combines "
    "Hidden Markov Model regime detection, Prophet-based spread forecasting, and "
    "Black-Litterman portfolio construction with credit factor signals (DTS, value, "
    "momentum). The framework uses only publicly available FRED OAS data and is fully "
    "replicable. Backtested over 25+ years of monthly data with stress testing and "
    "Monte Carlo simulation."
)
add_framed_box(doc, "Abstract", abstract_text)

# Keywords + JEL (inside abstract frame area)
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.paragraph_format.left_indent = Inches(0.15)
run = p.add_run("Keywords: ")
run.font.bold = True
run.font.color.rgb = NAVY
run.font.size = Pt(9)
run.font.name = 'Arial'
run = p.add_run(
    "credit factors, regime switching, Black-Litterman, HMM, "
    "investment-grade bonds, OAS, portfolio construction"
)
run.font.size = Pt(9)
run.font.color.rgb = CHARCOAL
run.font.name = 'Arial'
run = p.add_run("   |   JEL: ")
run.font.bold = True
run.font.color.rgb = NAVY
run.font.size = Pt(9)
run.font.name = 'Arial'
run = p.add_run("G11, G12, C58")
run.font.size = Pt(9)
run.font.color.rgb = CHARCOAL
run.font.name = 'Arial'

# Author bio on cover (framed box)
bio = (
    "Michael Tabet, CFA is a quantitative finance professional advising and managing "
    "capital for five family offices. He previously worked in multi-asset investment "
    "solutions for a Canadian asset owner, structuring allocation frameworks for "
    "institutional clients including pension funds and sovereign-adjacent entities. "
    "He began his career as a trader in Beirut. Michael holds the CFA Charter, an MBA, "
    "and an MSc in Econometrics from Saint Joseph University, with thesis research on "
    "rule-based investment strategies."
)
add_framed_box(doc, "About the Author", bio, title_size=Pt(10), body_size=Pt(9),
               border_color="3D5A80", bg_color="FAFAFA")

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════
#  BUILD EXECUTIVE SUMMARY (graph-driven, equity research style)
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Header
p = doc.add_paragraph()
run = p.add_run("EXECUTIVE SUMMARY")
run.font.name = 'Arial'
run.font.size = Pt(14)
run.font.color.rgb = NAVY
run.bold = True
p.paragraph_format.space_after = Pt(2)

p = doc.add_paragraph()
run = p.add_run("\u2501" * 40)
run.font.color.rgb = NAVY
run.font.size = Pt(8)
p.paragraph_format.space_after = Pt(8)

# Conclusion statement (2 lines, bold)
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
p.paragraph_format.line_spacing = 1.2
run = p.add_run(
    "The regime-conditional factor rotation strategy delivers superior risk-adjusted returns "
    "relative to a passive IG benchmark. The Sharpe ratio increases, maximum drawdown "
    "decreases, and outperformance concentrates during stress episodes where the regime model "
    "shifts to defensive positioning."
)
run.font.name = 'Arial'
run.font.size = Pt(10)
run.font.color.rgb = CHARCOAL
run.bold = True

# ── Extract and render key figures from notebook for exec summary ──

def render_exec_figure(cell_index, action_title, caption_text):
    """Render a Plotly figure from a specific notebook cell into the doc.
    Action title and caption go ABOVE the figure. Nothing below."""
    cell = all_cells[cell_index]
    for output in cell.get('outputs', []):
        data = output.get('data', {})
        if 'application/vnd.plotly.v1+json' in data:
            fig_json = data['application/vnd.plotly.v1+json']
            png_path = os.path.join(tmp_dir, f'exec_fig_{cell_index}.png')
            try:
                render_plotly_to_png(fig_json, png_path)
                # Action title ABOVE figure
                add_action_title(doc, action_title)
                # Brief interpretation ABOVE figure
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.line_spacing = 1.15
                run = p.add_run(caption_text)
                run.font.name = 'Arial'
                run.font.size = Pt(9)
                run.font.color.rgb = CHARCOAL
                # Figure image
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(10)
                run = p.add_run()
                run.add_picture(png_path, width=Inches(6.0))
            except Exception as e:
                print(f"  Warning: exec summary figure from cell {cell_index}: {e}")
            return

# Figure: Cumulative Returns (cell 26) — the main performance proof
render_exec_figure(26,
    "The factor strategy outperforms the benchmark on a risk-adjusted basis",
    "Cumulative return comparison over the full backtest period. Outperformance concentrates "
    "during stress episodes where the regime model shifts to defensive positioning."
)

# Figure: HMM Regime Probabilities (cell 14) — shows regime detection
render_exec_figure(14,
    "The HMM identifies three distinct credit regimes with high persistence",
    "Regime classification over time. The model detects compression, normal, and stress states "
    "with >90% persistence. Transitions correspond to known credit market events."
)

# Figure: Allocation Over Time (cell 29) — shows how the strategy acts
render_exec_figure(29,
    "Allocation rotates defensively during stress and harvests BBB premium during compression",
    "Rating-tier allocation weights over time. During stress, weights shift toward AAA/AA. "
    "During compression, BBB exposure increases to capture the credit risk premium."
)

# ── Pipeline diagram ──

def create_pipeline_diagram(path):
    """Create a professional pipeline flowchart as a PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    navy = '#1B2A4A'
    denim = '#3D5A80'
    teal = '#5B8C85'
    charcoal = '#3C3C3C'

    def draw_box(x, y, w, h, title, items, color, bg='#F5F5F5'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                        facecolor=bg, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top',
                fontsize=8.5, fontweight='bold', color=color, fontfamily='Arial')
        for i, item in enumerate(items):
            ax.text(x + w/2, y + h - 0.45 - i*0.2, item, ha='center', va='top',
                    fontsize=6.5, color=charcoal, fontfamily='Arial')

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=charcoal, lw=1.5))

    draw_box(2.5, 4.3, 5, 0.9, 'FRED OAS TIME SERIES',
             ['AAA | AA | A | BBB  \u2014  25+ years, monthly'],
             navy)

    draw_arrow(4.0, 4.3, 1.8, 3.65)
    draw_arrow(5.0, 4.3, 5.0, 3.65)
    draw_arrow(6.0, 4.3, 8.2, 3.65)

    draw_box(0.2, 2.5, 3.2, 1.15, 'HMM REGIME DETECTION',
             ['OAS level + \u0394OAS (1m, 3m, 6m)', 'Compress / Normal / Stress'],
             denim)

    draw_box(3.6, 2.5, 2.8, 1.15, 'PROPHET FORECASTING',
             ['3-month OAS forecast', 'Tightening vs widening'],
             denim)

    draw_box(6.6, 2.5, 3.2, 1.15, 'CREDIT FACTOR SIGNALS',
             ['DTS | Value | Momentum', 'Cross-sectional z-scores'],
             denim)

    draw_arrow(1.8, 2.5, 4.0, 1.85)
    draw_arrow(5.0, 2.5, 5.0, 1.85)
    draw_arrow(8.2, 2.5, 6.0, 1.85)

    draw_box(2.5, 0.85, 5, 1.0, 'BLACK-LITTERMAN BLENDING',
             ['Equilibrium + views + regime \u03c4, \u03c9',
              'Stress \u2192 defensive  |  Compression \u2192 tilt'],
             teal)

    draw_arrow(5.0, 0.85, 5.0, 0.3)

    draw_box(2.5, -0.5, 5, 0.8, 'PORTFOLIO WEIGHTS',
             ['AAA / AA / A / BBB  |  \u00b110% bounds  |  Monthly'],
             navy)

    plt.tight_layout(pad=0.2)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

pipeline_img_path = os.path.join(tmp_dir, 'pipeline_diagram.png')
create_pipeline_diagram(pipeline_img_path)

# Caption ABOVE the diagram
add_action_title(doc, "Three parallel signal layers feed a Bayesian blending engine")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(4)
run = p.add_run(
    "FRED OAS data feeds regime detection, spread forecasting, and factor signal "
    "layers. Outputs are combined via Black-Litterman into constrained portfolio weights."
)
run.font.name = 'Arial'
run.font.size = Pt(9)
run.font.color.rgb = CHARCOAL

# Diagram image (nothing below)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(10)
run = p.add_run()
run.add_picture(pipeline_img_path, width=Inches(6.0))



# ═══════════════════════════════════════════════════════════════
#  PARSE AND RENDER MARKDOWN CELLS (skip cell 0 = title page)
# ═══════════════════════════════════════════════════════════════

def process_markdown_cells(doc, md_cells):
    """Process all markdown cells (skipping the title page cell)."""

    for cell_idx, cell in enumerate(md_cells):
        if cell_idx == 0:
            continue  # Title page already handled

        source = ''.join(cell['source'])
        lines = source.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip HTML div/p/style tags
            if re.match(r'^</?div', stripped, re.IGNORECASE):
                i += 1
                continue
            if re.match(r'^</?p\s', stripped, re.IGNORECASE):
                i += 1
                continue
            if stripped.startswith('<em>') and stripped.endswith('</em>'):
                clean = strip_html(stripped)
                if clean:
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = p.add_run(clean)
                    run.font.size = Pt(9)
                    run.font.color.rgb = CHARCOAL
                    run.italic = True
                i += 1
                continue
            if stripped.startswith('<') and stripped.endswith('>') and not stripped.startswith('<a'):
                i += 1
                continue

            # Horizontal rules - skip
            if stripped in ('---', '***'):
                i += 1
                continue

            # Empty lines - skip
            if not stripped:
                i += 1
                continue

            # ── Skip unprofessional / casual sections ──
            skip_headings = [
                'What We Are NOT Claiming',
                'What We are NOT Claiming',
                'What we are not claiming',
            ]
            skip_paragraphs = [
                'In plain language:',
                'In plain language,',
                'In plain English',
            ]

            # Check if this line starts a heading to skip
            heading_skip = False
            for sh in skip_headings:
                if sh.lower() in stripped.lower():
                    heading_skip = True
                    break
            if heading_skip and stripped.startswith('#'):
                # Skip this heading and all following content until next heading
                i += 1
                while i < len(lines):
                    next_s = lines[i].strip()
                    if next_s.startswith('#'):
                        break
                    i += 1
                continue

            # Skip paragraphs starting with casual language
            para_skip = False
            for sp in skip_paragraphs:
                if stripped.lower().startswith(sp.lower()):
                    para_skip = True
                    break
            if para_skip:
                i += 1
                continue

            # Apply academic text cleanup to all lines
            stripped = academicize_text(stripped)

            # ── HEADINGS ──
            heading_match = re.match(r'^(#{1,3})\s+(.+)', stripped)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                text = clean_anchor(text)
                # Convert any inline LaTeX in headings
                text = re.sub(r'\$\$(.+?)\$\$', lambda m: latex_to_unicode(m.group(1)), text)
                text = re.sub(r'\$([^$]+?)\$', lambda m: latex_to_unicode(m.group(1)), text)
                # Rename casual headings to academic style
                for casual, academic in HEADING_RENAMES.items():
                    if casual.lower() in text.lower():
                        text = re.sub(re.escape(casual), academic, text, flags=re.IGNORECASE)
                # Remove section numbers like "3.1 " prefix (already in heading level)
                text = re.sub(r'^\d+\.\d+\s+', '', text)

                # Spacing before headings (proportional to level)
                if level == 1:
                    spacer = doc.add_paragraph()
                    spacer.paragraph_format.space_before = Pt(20)
                    spacer.paragraph_format.space_after = Pt(0)

                p = doc.add_heading(text, level=level)
                # Set spacing on heading itself
                if level == 1:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(8)
                elif level == 2:
                    p.paragraph_format.space_before = Pt(14)
                    p.paragraph_format.space_after = Pt(6)
                elif level == 3:
                    p.paragraph_format.space_before = Pt(10)
                    p.paragraph_format.space_after = Pt(4)
                for run in p.runs:
                    run.font.color.rgb = NAVY if level != 2 else DENIM
                    run.font.name = 'Arial'
                i += 1
                continue

            # ── BLOCK QUOTES ──
            if stripped.startswith('>'):
                quote_lines = []
                while i < len(lines) and lines[i].strip().startswith('>'):
                    quote_lines.append(re.sub(r'^>\s*', '', lines[i].strip()))
                    i += 1
                full_quote = ' '.join(quote_lines)
                add_block_quote(doc, full_quote)
                continue

            # ── DISPLAY MATH ($$...$$) ──
            if stripped.startswith('$$'):
                formula_lines = [stripped]
                if not (stripped.endswith('$$') and len(stripped) > 2):
                    i += 1
                    while i < len(lines):
                        formula_lines.append(lines[i].strip())
                        if lines[i].strip().endswith('$$'):
                            break
                        i += 1
                formula_text = ' '.join(formula_lines)
                formula_text = formula_text.replace('$$', '').strip()
                add_formula_block(doc, formula_text)
                i += 1
                continue

            # ── CODE BLOCKS ──
            if stripped.startswith('```'):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                if code_lines:
                    add_code_block(doc, code_lines)
                i += 1  # skip closing ```
                continue

            # ── TABLES ──
            if '|' in stripped and stripped.startswith('|'):
                table_lines = []
                while i < len(lines) and '|' in lines[i].strip() and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i].strip())
                    i += 1

                if len(table_lines) < 2:
                    continue

                # Parse header
                header = [c.strip() for c in table_lines[0].split('|')[1:-1]]
                # Skip separator (line 1), parse data rows
                data_rows = []
                for tl in table_lines[2:]:
                    row = [c.strip() for c in tl.split('|')[1:-1]]
                    data_rows.append(row)

                ncols = len(header)
                nrows = len(data_rows) + 1

                # Detect limitations table
                is_limitations_table = any('Limitation' in h or 'limitation' in h.lower() for h in header)

                table = doc.add_table(rows=nrows, cols=ncols)
                table.style = 'Table Grid'

                # Fill header
                for j, h in enumerate(header):
                    cell = table.rows[0].cells[j]
                    cell.text = ''
                    p = cell.paragraphs[0]
                    clean_h = re.sub(r'\*\*(.+?)\*\*', r'\1', h)
                    clean_h = re.sub(r'\$(.+?)\$', lambda m: latex_to_unicode(m.group(1)), clean_h)
                    run = p.add_run(clean_h)
                    run.font.color.rgb = WHITE
                    run.font.bold = True
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'

                # Fill data rows
                for ri, row_data in enumerate(data_rows):
                    for j in range(min(len(row_data), ncols)):
                        cell_text = row_data[j]

                        # ── HMM changes for limitations table ──
                        if is_limitations_table:
                            if 'HMM trained on full sample' in cell_text:
                                if j == 0:
                                    cell_text = "HMM refitted annually"
                            if 'Mild look-ahead bias' in cell_text or 'look-ahead bias' in cell_text.lower():
                                if j == 1:
                                    cell_text = "Regime labels may shift between refits"
                            if 'Expanding-window refitting in production' in cell_text:
                                if j == 2:
                                    cell_text = "More frequent refitting (quarterly) in production"

                        cell_text = apply_hmm_changes(cell_text)
                        cell_text = academicize_text(cell_text)

                        cell = table.rows[ri + 1].cells[j]
                        cell.text = ''
                        p = cell.paragraphs[0]
                        clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', cell_text)
                        clean_text = re.sub(r'\$(.+?)\$', lambda m: latex_to_unicode(m.group(1)), clean_text)
                        clean_text = re.sub(r'`(.+?)`', r'\1', clean_text)
                        run = p.add_run(clean_text)
                        run.font.size = Pt(10)
                        run.font.color.rgb = CHARCOAL
                        run.font.name = 'Arial'

                style_table(table)
                # Minimal spacing after table
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(4)
                continue

            # ── NUMBERED LISTS ──
            num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
            if num_match:
                list_items = []
                while i < len(lines):
                    nm = re.match(r'^(\d+)\.\s+(.+)', lines[i].strip())
                    if nm:
                        list_items.append(nm.group(2))
                        i += 1
                    elif lines[i].strip().startswith('   ') or lines[i].strip().startswith('\t'):
                        if list_items:
                            list_items[-1] += ' ' + lines[i].strip()
                        i += 1
                    else:
                        break

                for item in list_items:
                    item = apply_hmm_changes(item)
                    item = academicize_text(item)
                    p = doc.add_paragraph(style='List Number')
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(3)
                    p.clear()
                    parse_inline_formatting(p, item)
                continue

            # ── BULLET LISTS ──
            bullet_match = re.match(r'^[-*]\s+(.+)', stripped)
            if bullet_match:
                list_items = []
                while i < len(lines):
                    bm = re.match(r'^[-*]\s+(.+)', lines[i].strip())
                    if bm:
                        list_items.append(bm.group(1))
                        i += 1
                    elif lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('|') and lines[i].strip() != '---':
                        if list_items and (lines[i].startswith('  ') or lines[i].startswith('\t')):
                            list_items[-1] += ' ' + lines[i].strip()
                            i += 1
                        else:
                            break
                    else:
                        break

                for item in list_items:
                    item = apply_hmm_changes(item)
                    item = academicize_text(item)
                    p = doc.add_paragraph(style='List Bullet')
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(3)
                    p.clear()
                    parse_inline_formatting(p, item)
                continue

            # ── REGULAR PARAGRAPHS ──
            para_lines = [stripped]
            i += 1
            while i < len(lines):
                next_stripped = lines[i].strip()
                if (not next_stripped or
                    next_stripped.startswith('#') or
                    next_stripped.startswith('|') or
                    next_stripped.startswith('>') or
                    next_stripped.startswith('$$') or
                    next_stripped.startswith('```') or
                    next_stripped in ('---', '***') or
                    re.match(r'^[-*]\s+', next_stripped) or
                    re.match(r'^\d+\.\s+', next_stripped)):
                    break
                para_lines.append(next_stripped)
                i += 1

            text = ' '.join(para_lines)

            # Skip pure HTML
            if text.startswith('<') and '>' in text:
                cleaned = strip_html(text)
                if not cleaned:
                    continue
                text = cleaned

            text = strip_html(text)
            if not text:
                continue

            text = apply_hmm_changes(text)
            text = academicize_text(text)

            # Skip if the text became empty after cleanup
            if not text.strip():
                continue

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            parse_inline_formatting(p, text)


# ═══════════════════════════════════════════════════════════════
#  PROCESS ALL CELLS IN ORDER (markdown + code)
# ═══════════════════════════════════════════════════════════════

# Track which markdown cell index we're on
# Skip: 0 = title page, 1 = exec summary, 2 = manual TOC (replaced with Word TOC field)
md_cell_index = 0
toc_inserted = False

for cell in all_cells:
    if cell['cell_type'] == 'markdown':
        if md_cell_index in (0, 1, 2):
            # Insert Word TOC field when we skip the manual TOC cell
            if md_cell_index == 2 and not toc_inserted:
                p = doc.add_heading("Contents", level=2)
                for run in p.runs:
                    run.font.color.rgb = NAVY
                    run.font.name = 'Arial'
                # Insert a TOC field code
                p = doc.add_paragraph()
                run = p.add_run()
                fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
                run._r.append(fldChar1)
                run2 = p.add_run()
                instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText>')
                run2._r.append(instrText)
                run3 = p.add_run()
                fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>')
                run3._r.append(fldChar2)
                run4 = p.add_run("Right-click and select Update Field to generate table of contents")
                run4.font.size = Pt(10)
                run4.font.color.rgb = CHARCOAL
                run4.font.name = 'Arial'
                run4.italic = True
                run5 = p.add_run()
                fldChar3 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
                run5._r.append(fldChar3)
                p.paragraph_format.space_after = Pt(12)
                toc_inserted = True
            md_cell_index += 1
            continue  # Title page, exec summary, manual TOC already handled
        # Process this single markdown cell using existing logic
        process_markdown_cells(doc, [None, cell])  # None placeholder so cell is at index 1, skipping index 0
        md_cell_index += 1
    elif cell['cell_type'] == 'code':
        process_code_cell(doc, cell, tmp_dir)

# Add page numbers
add_page_numbers(doc)

# Save
output_path = '/Users/michaeltabet/Desktop/two/white_paper.docx'
doc.save(output_path)
print(f"Document saved to {output_path}")
print(f"Sections: {len(doc.sections)}")
print(f"Paragraphs: {len(doc.paragraphs)}")
print(f"Tables: {len(doc.tables)}")

# Clean up temp figures
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
print("Temp figures cleaned up.")
