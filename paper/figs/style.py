# Shared style for the paper figures (matplotlib only): fonts, physical sizes, one method-to-colour mapping used in every
# figure, result loading, and the export step with the blocking panel-alignment gate of the figure skill.
import glob, json, os, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

# The local auditor is bundled with the figure source for portable reproduction.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

PILOT = str(Path(__file__).resolve().parents[2] / 'pilot')
OUT = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
SINGLE, ONE_HALF, DOUBLE = 89 * MM, 140 * MM, 183 * MM
# figures are drawn at Elsevier's 1.5-column width, which is also about the text width of the single-column manuscript,
# so text keeps its size (6-7 pt) both in the review PDF and in the final layout
FIGW = ONE_HALF

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.frameon': False,
    'lines.linewidth': 1.0,
})

# one neutral family (references), one blue baseline family, one red family for NIRA and its ablation
METHOD_STYLE = {
    'M0_oracle':        dict(label='Clean-seed oracle', color='#272727', marker='*', ms=6.5, mfc='white'),
    'M1_naive':         dict(label='Split CP', color='#8C8C8C', marker='s', ms=3.6),
    'M2b_rule_filter':  dict(label='Rel.-filtered CP (RULE)', color='#7884B4', marker='v', ms=4.0),
    'M3_em_1view':      dict(label='Single-view EM', color='#B4C0E4', marker='D', ms=3.2),
    'B1_nacp_true_eta': dict(label='NACP, true $\\eta$', color='#484878', marker='^', ms=4.0),
    'B2_nacp_mpe_eta':  dict(label='NACP, estimated $\\eta$', color='#484878', marker='^', ms=4.0, mfc='white'),
    'M4_em_2view':      dict(label='NIRA w/o name-hard kernel', color='#E9A6A1', marker='o', ms=3.8),
    'M4n_em_namehard':  dict(label='NIRA (ours)', color='#B64342', marker='o', ms=4.6),
}
SPLITS = [('zh_en', 'ZH–EN'), ('ja_en', 'JA–EN'), ('fr_en', 'FR–EN')]
SPLITS_ALL = SPLITS + [('fbdb', 'FB15K–DB15K')]
MODELS = ['RULE', 'PMF']


def load_runs(prefer_baselines=True):
    """(model, split, cond, seed) -> result dict for DBP15K (the re-analysis with the NACP baselines when it exists)
    and FB15K-DB15K (analysed with the NACP baselines from the start)."""
    dirs = {model: [f'label_free_v2/main/{model.lower()}'] for model in ('RULE', 'PMF')}
    runs = {}
    for model, cands in dirs.items():
        for d in (cands if prefer_baselines else [c for c in cands if not c.endswith('_b')]):
            for p in glob.glob(f'{PILOT}/runs/{d}/*_results.json'):
                m = re.match(r'(zh_en|ja_en|fr_en|fbdb)_(syn0\.2|syn0\.5|llm)_s(\d+)_results\.json', os.path.basename(p))
                if not m:
                    continue
                key = (model, m.group(1), m.group(2), int(m.group(3)))
                if key not in runs:
                    runs[key] = json.load(open(p, encoding='utf-8'))
    assert len(runs) == 72, f'Expected 72 corrected main results, found {len(runs)}'
    return runs


def agg(r, key, method, field):
    a = r['results'][key]['agg']
    if method not in a or field not in a[method]:
        return None
    return a[method][field]['mean']


def panel_label(ax, label, x_offset_pt=-14, y_offset_pt=2):
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + off, fontsize=8, fontweight='bold', ha='left', va='bottom')


def save(fig, name, alignment=None):
    """blocking alignment gate (multi-panel), then vector PDF/SVG and a 600-dpi PNG preview"""
    base = os.path.join(OUT, name)
    if alignment is not False:
        require_matplotlib_panel_alignment(fig, json_out=base + '.alignment.json', overlay_svg=base + '.alignment.svg',
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
                                           **(alignment or {}))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.svg')
    fig.savefig(base + '.png', dpi=600)
    fig.savefig(base + '.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return base + '.pdf'
