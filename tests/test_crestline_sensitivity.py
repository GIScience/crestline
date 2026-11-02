from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely

from crestline.crestline import crestline
from test_crestline import CASES, center_raster
from test_crestline_ablation import _scene, MODS
from test_crestline_occluded import OCCLUDED, _ring, _occlude

# Define output directory matching requested path
TEST_OUTPUT_DIR = Path(__file__).parent / 'output' / 'test_sensitivity'

# Define parameter value grids
SPUR_VALUES = (1, 1.5, 2, 2.5, 3)
TIP_VALUES = (0.05, 0.25, 0.5, 0.75, 0.95)
RADIUS_K_VALUES = (0.05, 0.15, 0.35, 0.55, 0.65)

# Global accumulator to track results across test runs for the final report
_TEST_RESULTS = []


# Standalone builder functions for test geometries
def make_parallel_roads():
    return _scene((102, 410), (5, 60, 5, 405), (85, 12, 5, 405)).mask


def make_noisy_road():
    road = _scene((80, 300), (25, 30, 10, 290))
    for row, col in MODS:
        road.mask[row: row + 6, col: col + 6] = True
    return road.mask


def make_occluded_road():
    road = _scene((81, 420), (20, 41, 10, 410))
    road.mask[36:39, 150:156] = False
    return road.mask


def make_pinched_road():
    road = _scene((90, 300), (10, 30, 10, 290), (70, 5, 10, 290))
    road.mask[70:72, 150:154] = False
    road.mask[73:75, 150:154] = False
    return road.mask


def make_cut_road():
    road = _scene((90, 300), (10, 30, 10, 290), (70, 5, 10, 290))
    road.mask[70:75, 150:152] = False
    return road.mask


def make_diagonal_road():
    rows, cols = np.indices((200, 200))
    return np.abs(rows - cols) <= 21


# Comprehensive dictionary of all test geometries across files
ALL_SENSITIVITY_CASES = {
    **{name: (lambda m=make(): m, n) for name, (make, n) in CASES.items()},
    'ablation_parallel_roads': (make_parallel_roads, 2),
    'ablation_noisy_road': (make_noisy_road, 1),
    'ablation_occluded_road': (make_occluded_road, 1),
    'ablation_pinched_road': (make_pinched_road, 2),
    'ablation_cut_road': (make_cut_road, 3),
    'ablation_diagonal_road': (make_diagonal_road, 1),
    'occluded_car': (lambda: OCCLUDED['car'][0](), 2),
    'occluded_centre': (lambda: OCCLUDED['centre'][0](), 2),
    'occluded_trees': (lambda: OCCLUDED['trees'][0](), 2),
    'occluded_crossing': (lambda: OCCLUDED['crossing'][0](), 4),
    'occluded_ring': (_ring, 1),
    'occluded_blocked_ring': (lambda: _occlude(_ring(), (15, 100, 3, 3)), 1),
}


@pytest.fixture(scope='session', autouse=True)
def generate_sensitivity_report_at_end(request):
    """
    Session fixture that compiles and writes out tests/output/test_sensitivity/sensitivity_report.md
    separating passing and failing configurations into distinct sections after all tests finish.
    """
    yield

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TEST_OUTPUT_DIR / "sensitivity_report.md"

    total_runs = len(_TEST_RESULTS)
    if total_runs == 0:
        return

    failing_results = [r for r in _TEST_RESULTS if r['status'] == '**FAIL**']
    passing_results = [r for r in _TEST_RESULTS if r['status'] == 'PASS']
    total_failures = len(failing_results)

    report_lines = [
        "# Crestline Algorithm Sensitivity Analysis Report",
        "\nThis report summarizes behavioral variations across parameter sweeps for `spur`, `tip`, and `radius_k` against default baselines.\n",
        "## Executive Summary",
        f"- **Total Configurations Tested:** {total_runs}",
        f"- **Matching Baseline (Pass):** {total_runs - total_failures}",
        f"- **Diverging Behavior (Fail/Sensitive):** {total_failures}\n",

        "## Failing (Sensitive) Configurations",
        "The following parameter combinations produced outputs that diverged from the baseline:\n"
    ]

    if failing_results:
        report_lines.extend([
            "| Test Geometry | Spur | Tip | Radius_k | Status | Lines Generated | Plot Path |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
        ])
        for r in failing_results:
            report_lines.append(
                f"| {r['name']} | {r['spur']} | {r['tip']} | {r['radius_k']} | {r['status']} | {r['n_lines']} | `{r['rel_path']}` |"
            )
    else:
        report_lines.append("_No failing configurations found._\n")

    report_lines.extend([
        "\n## Passing Configurations",
        "The following parameter combinations produced outputs that matched the baseline exactly:\n",
        "| Test Geometry | Spur | Tip | Radius_k | Status | Lines Generated | Plot Path |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ])

    for r in passing_results:
        report_lines.append(
            f"| {r['name']} | {r['spur']} | {r['tip']} | {r['radius_k']} | {r['status']} | {r['n_lines']} | `{r['rel_path']}` |"
        )

    final_report = "\n".join(report_lines)
    report_path.write_text(final_report)
    print(f"\n[Sensitivity Report] Generated summary report at: {report_path}")


@pytest.mark.parametrize('name', list(ALL_SENSITIVITY_CASES.keys()))
@pytest.mark.parametrize('spur', SPUR_VALUES)
@pytest.mark.parametrize('tip', TIP_VALUES)
@pytest.mark.parametrize('radius_k', RADIUS_K_VALUES)
def test_crestline_output_comparison(name, spur, tip, radius_k):
    """
    Compares parameterized crestline outputs against baseline outputs.
    Saves output plots, tracks results, and asserts exact match.
    """
    make, _ = ALL_SENSITIVITY_CASES[name]
    mask = make()

    baseline_lines = crestline(mask, simplify=1.0)
    lines = crestline(
        mask,
        simplify=1.0,
        spur=spur,
        tip=tip,
        radius_k=radius_k
    )

    assert lines is not None
    assert baseline_lines is not None

    # Always generate and save the diagnostic plot first
    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_facecolor('#0b0b12')

    try:
        raster_bg = center_raster(mask)
        ax.imshow(np.ma.masked_where(~mask, raster_bg), cmap='coolwarm', vmin=0, vmax=1, extent=(0, w, h, 0))
    except Exception:
        ax.imshow(mask, cmap='gray', extent=(0, w, h, 0))

    for g in lines.geoms:
        ax.plot(*g.xy, 'o-', color='#00e5ff', lw=2, ms=3)

    ax.set_title(f'{name} (s={spur}, t={tip}, r={radius_k}): {len(lines.geoms)} lines', fontsize=7)

    output_dir = TEST_OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f'spur_{spur}_tip_{tip}_rk_{radius_k}.png'
    fig.savefig(output_dir / filename, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Check if output matches baseline
    passed = lines.equals_exact(baseline_lines, tolerance=1e-3)
    status = "PASS" if passed else "**FAIL**"
    rel_path = f"{name}/{filename}"

    # Record result for session report
    _TEST_RESULTS.append({
        'name': name,
        'spur': spur,
        'tip': tip,
        'radius_k': radius_k,
        'status': status,
        'n_lines': len(lines.geoms),
        'rel_path': rel_path
    })

    # Enforce the test criteria
    assert passed
