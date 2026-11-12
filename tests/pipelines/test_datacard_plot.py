"""mass.pdf: the background curve must show the fitted yield nbkg in the full mass range.

Bug (before the fix): expo.plotOn(frame) scaled the curve to the events of the last plotted dataset. Blinded data
has only the sidebands, so the curve was ~30% too low (sideband events spread over the full range).
The three cases: blinded sideband fit (the analysis now), full-range fit, unblinded data with a signal and a
sideband fit. In all three the curve must hold nbkg events.
"""
import pytest

ROOT = pytest.importorskip("ROOT")

from tau3mu_gnn.analysis import datacard as dc  # noqa: E402

LO, HI, TAU, WIN = 1.6, 2.0, 1.78, 0.060
N_BKG, N_SIG = 20000, 2000


def curve_events(frame, name="bkg_curve"):
    """Events under the curve in [LO, HI]: the frame is in events per bin of dc.NBINS bins."""
    curve = frame.getCurve(name)
    assert curve, f"no curve {name!r} on the frame"
    return curve.average(LO, HI) * dc.NBINS


def setup(seed=1986):
    """Variables and pdfs as in make_datacard, plus a toy background sample (exponential, slope -2)."""
    ROOT.RooRandom.randomGenerator().SetSeed(seed)
    mass = ROOT.RooRealVar("mass", "3-#mu mass", LO, HI, "GeV")
    mass.setRange("left", LO, TAU - WIN)
    mass.setRange("right", TAU + WIN, HI)
    slope = ROOT.RooRealVar("slope", "slope", *dc.SLOPE_START)
    expo = ROOT.RooExponential("bkg_expo", "bkg_expo", mass, slope)
    nbkg = ROOT.RooRealVar("nbkg", "nbkg", *dc.NBKG_START)
    model = ROOT.RooAddPdf("bkg_extended_expo", "bkg_extended_expo", ROOT.RooArgList(expo), ROOT.RooArgList(nbkg))
    true_slope = ROOT.RooRealVar("true_slope", "true_slope", -2.0)
    truth = ROOT.RooExponential("truth", "truth", mass, true_slope)
    bkg = truth.generate(ROOT.RooArgSet(mass), N_BKG)
    return mass, expo, nbkg, model, bkg, [slope, true_slope, truth]  # keep the RooFit objects alive


def fit_and_draw(mass, expo, nbkg, model, data, fit_range=None):
    frame = mass.frame()
    data.plotOn(frame, ROOT.RooFit.Binning(dc.NBINS))
    opts = [ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1)]
    if fit_range:
        opts.append(ROOT.RooFit.Range(fit_range))
    model.fitTo(data, *opts)
    dc.draw_background(frame, expo, nbkg)
    return frame


def test_blinded_sideband_fit_curve_holds_nbkg():
    """The analysis now: only sidebands in the data, fit on the sidebands."""
    mass, expo, nbkg, model, bkg, _keep = setup()
    sidebands = bkg.reduce(f"abs(mass - {TAU}) > {WIN}")
    frame = fit_and_draw(mass, expo, nbkg, model, sidebands, "left,right")

    n_curve = curve_events(frame)
    assert n_curve == pytest.approx(nbkg.getVal(), rel=1e-3)
    assert n_curve == pytest.approx(N_BKG, rel=0.03)  # nbkg extrapolates the sidebands to the full range
    # the old curve held only the sideband events: make sure the test tells the two apart
    assert n_curve > 1.2 * sidebands.sumEntries()


def test_full_range_fit_curve_holds_all_events():
    """Fit over the full range: nbkg = number of events, so the curve holds all of them (the old line did too)."""
    mass, expo, nbkg, model, bkg, _keep = setup()
    frame = fit_and_draw(mass, expo, nbkg, model, bkg)

    n_curve = curve_events(frame)
    assert nbkg.getVal() == pytest.approx(bkg.sumEntries(), rel=1e-3)
    assert n_curve == pytest.approx(nbkg.getVal(), rel=1e-3)


def test_unblinded_with_signal_curve_shows_background_only():
    """Unblinded data with a signal peak, fit on the sidebands: the curve is the background predicted by the
    sidebands, so the signal stays visible above it (the old line was scaled to all events, signal included)."""
    mass, expo, nbkg, model, bkg, keep = setup()
    mean = ROOT.RooRealVar("mean", "mean", TAU)
    width = ROOT.RooRealVar("width", "width", 0.02)
    gaus = ROOT.RooGaussian("sig", "sig", mass, mean, width)
    data = bkg.Clone("data")
    data.append(gaus.generate(ROOT.RooArgSet(mass), N_SIG))
    frame = fit_and_draw(mass, expo, nbkg, model, data, "left,right")

    n_curve = curve_events(frame)
    assert n_curve == pytest.approx(nbkg.getVal(), rel=1e-3)
    assert n_curve == pytest.approx(N_BKG, rel=0.03)
    assert n_curve < data.sumEntries() - 0.5 * N_SIG
    keep += [mean, width, gaus, data]


def test_make_datacard_draws_background_with_nbkg(tmp_path, monkeypatch):
    """make_datacard uses draw_background, and the curve holds the yield written to the datacard."""
    from test_limit import toy_stage3_file

    sig = toy_stage3_file(tmp_path / "in" / "signal_with_gnn_scores.root", 4000, True, 1)
    bkg = toy_stage3_file(tmp_path / "in" / "background_with_gnn_scores.root", 40000, False, 2)
    seen = {}
    original = dc.draw_background

    def spy(frame, pdf, nbkg, name="bkg_curve"):
        original(frame, pdf, nbkg, name)
        seen["curve"] = curve_events(frame, name)
        seen["nbkg"] = nbkg.getVal()

    monkeypatch.setattr(dc, "draw_background", spy)
    res = dc.make_datacard([sig], [bkg], 3.9, 0.5, tmp_path / "card")

    assert seen, "make_datacard did not call draw_background"
    assert seen["curve"] == pytest.approx(seen["nbkg"], rel=1e-3)
    assert seen["nbkg"] == pytest.approx(res.nbkg, rel=1e-6)
