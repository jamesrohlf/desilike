"""LINX -> Boltzmann coupling (BBNConsistentCosmology) on the linx-bbn PR branch, with CAMB and with CLASS."""
import sys
import time

import numpy as np
import cosmoprimo
from desilike.base import SumLikelihood, build, get_params
from desilike.parameter import Parameter, VariableCollection
from desilike.theories.bbn import linx_bbn_cosmology
from desilike.likelihoods.bbn import LINXBBNLikelihood
from desilike.likelihoods.bao import DESIDR2BAOLikelihood

BAO_DIR = '/global/homes/r/rohlf/linx_jr/data/desi_bao_dr2'
results = {}
for engine in ['camb', 'class']:
    print(f'\n========== engine = {engine} ==========', flush=True)
    params = VariableCollection([
        Parameter('omega_b', value=0.02200, prior=dict(limits=[0.019, 0.025])),
        Parameter('Omega_m', value=0.300, prior=dict(limits=[0.01, 0.99])),
        Parameter('h', value=0.680, prior=dict(limits=[0.2, 1.0])),
    ])
    N_eff = Parameter('N_eff', value=3.044, prior=dict(limits=[2., 4.5]))
    cosmo, bbn = linx_bbn_cosmology(params, engine=engine, fiducial=cosmoprimo.fiducial.DESI(engine=engine), N_eff=N_eff)
    bao = DESIDR2BAOLikelihood(cosmo=cosmo, data_dir=BAO_DIR)   # default fiducial: needs CLASS
    bbn_like = LINXBBNLikelihood(theory=bbn, measurements=('Aver2026', 'Cooke2018'))
    like = SumLikelihood(bao, bbn_like)
    varied = [p.name for p in get_params(like).select(varied=True)]
    print('varied:', [name for name in varied if not name.startswith('q_')],
          f'+ {sum(name.startswith("q_") for name in varied)} q_*', flush=True)
    pipe = build(like)
    d = {p.name: p._value for p in get_params(like)}
    points = {'defaults': d,
              'omega_b+0.0005': {**d, 'omega_b': d['omega_b'] + 0.0005},
              'N_eff=3.3': {**d, 'N_eff': 3.3}}
    results[engine] = {}
    for label, point in points.items():
        t = time.time()
        total = float(pipe(point))
        dt = time.time() - t
        built = cosmo._cosmo
        yhe_linx, neff_linx = float(bbn.YHe), float(bbn.N_eff_bbn)
        rs = built.get_thermodynamics().rs_drag
        # does the engine respond to YHe? rebuild the same cosmology with YHe shifted by +0.01
        rs_shift = built.clone(YHe=built['YHe'] + 0.01).get_thermodynamics().rs_drag
        ok_yhe = bool(np.isclose(built['YHe'], yhe_linx, rtol=1e-12))
        ok_neff = bool(np.isclose(built['N_eff'], neff_linx, rtol=1e-12))
        responds = abs(rs_shift - rs) > 1e-3
        print(f'[{label}] total logpdf {total:.4f}  ({dt:.1f} s)', flush=True)
        print(f'   LINX: YHe {yhe_linx:.6f}  N_eff_bbn {neff_linx:.5f}  D/H {float(bbn.D_H) * 1e5:.4f}e-5', flush=True)
        print(f'   {engine} inputs: YHe {built["YHe"]:.6f} (match {ok_yhe})  N_eff {built["N_eff"]:.5f} (match {ok_neff})', flush=True)
        print(f'   rs_drag {rs:.4f} Mpc; with YHe+0.01: {rs_shift:.4f} (d = {rs_shift - rs:+.4f}, responds {responds})', flush=True)
        results[engine][label] = dict(total=total, rs=rs, yhe=yhe_linx, dt=dt, ok=ok_yhe and ok_neff and responds)

print('\n========== CAMB vs CLASS ==========', flush=True)
for label in results['camb']:
    a, b = results['camb'][label], results['class'][label]
    print(f'[{label}] rs_drag rel diff {abs(a["rs"] / b["rs"] - 1):.1e}   '
          f'total logpdf camb {a["total"]:.4f} class {b["total"]:.4f} (diff {a["total"] - b["total"]:+.4f})   '
          f'LINX YHe identical: {a["yhe"] == b["yhe"]}', flush=True)
allok = all(r['ok'] for e in results.values() for r in e.values())
print('\nALL CHECKS PASSED' if allok else '\nSOME CHECKS FAILED', flush=True)
sys.exit(0 if allok else 1)
