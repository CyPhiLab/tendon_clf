"""Markdown tables from evaluate.py results."""
import sys, json
import numpy as np

FIT = ['20261103_174014', '20261103_191933']


def slip_stats(d):
    n = len(d['pairs'])
    matched = [(a, b) for a, b in d['pairs'] if b]
    within = sum(abs(b['cur'] - a['cur']) <= 0.03 for a, b in matched)
    don = [abs(b['cur'] - a['cur']) for a, b in matched]
    dsz = [abs(b['size'] - a['size']) for a, b in matched]
    return n, len(matched), within, don, dsz, len(d['spur'])


def main(path, names):
    res = json.load(open(path))
    runs = [r for r in res[names[0]] if not r.startswith('_')]
    val = [r for r in runs if r not in FIT]
    print('| option | 174014 | 191933 | fit mean | validation mean (5 runs) | slips matched (of 13) | onset within 0.03 A | mean onset error [A] | mean size error [rad] | spurious slips | max spring-back [rad] | cost [ms per sim s] |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in names:
        R = res[n]
        mk = {r: 1e3 * R[r]['marker_rms'] for r in runs}
        tot = [0, 0, 0, [], [], 0]
        sb = 0.0
        for r in FIT:
            s = slip_stats(R[r])
            for i in (0, 1, 2, 5):
                tot[i] += s[i]
            tot[3] += s[3]; tot[4] += s[4]
            sb = max(sb, max(R[r]['spring_sim']))
        on = f'{np.mean(tot[3]):.3f}' if tot[3] else '-'
        sz = f'{np.mean(tot[4]):.2f}' if tot[4] else '-'
        print(f"| {n} | {mk[FIT[0]]:.1f} | {mk[FIT[1]]:.1f} | {np.mean([mk[r] for r in FIT]):.1f} | "
              f"{np.mean([mk[r] for r in val]):.1f} | {tot[1]} | {tot[2]} | {on} | {sz} | {tot[5]} | {sb:.3f} | "
              f"{1e3 * R['_cost']:.0f} |")
    rec_sb = max(max(res[names[0]][r]['spring_rec']) for r in FIT)
    print(f'\nrecorded max spring-back: {rec_sb:.3f} rad')
    print('\nValidation runs, marker rms [mm]:\n')
    print('| option | ' + ' | '.join(r.replace('20261103_target_', '') for r in val) + ' |')
    print('|---|' + '---|' * len(val))
    for n in names:
        print(f'| {n} | ' + ' | '.join(f"{1e3 * res[n][r]['marker_rms']:.1f}" for r in val) + ' |')
    for r in FIT:
        print(f'\nPer phase, {r} [mm]:\n')
        phases = [p for p, _ in res[names[0]][r]['phase']]
        print('| phase | ' + ' | '.join(names) + ' |')
        print('|---|' + '---|' * len(names))
        for i, p in enumerate(phases):
            row = []
            for n in names:
                d = dict(res[n][r]['phase'])
                row.append(f'{1e3 * d[p]:.0f}' if p in d else '-')
            print(f'| {p} | ' + ' | '.join(row) + ' |')
    for n in names[-2:]:
        print(f'\nSlips, option {n} (recorded vs simulated; onset current [A] / size [rad]):\n')
        print('| run | t [s] | motor | recorded | simulated |')
        print('|---|---|---|---|---|')
        for r in FIT:
            for a, b in res[n][r]['pairs']:
                sim = f"{b['cur']:+.3f} / {b['size']:+.2f} (t={b['t']:.1f})" if b else 'none'
                print(f"| {r[-6:]} | {a['t']:.1f} | {a['motor']} | {a['cur']:+.3f} / {a['size']:+.2f} | {sim} |")
            for b in res[n][r]['spur']:
                print(f"| {r[-6:]} | {b['t']:.1f} | {b['motor']} | none | {b['cur']:+.3f} / {b['size']:+.2f} (spurious) |")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','))
