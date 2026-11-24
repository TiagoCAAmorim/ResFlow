# Well layout optimisation

How `generate_well_layouts.py` turns a permeability patch into well positions and roles, with no reservoir
simulation. Code: `well_layout.py` (counts, roles, one patch end to end) and `well_placement/` (the optimiser). Options are listed in docs/02_build_decks.md.

## Overview

For one patch and a list of requested well counts (P producers / I injectors):

1. **Index map.** Compute `Y_at` for every cell: how much oil volume a well there would drain, discounted by distance.
2. **Room and counts.** Find how many wells fit; revise the counts that do not.
3. **Annealing.** For each count, place `P + I` wells as if all were producers, maximising a score built from `Y_at`.
4. **Validity.** Reject layouts with wells on tight rock, isolated wells, or a very weak well. Retry with another seed;
   skip the count if no attempt is valid.
5. **Roles.** Choose which of the optimised positions become injectors.

The annealing never sees injectors. It finds good *well positions*; roles are assigned afterwards from the positions.

## 1. The index map `Y_at`

*Equivalent distance.* Each cell is stretched by `alpha = sqrt((POR/0.10) * (100 / k))` relative to a reference rock
(POR 0.10, k 100 mD), per direction. The distance between two cells is the shortest path over the grid (I/J
neighbour connections only, no diagonals) with those stretched lengths. Tight, porous rock is "far"; permeable rock is "close". On
the reference rock, equivalent distance equals metres.

*Discounted volume.* With pore volume `V` per cell and `decay = ln(1 + rate) / x_base`:

```
Y_at(i) = sum over cells j of  V_j * exp(-decay * d_eq(i, j))
```

Volume far from a well in equivalent distance counts less, like a discounted cash flow (`--rate`, `--x-base`). `Y_at`
is a property of a cell: it ranks where a single well would do best. It is computed once per patch.

## 2. Room and well counts

`max_wells_fit` places wells greedily on the best `Y_at` cells that are at least `--d-min` apart and counts how many
fit (the room). With `--k-min`, only cells at or above `k_min` are used.

A requested count (P, I) with `P + I` above the room is revised by removing wells one at a time, a producer first, then
an injector, and so on, down to 1P1I. Injectors only go below 2 when there are 2 producers:

```
4P2I -> 3P2I -> 2P2I -> 2P1I -> 1P1I          (never 4P1I)
```

The first revised count that fits is used. It is skipped if it equals another requested count or one already made for
the patch, so a patch never has two layouts with the same well count. Skips and reasons go to `skipped_cases.csv`.

## 3. Annealing

All `n = P + I` wells are positions to find. A well's index is `Y_at(well)`: the sum of `V * exp(-decay * d_eq)` over
the cells that are nearest (in equivalent distance) to that well. The score of a set of wells is

```
score = total * (1 - penalty)
total   = sum of the wells' Y_at
penalty = w_bal * CV(Y_at per well)                          wells should drain similar volumes
        + w_dist * sum max(0, (d_min - d)/d_min)^2           wells should not be closer than d_min
```

`CV` is the standard deviation divided by the mean. Defaults: `w_bal` 0.25, `w_dist` 1.

**Search.** A greedy start (best `Y_at` cells respecting `d_min`), then simulated annealing for `--n-iter` moves. A
move relocates one well, usually to a cell within a radius that shrinks as the run cools, sometimes to a random cell
(10%). The start temperature comes from the spread of 50 random layouts and falls geometrically to 1e-3 of it. A
final 1-opt sweep tries every cell within three cell widths of each well and keeps improvements. The seed is derived
from `--seed` and the patch id, so runs are reproducible.

**Tight rock and isolation (`--k-min` > 0).** Cells below `k_min` are treated as barriers; inactive cells too.
Communication is only through I and J neighbours, never diagonals, and never between layers. The connected groups of
cells are labelled once before the annealing, so checking a well is a lookup. During annealing:

- a well on a cell below `k_min` has `Y_at = 0`;
- a well with no other well in its group (isolated) has `Y_at = 0`; this is not applied with a single well.

A zeroed well lowers the total and raises the imbalance, so the search moves away from it. As a search aid, the start
and the moves use only cells at or above `k_min`.

## 4. Validity and retries

After the annealing (and the polish) a layout is **invalid** if any of these holds:

| Check | Rule |
|---|---|
| Tight rock | a well is on a cell below `--k-min` |
| Isolation | a well shares no group with any other well (more than one well) |
| Weak well | a well's `Y_at` is below `--score-min-frac` x the mean `Y_at` of the *other* wells |
| Group capacity | the wells sit in more groups than the injectors, or the producers, can serve |

Only groups that contain wells count; a patch may have more groups than that. An invalid layout is re-annealed with
`seed + 1`, then `seed + 2` (`--max-attempts`, default 3). If none is valid, that well count is skipped with the
reason in `skipped_cases.csv`. All the checks are off by default (`--k-min 0`, `--score-min-frac 0`); the
fluvial (Upper Ness) layouts used `--k-min 10 --score-min-frac 0.2`.

## 5. Roles

Roles are assigned to the optimised positions, using each well's `Y_at` (low `Y_at` = poor producer = injector
candidate).

1. **One injector per group.** Every group that has wells gets an injector first, at its lowest-`Y_at` position. A
   group always keeps at least one producer. Without `--k-min` all wells form one group.
2. **Remaining injectors.** Each goes to the group with the lowest ratio *injectors / producers* (ties: the larger
   group), among groups that still have two or more producers-to-be. Inside that group it takes the lowest-`Y_at`
   position that
   - is at least `--inj-factor` x `--d-min` from every injector already chosen, and
   - is not **behind** an injector: no injector of the group is nearer to the position than some producer of the
     group while lying within 30 degrees (`SHADOW_DEG`) of the line to that producer. Such an injector would push its
     water ahead of the new one, so the new injector adds little sweep.
3. **Fallback.** If no position meets both rules, the distance rule is dropped, then the shadow rule. These are
   flagged in `inj_spacing_relaxed` and `inj_behind_injector` in `well_locations.csv`.
4. The other positions are producers; P1 has the highest `Y_at`.

## Outputs

`well_locations.csv` has one row per case: producer and injector `(I, J)`, `reduced` (revised count), `score`, the two
relaxation counters. `skipped_cases.csv` lists every count not produced and why. `config_used.yaml` records the options.

## Limits

- Injectors are not part of the score: positions are optimised for drainage, then roles are picked from them.
- Groups, isolation and the barrier rule use only the `k_min` threshold, not the flow capacity of the path.
- The shadow angle (30 degrees) and the 10% random-jump rate are judgement values, not calibrated against simulation.
