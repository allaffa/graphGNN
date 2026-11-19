# extract: GrainGNN training data from ExaCA

The `GrainID` field of an ExaCA run -> the sample pickle `train.py` reads. One command. Self-contained; the
training code is not needed.

## Install

```sh
pip install .              # from the repository root
```
Python >= 3.9, numpy, scipy, vtk. Without installing: run from the repository root.

## Use

One directory per run, each holding its `input.json` and the VTK it wrote:
```sh
python -m extract runs/run1 runs/run2 runs/run3 dataset_exaca.pkl
```
Several runs in one directory, one json each:
```sh
python -m extract input1.json input2.json input3.json dataset_exaca.pkl
```
Only the VTK files, no json: G (K/um), R (m/s) and seed per file, then one call to merge:
```sh
python -m extract Data1.vtk d1.pkl --G 1.9 --R 0.558 --seed 1 --orientations GrainOrientationVectors.csv
python -m extract Data2.vtk d2.pkl --G 5.0 --R 1.0   --seed 2 --orientations GrainOrientationVectors.csv
python -m extract d1.pkl d2.pkl dataset_exaca.pkl
```
Then:
```sh
python train.py --dataset exaca
```
`train.py --dataset NAME` reads `dataset_NAME.pkl` from the working directory. `graingnn-extract` is the same
command after `pip install`.

From a run directory or json: G = `TemperatureData.G` / 1e6, R = `TemperatureData.R` / `TemperatureData.G`,
seed = `RandomSeed`, the VTK = `Printing.PathToOutput/OutputFile.vtk`, the orientation table = `GrainOrientationFile`
when it is an absolute path or lies in the run directory, else `--orientations`.

The stacks are shuffled with a fixed seed into one list; `train.py` takes the tail as the validation split. On a
cluster, extract each run in its own job to its own pickle and merge the pickles with one call.

| option | default | |
|---|---|---|
| `--orientations CSV` | from the json | the orientation table of the ExaCA install, `share/ExaCA/GrainOrientationVectors.csv` |
| `--G`, `--R`, `--seed` | from the json | for a bare VTK file: K/um, m/s, integer |
| `--span N` | per stack | layers between a state and its labels; default: the largest span with fewer than 9 switches and 4.5 eliminations per window |
| `--smooth N` | 5 | window in layers of the line fit along z applied to junction tracks; `0` keeps the raw positions |
| `--z-range UM` | 48 | height that z and span are scaled by |
| `--jobs N` | 1 | stacks extracted in parallel |
| `--shuffle-seed N` | 35 | stack order in the dataset; `-1` keeps the given order |
| `--save-traj DIR` | | also the raw and painted junction trajectories per stack, npz |
| `--theta-x` | tilted | azimuth of the tilted `<100>` axis, or `lateral` for the in-plane axis |

## What ExaCA must provide

- Directional solidification, periodic in x and y.
- The final `GrainID` field written once: `Printing.Interlayer.Fields: ["GrainID"]`, ASCII or binary VTK.
  Time snapshots (`Printing.Intralayer`) are not used.
- The `input.json` of the run, and the orientation table the run used.
- [`examples/input.json`](examples/input.json): the campaign setup. 0.25 um cells, 40 x 40 x 50 um, no nucleation,
  `Substrate.Epsilon` 0.002 (cells per degree; keep `Epsilon x CellSize` fixed when changing the cell size).

ExaCA version notes (pipit branch, 07f2d5d). Three fixes were applied for the campaign: `acos(|z|)` in the capture
rule (CAinterface.hpp), substrate seed coordinates padded to local and the periodic period `nx - 2` (CAcelldata.hpp).
`UndercoolingStart` is written as zero everywhere (`fmin` where `fmax` is meant); it is not used here. Above 82.6 M
cells an int32 overflow aborts the run; use two or more MPI ranks.

## Output

A list of samples. A sample is one layer k with the labels of layer k + span, a plain namespace in the attribute
layout of `graph_datastruct.GrainHeterograph`:

| attribute | content |
|---|---|
| `feature_dicts` | `grain` (G, 10): x, y, z, area, cosx, sinx, cosz, sinz, span, darea history; `joint` (J, 8): x, y, z, G, R, span, dx, dy history |
| `target_dicts` | `grain` darea x20; `joint` dx, dy x5; `grain_event` eliminated; `edge_event` neighbour switch (1), kept (0), gone (-100); `edge` signed length change x2.5 |
| `mask` | `grain` has cells; `joint` same junction with the same grains at k and k + span; `edge` still joined |
| `edge_index_dicts`, `edge_weight_dicts` | grain -> joint and joint -> grain (distance to centroid), joint -> joint (edge length) |
| `features`, `targets`, `targets_scaling`, `edge_type`, `events` | names and scalings; the model sizes itself from them |
| `physical_params` | G, R, seed, height (layer index) |
| `vertex2joint`, `edges`, `gids`, `span` | junction triples, edge list, original grain ids, span in layers |

Units as in the paper: positions in the unit square, area as a fraction of the slice, z and span as fractions of
`--z-range`, G as 1 - G/10, R as R/2, displacements the shortest periodic difference.

## How it works

| stage | modules | |
|---|---|---|
| 1 slice | `slice_cleanup`, `boundary_tracing`, `slice_graph` | one layer -> junctions, boundaries, grains; islands repainted, degree-4 corners split, positions by line fit, Euler check |
| 2 track | `junction_matching`, `stack_tracking`, `trajectory` | junctions matched layer to layer by grain triple; events: eliminate, two-sided, enclosed, switch; tracks smoothed along z |
| 3 view | `graingnn_view` | what GrainGNN can represent: a grain is dead from its first layer with at most two junctions, its cells go to the neighbours |
| 3 samples | `samples`, `dataset` | states, labels, masks; span calibration; pickles |
| readers | `exaca_reader`, `grain_field`, `periodic_grid` | run directory, json, VTK; grids periodic in x, y, both or neither |

## Checks

- Every layer of the campaign stacks is a valid trivalent network (200/200 at 0.25 um). About 0.05% of cells are
  changed by the cleanup.
- 48 ms per 160 x 160 layer, about 10 s per 200-layer stack.

```sh
pip install ".[test]"
pytest tests/test_slice_cleanup.py tests/test_slice_graph.py tests/test_stack_tracking.py tests/test_samples.py tests/test_cli.py
```

Rollout evaluation against ExaCA, which needs the training code: `exaca_rollout_data.py` at the repository root.
