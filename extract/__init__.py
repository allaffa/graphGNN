"""ExaCA-centred graph extraction for GrainGNN.

grain_field        Field: the grain-id stack ids[k, y, x] with cell size, layer spacing, periodicity, orientations
exaca_reader       read_vtk (ExaCA GrainID VTK file), exaca_theta (orientation table -> theta_x, theta_z per grain)
cupf_reader        cuPF readers, used only to validate against the legacy extractor
periodic_grid      roll / wrap helpers and single-pass connected components on a grid periodic in x, y, both or neither
slice_cleanup      remove_islands, split_bad_corners: the minimal pixel edits made before extraction
boundary_tracing   boundary segments on corners, junction corners and clusters, traced edges, lens handling
network_types      Junction, Edge, Grain, SliceGraph
slice_graph        build_slice_graph: cleanup -> junctions -> refined positions -> grains -> validation (layer 1)
junction_matching  match_junctions: link the junctions of two consecutive slices (triple, then shared grains + distance)
trajectory         Trajectory: the per-run store in slot form, save/load, smooth_tracks (line fit along z)
stack_tracking     track_stack: layer 1 on every slice, slots, events (eliminate, two_sided, switch, appear) (layer 2)
"""
from .grain_field import Field
from .exaca_reader import read_vtk, exaca_theta
from .network_types import Junction, Edge, Grain, SliceGraph
from .slice_graph import build_slice_graph
from .junction_matching import match_junctions, Matching
from .trajectory import Trajectory, smooth_tracks
from .stack_tracking import track_stack
