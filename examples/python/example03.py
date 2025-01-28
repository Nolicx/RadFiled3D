from RadFiled3D.RadFiled3D import FieldStore, RadiationFieldMetadataV1, HistogramVoxel
from plotly import graph_objects as go
import numpy
import sys


# This file demonstrates how to load a radiation field files metadata and which information can be extracted from it.

radiation_file = sys.argv[1]

metadata: RadiationFieldMetadataV1 = FieldStore.load_metadata(radiation_file)
energy_eV = metadata.get_header().simulation.tube.max_energy_eV
print(f"Energy: {energy_eV / 1e3} keV")
direction = metadata.get_header().simulation.tube.radiation_direction
print(f"Direction: {direction}")
origin = metadata.get_header().simulation.tube.radiation_origin
print(f"Origin: {origin}")
geometry = metadata.get_header().simulation.geometry
print(f"Geometry: {geometry}")
particles = metadata.get_header().simulation.primary_particle_count
print(f"Particles: {particles}")
dyn_data_keys = metadata.get_dynamic_metadata_keys()
print(f"Dynamic metadata keys: {dyn_data_keys}")
simulation_duration_s = metadata.get_dynamic_metadata("simulation_duration_s").get_data()
print(f"Simulation duration: {simulation_duration_s} s")
tube_spectrum: HistogramVoxel = metadata.get_dynamic_metadata("tube_spectrum")
print(f"Tube spectrum: {tube_spectrum.get_histogram()}")


# plot tube spectrum
fig = go.Figure(data=go.Scatter(x=numpy.arange(0, (tube_spectrum.get_bins() - 1) * tube_spectrum.get_histogram_bin_width(), tube_spectrum.get_histogram_bin_width()) - tube_spectrum.get_histogram_bin_width(), y=tube_spectrum.get_histogram()))
fig.show()
