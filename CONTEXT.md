# MEVIEW QA Viewer

A local toolset for auditing and inspecting MEVIEW and LFANN dense-mesh sequences.

## Language

**Mesh sequence**:
The numbered vertex, topology, and illustration assets for one dataset variant, subject, and video.
_Avoid_: dataset, mesh frame set

**Active window**:
The inclusive onset-to-offset frame interval supplied for a mesh sequence.
_Avoid_: range, timeline

**MediaPipe frame record**:
The normalized landmark result associated with one mesh sequence frame.
_Avoid_: landmark file, face result
