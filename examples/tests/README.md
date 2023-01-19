

# Validating instrument geometry
The idea behind this collection of unit tests is to programmatically
validate the instrument geometry.

The foundation for this is the mapping between pixel values (logical) and
pixel positions (physical).

See detector_geometry_from_json.py in utils/

For a lot of the validation we do not need to know the absolute position of the
pixels. Most of the time we can test relative positions instead.

Generally the linkage between logical and physical geometry requires information
from
  * ECDC ICDs where the logical geometry is specified.
  * Technical drawings, Excel sheets, PowerPoint slides describing physical
  distances, offsets, translations and rotations of the individual pixels (or voxels).

The idea is that from one or more pixel values we can calculate points or
vectors for important detector directions and then compare these with the
expected values according to the technical drawings to within a specified precision.

Generally we should aim for a precision of 1/1000 mm which very likely puts on
the safe side by a large margin. In many cases a much higher precision will be
achievable.

## Basic validations
The following is a catalogue of testing ideas

### Min and max pixels
  * Test that all pixel values specified in the ICD can be queried for (x, y, z) values
  * Test that max_pixel + 1 is invalid

### Basic sanity checks
  * Test that all positions are within the physical boundary of the instrument
  * Test that all z, y, z values are positive (if applicable)
  * Test relative positions: for example that the distance of all pixels in 'bank0'
for LoKI are at a greater z distance than all other banks.

### Intra-detector tests
These are tests within a small scope. Typically, bounded electronically. 
Example are blades, tubes and straws.
  * Check distances between adjacent pixels (or adjacent wires, strips, straw, ...)
  * Check relative positions: back straws have larger z than front straws, etc.

### Inter-detector tests
This scope compares distances (and angles, etc.) between electrical components. 
For example the distances between tubes in a detector panel or between cassettes.
