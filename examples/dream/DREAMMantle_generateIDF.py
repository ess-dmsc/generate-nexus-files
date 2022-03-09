import numpy as np
import os
import sys

"""
Reads the vertices and centers of the voxels from
an off-like file generated with another python script 
(not in this repository).
The info contained in the off-like file is based on 
GEANT4 results.
"""


def read_off(filename):
    """
    Reads input OFF to create the IDF
    Only tested with description of the mantle detectors only
    """
    with open(filename) as f:
        # first line [0] has only the word OFF
        lines = f.readlines()
        if lines[0].find('OFF') < 0:
            print('not an OFF file')
            return None, None
        # second line [1] has counts for ....
        counts = lines[2].split()
        vertex_count = int(counts[0])
        vox_count = int(counts[1])
        # then follows vertices from lines[3] to lines[3+vertex_count]
        vertices = np.asarray([float(s) for s in lines[3].split()])

        for line in lines[4:3 + vertex_count]:
            vertices = np.vstack(
                (
                    vertices,
                    np.asarray([float(s) for s in line.split()])
                )
            )
        # now extract the centers lines[2+vertex_count] to lines(-1)
        centers = np.asarray([float(s)
                              for s in lines[3 + vertex_count].split()])
        for line in lines[3 + vertex_count + 1:3 + vertex_count + vox_count]:
            if len(line) > 0:
                centers = np.vstack(
                    (
                        centers,
                        np.asarray([float(s) for s in line.split()])
                    )
                )
        return vertices, centers


class GenerateDREAMIDF(object):
    """
    Generates an Instrument Definition File (IDF) for the DREAM
    instrument (mantle detectors only)
    Written by Irina Stefanescu, ESS DG in January 2022.
    Inspired by the script GenerateCSPEC.py
    Debugged with the generous help of Celine Durniak, ESS DMSC.
    https://github.com/mantidproject/mantidgeometry/tree/main/Multigrid
    """
    _dist_det = 1.1  # metres
    _acceptance_angle_per_segment = 1.24211  # degrees
    _nb_segments_per_module = 6  # for mantle

    def __init__(self, filename, vertices, centers, num_mod):
        self.filename = filename
        self.vertices = vertices
        self.centers = centers
        self.fileHandle = None
        self.num_mod = int(num_mod)

    def _open_file(self):
        self.fileHandle = open(self.filename, "w")

    def _close_file(self):
        self.fileHandle.close()

    def _write_header_and_defaults(self):
        self.fileHandle.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        self.fileHandle.write(
            "<instrument xmlns=\"http://www.mantidproject.org/IDF/1.0\"\n"
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
            "xsi:schemaLocation=\"http://www.mantidproject.org/IDF/1.0/IDFSchema.xsd\"\n"
            "name=\"VoxelDetector\" valid-from   =\"2021-11-01 00:00:01\"\n"
            "valid-to     =\"2100-12-31 23:59:59\"\n"
            "last-modified=\"2021-11-05 12:00:00\">\n\n"
        )

        self.fileHandle.write("<defaults>\n")
        self.fileHandle.write("<length unit=\"meter\"/>\n")
        self.fileHandle.write("<angle unit=\"degree\"/>\n")
        self.fileHandle.write(
            "<location r=\"0.0\" t=\"0.0\" p=\"0.0\" ang=\"0.0\""
            " axis-x=\"0.0\" axis-y=\"0.0\" axis-z=\"1.0\"/>\n"
        )
        self.fileHandle.write("<reference-frame>\n")
        self.fileHandle.write(
            "<!-- The z-axis is set parallel to and in the direction of the "
            "beam. The y-axis points up and the coordinate system is right "
            "handed. -->\n"
        )
        self.fileHandle.write("<along-beam axis=\"z\"/>\n")
        self.fileHandle.write("<pointing-up axis=\"y\"/>\n")
        self.fileHandle.write("<handedness val=\"right\"/>\n")
        self.fileHandle.write("<origin val=\"beam\" />\n")
        self.fileHandle.write("</reference-frame>\n")
        self.fileHandle.write("<default-view view=\"cylindrical_y\"/>\n")
        self.fileHandle.write("</defaults>\n\n")

    def _write_footer(self):
        self.fileHandle.write("\n</instrument>")

    def _write_source(self):
        self.fileHandle.write("<!-- ***** SOURCE ***** -->\n")
        self.fileHandle.write("<component type=\"cold_source\">\n")
        self.fileHandle.write("\t<location z=\"-76.2\">")
        self.fileHandle.write("<facing val=\"none\"/>")
        self.fileHandle.write("</location>\n")
        self.fileHandle.write("</component>\n\n")

        self.fileHandle.write("<type name=\"cold_source\" is=\"Source\">\n")
        self.fileHandle.write("\t<properties />\n")
        self.fileHandle.write("\t<cylinder id=\"some-shape\">\n")
        self.fileHandle.write(
            "\t\t<centre-of-bottom-base r=\"0.0\" t=\"0.0\" p=\"0.0\" />\n")
        self.fileHandle.write("\t\t<axis x=\"0.0\" y=\"0.0\" z=\"1.0\" />\n")
        self.fileHandle.write("\t\t<radius val=\"0.01\" />\n")
        self.fileHandle.write("\t\t<height val=\"0.03\" />\n")
        self.fileHandle.write("\t</cylinder>\n")
        self.fileHandle.write("\t<algebra val=\"some-shape\" />\n")
        self.fileHandle.write("</type>\n\n")

    def _write_sample(self):
        self.fileHandle.write("<!-- ***** SAMPLE POSITION ***** -->\n")
        self.fileHandle.write("<component type=\"sample_position\">\n")
        self.fileHandle.write("\t<location>")
        self.fileHandle.write("<facing val=\"none\"/>")
        self.fileHandle.write("</location>\n")
        self.fileHandle.write("</component>\n\n")

        self.fileHandle.write(
            "<type name=\"sample_position\" is=\"SamplePos\">\n"
        )
        self.fileHandle.write("\t<properties />\n")
        self.fileHandle.write("\t<sphere id=\"some-shape\">\n")
        self.fileHandle.write(
            "\t\t<centre x=\"0.0\"  y=\"0.0\" z=\"0.0\" />\n"
        )
        self.fileHandle.write("\t\t<radius val=\"0.03\" />\n")
        self.fileHandle.write("\t</sphere>\n")
        self.fileHandle.write("\t<algebra val=\"some-shape\" />\n")
        self.fileHandle.write("</type>\n\n")

    def _write_mantle_module(self):
        self.fileHandle.write("<type name=\"Mantle\">\n")
        self.fileHandle.write(
            "\t<component type=\"Segment\" >\n")

        for i in range(self._nb_segments_per_module):
            seg_x = self._dist_det * \
                    np.cos(np.deg2rad(i * self._acceptance_angle_per_segment))
            seg_y = self._dist_det * \
                    np.sin(np.deg2rad(i * self._acceptance_angle_per_segment))
            ang_z = i * self._acceptance_angle_per_segment + 10
            self.fileHandle.write(
                "\t\t<location  x=\"" + str(seg_x) +
                "\" y=\"" + str(seg_y) +
                "\" z=\"0\" rot=\"" + str(ang_z) +
                "\" axis-x=\"0.0\" axis-y=\"0.0\" axis-z=\"1.0\""
                " name=\"Segment" +
                str(i + 1) + "\" />\n")

        self.fileHandle.write("\t</component>\n")
        self.fileHandle.write("</type>\n\n")

    def _write_segment(self):
        num_vert = 8
        shp = (self.vertices.size // 3)
        xi = np.zeros(shp, dtype=float, order='C')
        yi = np.zeros(shp, dtype=float, order='C')
        zi = np.zeros(shp, dtype=float, order='C')
        for i in range(self.vertices.size // 3 // num_vert):
            for j in range(num_vert):
                xi[num_vert * i + j] = \
                    round(self.vertices[num_vert * i + j, 0], 3)
                yi[num_vert * i + j] = \
                    round(self.vertices[num_vert * i + j, 1], 3)
                zi[num_vert * i + j] = \
                    round(self.vertices[num_vert * i + j, 2], 3)

        count_vox = 1

        for ii in range(0, self.vertices.size // 3, num_vert):
            name_vox = "Voxel" + str(count_vox)
            self.fileHandle.write(
                "<type name=\"" +
                name_vox +
                "\" is=\"detector\">\n"
            )
            count_vox += 1
            self.fileHandle.write("\t<hexahedron id=\"shape\">\n")

            self.fileHandle.write(
                "\t\t<left-back-bottom-point x=\"" + str(xi[ii + 4]) +
                "\" y=\"" + str(yi[ii + 4]) +
                "\" z=\"" + str(zi[ii + 4]) + "\"  />\n")
            self.fileHandle.write(
                "\t\t<left-front-bottom-point  x=\"" + str(xi[ii + 0]) +
                "\" y=\"" + str(yi[ii + 0]) +
                "\" z=\"" + str(zi[ii + 0]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<right-front-bottom-point  x=\"" + str(xi[ii + 1]) +
                "\" y=\"" + str(yi[ii + 1]) +
                "\" z=\"" + str(zi[ii + 1]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<right-back-bottom-point  x=\"" + str(xi[ii + 5]) +
                "\" y=\"" + str(yi[ii + 5]) +
                "\" z=\"" + str(zi[ii + 5]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<left-back-top-point x=\"" + str(xi[ii + 6]) +
                "\" y=\"" + str(yi[ii + 6]) +
                "\" z=\"" + str(zi[ii + 6]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<left-front-top-point  x=\"" + str(xi[ii + 2]) +
                "\" y=\"" + str(yi[ii + 2]) +
                "\" z=\"" + str(zi[ii + 2]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<right-front-top-point  x=\"" + str(xi[ii + 3]) +
                "\" y=\"" + str(yi[ii + 3]) +
                "\" z=\"" + str(zi[ii + 3]) + "\"  />\n"
            )
            self.fileHandle.write(
                "\t\t<right-back-top-point  x=\"" + str(xi[ii + 7]) +
                "\" y=\"" + str(yi[ii + 7]) +
                "\" z=\"" + str(zi[ii + 7]) + "\"  />\n"
            )
            self.fileHandle.write("\t</hexahedron>\n")
            self.fileHandle.write("\t<algebra val=\"shape\" />\n")

            self.fileHandle.write("\t<bounding-box>\n")
            self.fileHandle.write(
                "\t\t<x-min val=\"" + str(np.min(xi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write(
                "\t\t<x-max val=\"" + str(np.max(xi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write(
                "\t\t<y-min val=\"" + str(np.min(yi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write(
                "\t\t<y-max val=\"" + str(np.max(yi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write(
                "\t\t<z-min val=\"" + str(np.min(zi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write(
                "\t\t<z-max val=\"" + str(np.max(zi[ii:ii + 7])) + "\" />\n"
            )
            self.fileHandle.write("\t</bounding-box>\n")
            self.fileHandle.write("</type>\n\n")

        self.fileHandle.write("<type name=\"Segment\">\n")

        for jj in range(len(self.centers)):
            self.fileHandle.write(
                "\t<component type=\"Voxel" + str(jj + 1) + "\">\n"
            )
            self.fileHandle.write(
                "\t\t<location x=\"" + str(self.centers[jj, 0]) +
                "\" y=\"" + str(self.centers[jj, 1]) +
                "\" z=\"" + str(self.centers[jj, 2]) +
                "\" name=\"Voxel" + str(jj + 1) + "\" ></location>\n"
            )
            self.fileHandle.write("\t</component>\n")

        self.fileHandle.write("</type>\n\n")

    def _write_id_list(self, id_name, start, end):
        self.fileHandle.write(f"<idlist idname=\"{id_name}\">\n")
        self.fileHandle.write(
            "<id start=\"" + str(start) + "\" end=\"" + str(end) + "\" />\n"
        )
        self.fileHandle.write("</idlist>\n")

    def _write_all_components(self):
        for i in range(self.num_mod):
            name = f"_module_{i + 1}"
            init_angle = 0.  # in degrees
            ang_z = i * self._nb_segments_per_module * self._acceptance_angle_per_segment + \
                    init_angle
            pos_x = np.cos(np.deg2rad(ang_z))
            pos_y = np.sin(np.deg2rad(ang_z))

            self.fileHandle.write(
                f"<component type=\"Mantle\""
                f" idlist=\"{name}\" name=\"{name}\">\n"
            )
            self.fileHandle.write(
                "\t<location x=\"" + str(pos_x) +
                "\" y=\"" + str(pos_y) +
                "\" z=\"0\" rot=\"" + str(ang_z) +
                "\" axis-x=\"0.0\" axis-y=\"0.0\" axis-z=\"1.0\"></location>\n"
            )
            self.fileHandle.write("</component>\n")
        self.fileHandle.write("\n")

    def _write_all_ids(self):
        #  no_seg * no_counters * no_wires * no_strips
        id_span = self._nb_segments_per_module * 2 * 32 * 256

        for i in range(self.num_mod):
            start = i * id_span
            end = start + id_span - 1
            self._write_id_list(f"_module_{i + 1}", start, end)

    def generate(self):
        """
        Group all parts required in the IDF
        """
        self._open_file()
        # copied from GenerateCSPEC.py
        self._write_header_and_defaults()
        self._write_source()
        self._write_sample()

        self._write_all_components()
        self._write_mantle_module()
        self._write_segment()
        self._write_all_ids()
        self._write_footer()
        self._close_file()


if __name__ == "__main__":
    # Usage: python DREAMMantle_generateIDF.py DREAM_Mantel1seg.off
    off_file = sys.argv[1]
    assert os.path.exists(off_file), 'The input file could not be found'

    vertx, centx = read_off(off_file)
    gen = GenerateDREAMIDF("dream.xml", vertx, centx, 10)
    gen.generate()
