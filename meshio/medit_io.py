# -*- coding: utf-8 -*-
#
"""
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
for something like a specification.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
"""
import re
import logging
import numpy

from .mesh import Mesh


def read(filename):
    with open(filename) as f:
        points, cells, point_data, cell_data = read_buffer(f)

    return Mesh(points, cells, point_data, cell_data)


class _ItemReader:
    def __init__(self, file, delimiter=r"\s+"):
        # Items can be separated by any whitespace, including new lines.
        self._re_delimiter = re.compile(delimiter, re.MULTILINE)
        self._file = file
        self._line = []
        self._line_ptr = 0

    def next_items(self, n):
        """Returns the next n items.

        Throws StopIteration when there is not enough data to return n items.
        """
        items = []
        while len(items) < n:
            if self._line_ptr >= len(self._line):
                # Load the next line.
                line = next(self._file).strip()
                # Skip all comment and empty lines.
                while not line or line[0] == "#":
                    line = next(self._file).strip()
                self._line = self._re_delimiter.split(line)
                self._line_ptr = 0
            n_read = min(n - len(items), len(self._line) - self._line_ptr)
            items.extend(self._line[self._line_ptr : self._line_ptr + n_read])
            self._line_ptr += n_read
        return items

    def next_item(self):
        return self.next_items(1)[0]


def read_buffer(file):
    dim = 0
    cells = {}
    point_data = {}
    cell_data = {}

    reader = _ItemReader(file)

    meshio_from_medit = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Hexahedra": ("hexahedra", 8),
    }

    # key = keyword, value = number of values per entry
    ignored_fields = {
        "Corners": 1,
        "RequiredVertices": 1,
        "Ridges": 1,
        "RequiredEdges": 1,
        "Normals": 3,
        "Tangents": 3,
        "NormalAtVertices": 2,
        "NormalAtTriangleVertices": 3,
        "NormalAtQuadrilateralVertices": 3,
        "TangentAtEdges": 3,
    }

    while True:
        try:
            keyword = reader.next_item()
        except StopIteration:
            break

        assert keyword.isalpha()

        meshio_from_medit = {
            "Edges": ("line", 2),
            "Triangles": ("triangle", 3),
            "Quadrilaterals": ("quad", 4),
            "Tetrahedra": ("tetra", 4),
            "Hexahedra": ("hexahedra", 8),
        }

        if keyword == "MeshVersionFormatted":
            assert reader.next_item() == "1"
        elif keyword == "Dimension":
            dim = int(reader.next_item())
        elif keyword == "Vertices":
            assert dim > 0
            # The first value is the number of nodes
            num_verts = int(reader.next_item())
            points = numpy.empty((num_verts, dim), dtype=float)
            point_data[0] = numpy.empty(num_verts, dtype=int)
            for k in range(num_verts):
                # Read point and point data.
                data = numpy.array(reader.next_items(dim + 1), dtype=float)
                points[k] = data[:-1]
                point_data[0][k] = data[-1]
        elif keyword in meshio_from_medit:
            meshio_name, num = meshio_from_medit[keyword]
            # The first value is the number of elements
            num_cells = int(reader.next_item())
            cells[meshio_name] = numpy.empty((num_cells, num), dtype=int)
            cell_data[meshio_name] = {0: numpy.empty(num_cells, dtype=int)}
            for k in range(num_cells):
                data = numpy.array(reader.next_items(num + 1), dtype=int)
                # Store cell values and cell labels.
                cells[meshio_name][k] = data[:-1]
                cell_data[meshio_name][0][k] = data[-1]

            # adapt 0-base
            cells[meshio_name] -= 1
        elif keyword in ignored_fields:
            print("Warning: Field {} currently ignored by meshio.".format(keyword))
            # Skip the values.
            n_values = int(reader.next_item()) * ignored_fields[keyword]
            reader.next_items(n_values)
        else:
            assert keyword == "End", "Unknown keyword '{}'.".format(keyword)

    return points, cells, point_data, cell_data


def write(filename, mesh):
    with open(filename, "wb") as fh:
        fh.write(b"MeshVersionFormatted 1\n")
        fh.write(b"# Created by meshio\n")

        n, d = mesh.points.shape

        # Dimension info
        dim = "\nDimension {}\n".format(d)
        fh.write(dim.encode("utf-8"))

        # vertices
        fh.write(b"\nVertices\n")
        fh.write("{}\n".format(n).encode("utf-8"))
        # vertex labeles
        labels = numpy.ones(n, dtype=int)
        if mesh.point_data:
            if 0 in mesh.point_data:
                labels = mesh.point_data[0]
            elif mesh.point_data.keys():
                assert len(mesh.point_data) == 1, "Only 1-D labels supported."
                labels = list(mesh.point_data.values())[0]
        data = numpy.c_[mesh.points, labels]
        fmt = " ".join(["%r"] * d) + " %d"
        numpy.savetxt(fh, data, fmt)

        medit_from_meshio = {
            "line": ("Edges", 2),
            "triangle": ("Triangles", 3),
            "quad": ("Quadrilaterals", 4),
            "tetra": ("Tetrahedra", 4),
            "hexahedra": ("Hexahedra", 8),
        }

        for key, data in mesh.cells.items():
            try:
                medit_name, num = medit_from_meshio[key]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    key
                )
                logging.warning(msg)
                continue

            fh.write(b"\n")
            fh.write("{}\n".format(medit_name).encode("utf-8"))
            fh.write("{}\n".format(len(data)).encode("utf-8"))

            # cell data
            labels = numpy.ones(len(data), dtype=int)
            if mesh.cell_data:
                if 0 in mesh.cell_data[key]:
                    labels = mesh.cell_data[key][0]
                else:
                    assert len(mesh.cell_data[key]) == 1, "Only 1-D labels supported."
                    labels = list(mesh.cell_data[key].values())[0]

            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = " ".join(["%d"] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b"\nEnd\n")

    return
