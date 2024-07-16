import dolfin as df
import ufl
import meshio
import numpy as np
import logging
import os
import shlex
from subprocess import check_output
from pathlib import Path

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes
class Geometry:
    """
    defines the geometry of a problem

    *Attributes:*
        domain: geometrical domains
        mesh: generated mesh from xdmf format
        dim: dimension of problem
        facets: geometrical faces
        d_bound: List of Dirichlet boundaries
        n_bound: List of Neumann boundaries
    """
    def __init__(self):
        self.mesh = None
        self.element = None
        self.facet_function = None
        self.domain = None
        self.function_space = None
        self.ansatz_function = None
        self.test_function = None
        self.dx = None
        self.d_bound = None
        self.n_bound = None

class SolverParam:
    """
    contains all solver parameters, that can be set. Need for the sake of clean code
    """
    def __init__(self):
        self.solver_type = None
        self.maxIter = None
        self.rel = None
        self.abs = None

class General:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    pass

class External:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    pass

class FEM:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    def __init__(self):
        self.solver_param = SolverParam()
    pass

class Material:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    pass

class Time:
    """
    This is a dummy class to make a clustering of attributes possible
    """
    pass

class Parameters:
    """
    Parameters describing the problem are clustered in this class.

    *Attributes*:
        gen:        General parameters, such as titles or flags and switches
        time:       Time-dependent parameters
        mat:        Material parameters
        init:       Initial parameters
        fem:        Parameters related to finite element method (fem)
        add:        Parameters of addititives, in adaptive base models the user can add arbitrary additive components
        ext:        External paramters, such as external loads
    """
    def __init__(self):
        self.gen = General()
        self.time = Time()
        self.mat = Material()
        self.fem = FEM()

class Problem:
    """
    defines a Problem that describes the geometry, boundary and parameters. Should be super class for

    *Prototypes:*
        general:
        geometry: geometrical description
        parameters: all describing parameters
    """
    def __init__(self):
        self.param = Parameters()
        self.geom = Geometry()

# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
def mkdir_if_not_exist(dir):
    """
    Makes directory if not exists and returns the string

    *Arguments*:
        dir: String

    *Example*:
        dir = mkdir_if_not_exist(dir)
    """
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)
    return dir

def create_Quarter_Circle(esize: float, fac: float, rad1: float, rad2: float,
                          lay: int, dfile: str, struc_mesh=True) -> str:
    """
    creates a 2D quarter circle with an inner area and three outer boundary conditions. Gives Path
    IDs:
        1:= Inner rim
        2:= Outer rim
        3:= Bottom facet
        4:= Left facet
        5:= Circumfence
    """
    output = add_file_appendix(dfile, "geo")
    ele_size = esize * rad1
    with open(output, 'w') as f:
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write("Point(1) = {0, 0, 0, " + str(ele_size) + "};\n")
        f.write("Point(2) = {" + str(rad1) + ", 0, 0, " + str(ele_size * fac * rad1 / rad2) + "};\n")
        f.write("Point(3) = {" + str(rad2) + ", 0, 0, " + str(ele_size * fac) + "};\n")
        if struc_mesh:
            f.write("Line(1) = {1, 2};\n")
            f.write("Line(2) = {2, 3};\n")
            f.write("Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {\n")
            f.write("  Curve{1}; Curve{2}; Layers{" + str(lay) + "};\n")
            f.write("}\n")
        else:
            f.write("Point(4) = {0, " + str(rad1) + ", 0, " + str(ele_size * fac * rad1 / rad2) + "};\n")
            f.write("Point(5) = {0, " + str(rad2) + ", 0, " + str(ele_size * fac) + "};\n")
            f.write("Line(1) = {1, 2};\n")
            f.write("Line(2) = {2, 3};\n")
            f.write("Line(3) = {5, 4};\n")
            f.write("Line(4) = {4, 1};\n")
            f.write("Circle(3) = {2, 1, 4};\n")
            f.write("Circle(4) = {3, 1, 5};\n")
            f.write("Curve Loop(1) = {2, 1, 4};\n")
            f.write("Curve Loop(2) = {3, 1, 5};\n")
            f.write("Surface(1) = {1};\n")
            f.write("Surface(2) = {2};\n")
        f.write("Physical Surface(\"4\") = {2};\n")
        f.write("Physical Surface(\"5\") = {1};\n")
        f.write("Physical Curve(\"1\") = {1, 2};\n")
        f.write("Physical Curve(\"2\") = {4, 6};\n")
        f.write("Physical Curve(\"3\") = {5};\n")
    done = run_shell_command("gmsh -2 " + output)
    msh2xdmf(dfile, dfile + os.sep, correct_gmsh=True)
    return dfile + os.sep

def create_Rectangle(esize_height: float, esize_width: float, height1: float,
                     height2: float, width: float, dfile: str) -> str:
    """
        creates a 2D quarter circle with an inner area and three outer boundary conditions. Gives Path
        IDs:
            1:= Inner rim
            2:= Outer rim
            3:= Bottom facet
            4:= Left facet
            5:= Circumfence
        """
    output = add_file_appendix(dfile, "geo")
    with open(output, 'w') as f:
        f.write("SetFactory(\"OpenCASCADE\");\n")
        f.write("Point(1) = {0, 0, 0, " + str(esize_width) + "};\n")
        f.write("Point(2) = {" + str(width) + ", 0, 0, " + str(esize_width) + "};\n")
        f.write("Point(3) = {" + str(width) + ", " + str(height1) + ", 0, " + str(esize_height) + "};\n")
        f.write("Point(4) = {0, " + str(height1) + ", 0, " + str(esize_height) + "};\n")
        f.write("Point(5) = {0, " + str(height1+height2) + ", 0, " + str(esize_height) + "};\n")
        f.write("Point(6) = {" + str(width) + ", " + str(height1+height2) + ", 0, " + str(esize_height) + "};\n")
        f.write("Line(1) = {1, 2};\n")
        f.write("Line(2) = {2, 3};\n")
        f.write("Line(3) = {3, 4};\n")
        f.write("Line(4) = {4, 1};\n")
        f.write("Line(5) = {3, 6};\n")
        f.write("Line(6) = {6, 5};\n")
        f.write("Line(7) = {5, 4};\n")
        f.write("Curve Loop(1) = {1, 2, 3, 4};\n")
        f.write("Curve Loop(2) = {3, 5, 6, 7};\n")
        f.write("Surface(1) = {1};\n")
        f.write("Surface(2) = {2};\n")
        f.write("Physical Surface(\"5\") = {2};\n")
        f.write("Physical Surface(\"6\") = {1};\n")
        f.write("Physical Curve(\"1\") = {1};\n")
        f.write("Physical Curve(\"2\") = {2, 5};\n")
        f.write("Physical Curve(\"3\") = {6};\n")
        f.write("Physical Curve(\"4\") = {4, 7};\n")
    done = run_shell_command("gmsh -2 " + output)
    msh2xdmf(dfile, dfile + os.sep, correct_gmsh=True)
    return dfile + os.sep

def get_geometry(dfile: str) -> Geometry():
    """
    t.b.d.
    """
    g = Geometry()
    g.domain , g.facet_function = getXDMF(dfile + "/")
    g.mesh = g.facet_function.mesh()
    g.dim = g.mesh.geometric_dimension()
    return g

def run_shell_command(commandLine):
    """ Wrapper of subprocess.check_output
    Returns:
        Run command with arguments and return its output
    """
    logger = logging.getLogger(__name__)
    logger.info("Running %s", commandLine)
    return check_output(shlex.split(commandLine))

def add_file_appendix(file: str, type="msh"):
    """
    Adds file appendix if it is not set. File type is optional and default 
    is set to "msh". Returns file with appendix.

    *Arguments*
        file:       String of input file
        type:       String of appendix

    *Return*
        file:       String of file with appendix

    *Example*:
        var = add_file_appendix("brain_file"):
    """
    if not file.endswith("."+type):
        file += "."+type
    return file

def msh2xdmf(inputfile, outputfolder, correct_gmsh=False):
    """
    Generates from input msh file two or three output files in xdmf format for input into FEniCS. Output files are:
    (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputfile: input_msh_file
        outputfolder: output_folder

    *Example:*
        msh2xdmf("inputdata/Terzaghi.msh", "Terzaghi_2d")
    """
    if not inputfile.endswith(".msh"): inputfile += ".msh"
    try:
        Path(outputfolder).mkdir(parents=True, exist_ok=False)
    except (FileExistsError):
        print("Folder already exists")

    msh = meshio.read(inputfile)
    cells = {"tetra": None, "triangle": None, "line": None, "vertex": None}
    data = {"tetra": None, "triangle": None, "line": None, "vertex": None}

    for cell in msh.cells:
        if cells[cell.type] is None:
            cells[cell.type] = cell.data
        else:
            cells[cell.type] = np.vstack([cells[cell.type], cell.data])

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if data[key] is None:
            data[key] = msh.cell_data_dict["gmsh:physical"][key]
        else:
            data[key] = np.vstack([data[key], msh.cell_data_dict["gmsh:physical"][key]])

    if correct_gmsh:
        points = np.zeros((len(msh.points), 2))
        for i, point in enumerate(msh.points):
            points[i] = [point[0], point[1]]
    else:
        points = msh.points

    for key in cells:
        if cells[key] is not None:
            print("write ", key, "_mesh")
            mesh = meshio.Mesh(points=points, cells={key: cells[key]}, cell_data={"name_to_read": [data[key]]})
            meshio.write(outputfolder + os.sep + str(key) + ".xdmf", mesh)
    return True

# noinspection PyBroadException
def getXDMF(inputdirectory):
    """
    Gathers all needed input files from a respective folder in workingdata environment and returns
    the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputdirectory: input_folder

    *Example:*
        getXDMF("Terzaghi_2d")
    """
    try:
        xdmf_files = []
        if os.path.isfile(inputdirectory+"/tetra.xdmf"):
            mesh = df.Mesh()
            with df.XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(mesh)

            tetra_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile(inputdirectory + "/tetra.xdmf") as infile:
                infile.read(tetra_mvc, "name_to_read")
            tetra = df.MeshFunction("size_t", mesh, tetra_mvc)
            xdmf_files.append(tetra)

            triangle_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = df.MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            if os.path.isfile(inputdirectory+"/line.xdmf"):
                line_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
                with df.XDMFFile(inputdirectory + "/line.xdmf") as infile:
                    infile.read(line_mvc, "name_to_read")
                line = df.MeshFunction("size_t", mesh, line_mvc)
                xdmf_files.append(line)

            if os.path.isfile(inputdirectory + "point.xdmf"):
                point_mvc = df.MeshValueCollection("size_t", mesh)
                with df.XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = df.MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        else:
            mesh = df.Mesh()
            with df.XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(mesh)

            triangle_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile(inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = df.MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            line_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile(inputdirectory + "/line.xdmf") as infile:
                infile.read(line_mvc, "name_to_read")
            line = df.MeshFunction("size_t", mesh, line_mvc)
            xdmf_files.append(line)

            if os.path.isfile(inputdirectory+"/point.xdmf"):
                point_mvc = df.MeshValueCollection("size_t", mesh)
                with df.XDMFFile(inputdirectory + "/point.xdmf") as infile:
                    infile.read(point_mvc, "name_to_read")
                point = df.MeshFunction("size_t", mesh, point_mvc)
                xdmf_files.append(point)

        return xdmf_files
    except:
        print("input not working")
    pass

def set_output_file(name: str):
    """
    Initializes xdmf file of given name. That file can be filled with multiple fields using the same mesh

    *Arguments*
        name: File path    

    *Example*
        output_file = set_output_file("solution/3d_crashtest.xdmf")
    """
    xdmf_file = df.XDMFFile(name+".xdmf")
    xdmf_file.rename(name, "x")
    xdmf_file.parameters["flush_output"] = True                        
    xdmf_file.parameters["functions_share_mesh"] = True    
    return xdmf_file

def write_field2output(outputfile: df.XDMFFile, field: df.Function, fieldname: str, timestep: int, id_nodes=None, mesh=None):
    """
    writes field to outputfile, also can write nodal values into separated txt-files. Therefore, list of nodal id's and mesh should be given.
    In case of non-scalar fields, field_dim should be given.

    *Arguments:*
        outputfile: xdmf_file
        field: scalar, vector or tensor-valued field
        fieldname: String
        timestep: respective timestep
        id_nodes: list of node identifiers 
        mesh: respective mesh

    *Example:*
        write_field2output(xdmf_file, u, "displacement", t)
    """
    field.rename(fieldname, fieldname)
    outputfile.write(field, timestep)
    if id_nodes is not None:
        if timestep == 0:
            with open(outputfile.name() + "-" + fieldname + ".txt", "w") as myfile:
                myfile.write(str(field.value_rank())+"\t")
                for node in id_nodes:
                    myfile.write(str(node)+"\t")
                myfile.write("\n")
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node]))+"\t")
                myfile.write("\n")
        else:
            with open(outputfile.name() + "-" + fieldname + ".txt", "a") as myfile:
                myfile.write(str(timestep) + "\t")
                for node in id_nodes:
                    myfile.write(str(field(mesh.coordinates()[node]).tolist())+"\t")
                myfile.write("\n")
        return [[field(mesh.coordinates()[node]), node] for node in id_nodes]
    return True

def nonlinvarsolver(res, x, bcs, solver_param):
    """
    defines and initialises a non-linear variational problem and set up a solver scheme.

    *Arguments*
        res: residuum of problem, given by variational formulation of set of partial differential equations
        x:   solution vector, in terms of Ax=b
        bcs: dirichlet boundary conditions in form of a list
        Solver_param: Input class for solver parameters

    *Output*
        gives solver class that can be executed

    *Example*
        solver = nonlinvarsolver(residual_Momentum, x, dirichlet_boundaries, solver_parameters)
        solver.solve()
    """
    J = df.derivative(res, x)
    # Initialize solver
    problem = df.NonlinearVariationalProblem(res, x, bcs=bcs, J=J)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['maximum_iterations'] = solver_param.maxIter
    solver.parameters['newton_solver']['relative_tolerance'] = solver_param.rel
    solver.parameters['newton_solver']['absolute_tolerance'] = solver_param.abs
    solver.parameters['newton_solver']['linear_solver'] = solver_param.solver_type
    df.PETScOptions.set("-mat_mumps_cntl_1", 0.05)
    df.PETScOptions.set("-mat_mumps_icntl_23", 102400)
    return solver

def norm(field: df.Function):
    """
    t.b.d.
    """
    return ufl.sqrt(sum([x*x for x in field.split()]))
