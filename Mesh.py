import math
import numpy as np
import os
import matplotlib.pyplot as plt

class Node:
    """
    Represents a node in 2D space with x and y coordinates.
    """
    def __init__(self, x, y):
        """
        Initializes a Node object.

        Arguments:
            x: The x-coordinate of the node.
            y: The y-coordinate of the node.

        Returns:
            None
        """
        self.x = x
        self.y = y
    
    def get_x(self):
        """
        Retrieves the x-coordinate of the node.

        Arguments:
            None

        Returns:
            The x-coordinate of the node.
        """
        return self.x
    
    def get_y(self):
        """
        Retrieves the y-coordinate of the node.

        Arguments:
            None

        Returns:
            The y-coordinate of the node.
        """
        return self.y


class Triangle:
    def __init__(self, node_1, node_2, node_3):
        """
        Initializes a Triangle object with three nodes.

        Arguments:
            node_1: The first node of the triangle (instance of Node).
            node_2: The second node of the triangle (instance of Node).
            node_3: The third node of the triangle (instance of Node).

        Returns:
            None
        """
        # Ensure all inputs are instances of the Node class
        if not isinstance(node_1, Node) or not isinstance(node_2, Node) or not isinstance(node_3, Node):
            raise TypeError('Triangle consists of three nodes.')

        # Assign the nodes to the triangle
        self.n1 = node_1
        self.n2 = node_2
        self.n3 = node_3

    def get_nodes(self):
        """
        Retrieves the three nodes of the triangle.

        Arguments:
            None

        Returns:
            A tuple containing the three nodes (n1, n2, n3).
        """
        return self.n1, self.n2, self.n3

class Mesh:
    def __init__(self, coordinates, nodes):
        """
        Initializes the Mesh object by creating triangles from the given coordinates and nodes.

        Arguments:
            coordinates: A 2D array where each column represents the x and y coordinates of a node.
            nodes: A 2D array where each column contains the indices of the nodes forming a triangle.

        Returns:
            None
        """
        self.triangels = []  # List to store all the triangles in the mesh.

        # Number of triangles in the mesh.
        n_triangles = int(len(nodes[0]))

        # Loop through each triangle defined in the nodes array.
        for i in range(n_triangles):
            # Get the indices of the three nodes forming the triangle (adjusted for 0-based indexing).
            n1 = nodes[0][i] - 1
            n2 = nodes[1][i] - 1
            n3 = nodes[2][i] - 1

            # Retrieve the coordinates of the three nodes.
            x1 = coordinates[0][int(n1)]
            y1 = coordinates[1][int(n1)]
            x2 = coordinates[0][int(n2)]
            y2 = coordinates[1][int(n2)]
            x3 = coordinates[0][int(n3)]
            y3 = coordinates[1][int(n3)]

            # Create a Triangle object using the three nodes and add it to the list of triangles.
            triangle = Triangle(Node(x1, y1), Node(x2, y2), Node(x3, y3))
            self.triangels.append(triangle)

        # Compute and store the determinants of the Jacobian matrices for all triangles.
        self.determinants = self.compute_all_determinants(triangles=self.triangels)


    # Task 2
    def determinant(self, triangle):
        """
        Computes the determinant of the Jacobian matrix for a given triangle.

        Arguments:
            triangle: Triangle object to get coordinates of the three nodes.

        Returns:
            Determinant of the Jacobian matrix.
        """
        # Ensure the input is a Triangle object.
        if not isinstance(triangle, Triangle):
            raise TypeError('Can only compute determinant of type Triangle.')

        # Retrieve the three nodes of the triangle.
        n1, n2, n3 = triangle.get_nodes()

        # Construct the Jacobian matrix using the coordinates of the nodes.
        # The Jacobian matrix is formed by the vectors from n1 to n2 and n1 to n3.
        jacobian = [[(n2.get_x() - n1.get_x()), (n2.get_y() - n1.get_y())], 
                [(n3.get_x() - n1.get_x()), (n3.get_y() - n1.get_y())]]

        # Compute and return the determinant of the Jacobian matrix.
        # Determinant formula: ad - bc for a 2x2 matrix [[a, b], [c, d]].
        return jacobian[0][0]*jacobian[1][1] - jacobian[0][1]*jacobian[1][0]

    # Task 3
    def minimum_angle(self, triangle):
        """
        Computes the minimum angle of a given triangle.

        Arguments:
            triangle: A Triangle object for which the minimum angle is to be calculated.

        Returns:
            The smallest angle (in radians) among the three angles of the triangle.
        """
        # Retrieve the three nodes of the triangle.
        n1, n2, n3 = triangle.get_nodes()

        # Calculate vectors relative to the first node (n1).
        v1 = (n2.get_x() - n1.get_x(), n2.get_y() - n1.get_y())  # Vector from n1 to n2.
        v2 = (n3.get_x() - n1.get_x(), n3.get_y() - n1.get_y())  # Vector from n1 to n3.
        v3 = (n3.get_x() - n2.get_x(), n3.get_y() - n2.get_y())  # Vector from n2 to n3.

        # Compute the angle between v1 and v2 using the dot product formula.
        # Formula: cos(theta) = (v1 . v2) / (|v1| * |v2|).
        phi_12 = math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / 
                           (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)))

        # Compute the angle between -v1 and v3 using the dot product formula.
        # This is equivalent to the angle at the second node.
        phi_23 = math.acos((-v1[0]*v3[0] - v1[1]*v3[1]) / 
                           (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v3[0]**2 + v3[1]**2)))

        # Compute the third angle using the triangle angle sum property.
        # The sum of angles in a triangle is always pi radians.
        phi_13 = math.pi - phi_12 - phi_23

        # Return the smallest angle among the three.
        return min(phi_12, phi_23, phi_13)

    # Task 4
    def compute_all_determinants(self, triangles):
        """
        Computes the determinants of the Jacobian matrices for a list of triangles.
        Also checks if any triangle has an angle smaller than a threshold (2 degrees).

        Arguments:
            triangles: List of Triangle objects.

        Returns:
            A list of determinants for the given triangles.
        """
        determinants = []  # Initialize an empty list to store determinants.

        # Iterate through each triangle in the list.
        for t in triangles:
            # Check if the minimum angle of the triangle is less than 2 degrees (in radians).
            if self.minimum_angle(t) < (2 * math.pi / 360):
                raise ValueError('Too small angle in triangle.')

            # Compute the determinant of the Jacobian matrix for the triangle.
            determinants.append(self.determinant(t))

        # Return the list of computed determinants.
        return determinants

    # Task 5
    def total_integral(self, function):
        """
        Computes the total integral of a given function over the entire mesh.

        Arguments:
            function: A callable function that takes two arguments (x, y) and returns a value.

        Returns:
            The total integral of the function over all triangles in the mesh.
        """
        total_integral = 0  # Initialize the total integral to zero.

        # Iterate through each triangle in the mesh.
        for i, t in enumerate(self.triangels):
            # Retrieve the three nodes of the triangle.
            n1, n2, n3 = t.get_nodes()

            # Compute the integral over the current triangle using the midpoint rule.
            # The formula is (1/6) * |determinant| * (f(n1) + f(n2) + f(n3)).
            integral_triangle = (1/6) * abs(self.determinants[i]) * (
                function(n1.get_x(), n1.get_y()) + function(n2.get_x(), n2.get_y()) + function(n3.get_x(), n3.get_y())
            )

            # Add the triangle's contribution to the total integral.
            total_integral += integral_triangle

        # Return the computed total integral.
        return total_integral

    # Task 6
    def total_area(self):
        """
        Computes the total area of the mesh by summing up the areas of all triangles.

        Arguments:
            None

        Returns:
            The total area of the mesh.
        """
        total_area = 0  # Initialize the total area to zero.

        # Iterate through each triangle in the mesh.
        for i, t in enumerate(self.triangels):
            # Compute the area of the triangle using the determinant of its Jacobian matrix.
            # The formula for the area of a triangle is 0.5 * |determinant|.
            total_area += 0.5 * abs(self.determinants[i])

        # Return the computed total area.
        return total_area

    # Task 7
    def draw(self, save_path=None):
        """
        Draws the mesh by plotting the scaled triangles.
        Optionally saves the plot to a file if save_path is provided.

        Arguments:
            save_path: Optional; if provided, the plot will be saved to this path.

        Returns:
            None
        """
        fig, axes = plt.subplots(figsize=(10, 10))

        # Scaling factor for shrinking triangles towards their center of mass.
        scale = 0.2

        # Iterate through each triangle in the mesh.
        for t in self.triangels:
            # Retrieve the three nodes of the triangle.
            n1, n2, n3 = t.get_nodes()

            # Compute the center of mass (COM) of the triangle.
            com_x = (n1.get_x() + n2.get_x() + n3.get_x()) / 3
            com_y = (n1.get_y() + n2.get_y() + n3.get_y()) / 3

            # Plot the scaled triangle by moving each vertex closer to the COM.
            axes.plot(
                [n1.get_x() + (com_x - n1.get_x()) * scale, n2.get_x() + (com_x - n2.get_x()) * scale, n3.get_x() + (com_x - n3.get_x()) * scale, n1.get_x() + (com_x - n1.get_x()) * scale],
                [n1.get_y() + (com_y - n1.get_y()) * scale, n2.get_y() + (com_y - n2.get_y()) * scale, n3.get_y() + (com_y - n3.get_y()) * scale, n1.get_y() + (com_y - n1.get_y()) * scale],
                label='scaled'
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)

if __name__ == "__main__":
    # Create a subdirectory for plots if it doesn't exist
    plot_dir = "meshes/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Open results.txt for writing all print outputs
    results_file = open(os.path.join(plot_dir, "results.txt"), "w")

    def print_and_log(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=results_file)

    coordinates = np.loadtxt('meshes/coordinates_dolfin_coarse.txt')
    nodes = np.loadtxt('meshes/nodes_dolfin_coarse.txt')
    mesh = Mesh(coordinates=coordinates, nodes=nodes)
    mesh.draw(save_path=os.path.join(plot_dir, 'mesh_plot_dolfin_coarse.png'))
    print_and_log(f"Total area of dolfin_coarse mesh: {mesh.total_area()}, Total integral: {mesh.total_integral(lambda x, y: 1)}")

    # List of mesh sizes and corresponding file names
    mesh_sizes = [400, 1024, 2500, 10000]
    total_integrals = []
    total_areas = []

    for size in mesh_sizes:
        coordinates = np.loadtxt(f'meshes/coordinates_unitcircle_{size}.txt')
        nodes = np.loadtxt(f'meshes/nodes_unitcircle_{size}.txt')
        m = Mesh(coordinates=coordinates, nodes=nodes)
        integral = m.total_integral(lambda x, y: 1)
        area = m.total_area()
        total_integrals.append(integral)
        total_areas.append(area)
        # Save mesh plot for each mesh size
        m.draw(save_path=os.path.join(plot_dir, f'mesh_plot_{size}.png'))
        print_and_log(f"Mesh size: {size}, Total area: {area}, Total integral: {integral}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(mesh_sizes, total_areas, 's-', label='Total Area')
    plt.axhline(y=math.pi, color='tab:orange', linestyle='--', label='π', zorder=0)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Value')
    plt.title('Total Integral and Area vs Mesh Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'total_area_vs_mesh_size.png'))

    # Additional: Compute total integrals and areas for different functions on unitcircle meshes
    functions = [
        # (Description, LaTeX, Function, Analytical solution, Value)
        ("f(x, y) = 1", r"$f(x, y) = 1$", lambda x, y: 1, "π", math.pi),
        ("f(x, y) = x", r"$f(x, y) = x$", lambda x, y: x, "0", 0.0),
        ("f(x, y) = x**2 + y**2", r"$f(x, y) = x^2 + y^2$", lambda x, y: x**2 + y**2, "π/2", math.pi / 2),
        ("f(x, y) = exp(-x**2 - y**2)", r"$f(x, y) = e^{-x^2 - y^2}$", lambda x, y: np.exp(-x**2 - y**2), "π(1 - e^{-1})", math.pi * (1 - math.exp(-1))),
    ]

    mesh_sizes = [400, 1024, 2500, 10000]
    for fname, latex_name, func, analytical_str, analytical_val in functions:
        integrals = []
        diffs = []
        for size in mesh_sizes:
            coordinates = np.loadtxt(f'meshes/coordinates_unitcircle_{size}.txt')
            nodes = np.loadtxt(f'meshes/nodes_unitcircle_{size}.txt')
            m = Mesh(coordinates=coordinates, nodes=nodes)
            integral = m.total_integral(func)
            area = m.total_area()
            diff = abs(integral - analytical_val)
            integrals.append(integral)
            diffs.append(diff)
            print_and_log(f"{fname}: Mesh size {size}, Integral {integral:.6f}, Analytical: {analytical_str} ({analytical_val:.6f}), Difference: {diff:.6e}")

            # 3D plot of the function over the mesh nodes for all mesh sizes
            x = coordinates[0]
            y = coordinates[1]
            z = np.array([func(xi, yi) for xi, yi in zip(x, y)])

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0.2)
            ax.set_title(f"3D plot of {latex_name} on unitcircle mesh ({size} nodes)")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(latex_name)
            plt.tight_layout()
            # Use a safe filename
            safe_name = latex_name.replace(" ", "").replace("(", "").replace(")", "").replace("*", "star").replace("=", "").replace(",", "_").replace("$", "").replace("^", "pow").replace("{", "").replace("}", "")
            plt.savefig(os.path.join(plot_dir, f'3dplot_{safe_name}_{size}.png'))
            plt.close()

        # Plot calculated integral vs mesh size and compare to analytical value
        plt.figure(figsize=(8, 6))
        plt.plot(mesh_sizes, integrals, 'o-', label='Numerical Integral')
        plt.axhline(y=analytical_val, color='tab:orange', linestyle='--', label=f'Analytical: {analytical_str}')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Integral Value')
        plt.title(f'Integral of {latex_name} vs Mesh Size')
        plt.legend()
        plt.tight_layout()
        safe_name = latex_name.replace(" ", "").replace("(", "").replace(")", "").replace("*", "star").replace("=", "").replace(",", "_").replace("$", "").replace("^", "pow").replace("{", "").replace("}", "")
        plt.savefig(os.path.join(plot_dir, f'integral_vs_meshsize_{safe_name}.png'))
        plt.close()

    results_file.close()
