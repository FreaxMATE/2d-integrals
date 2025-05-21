import math
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y


class Triangle:
    def __init__(self, node_1, node_2, node_3):
        """
        Explanation

        Arguments:
            node_1: Explanation
            node_2: Explanation
            node_3: Explanation

        Returns:
            None
        """
        if not isinstance(node_1, Node) or not isinstance(node_2, Node) or not isinstance(node_3, Node):
            raise TypeError('Triangle consists of three nodes.')

        self.n1 = node_1
        self.n2 = node_2
        self.n3 = node_3
        pass

    def get_nodes(self):
        """
        Explanation

        Arguments:
            None

        Returns:
            None
        """
        return self.n1, self.n2, self.n3

class Mesh:
    def __init__(self, coordinates, nodes):
        """
        Explanation

        Arguments:
            coordinates: Explanation
            nodes: Explanation

        Returns:
            None
        """
        self.triangels = []

        n_triangles = int(len(nodes[0]))

        #if len(nodes.shape) == 1:
         #   nodes = np.expand_dims(nodes, axis=1)

        for i in range(n_triangles):
            n1 = nodes[0][i]-1
            n2 = nodes[1][i]-1
            n3 = nodes[2][i]-1

            x1 = coordinates[0][int(n1)]
            y1 = coordinates[1][int(n1)]
            x2 = coordinates[0][int(n2)]
            y2 = coordinates[1][int(n2)]
            x3 = coordinates[0][int(n3)]
            y3 = coordinates[1][int(n3)]

            triangle = Triangle(Node(x1, y1), Node(x2, y2), Node(x3, y3))
            self.triangels.append(triangle)

        self.determinants = self.list_determinants(triangles=self.triangels)


    # Task 2
    def determinant(self, triangle):
        """
        Explanation

        Arguments:
            triangle: Triangle to get coordinates of the three nodes

        Returns:
            determinant of the jacobian matrix
        """
        if not isinstance(triangle, Triangle):
            raise TypeError('Can only compute determinant of type Triangle.')

        n1, n2, n3 = triangle.get_nodes()

        jacobian = [[(n2.x - n1.x), (n2.y - n1.y)], [(n3.x - n1.x), (n3.y - n1.y)]]

        return jacobian[0][0]*jacobian[1][1] - jacobian[0][1]*jacobian[1][0]

    # Task 3
    def minimum_angle(self, triangle):
        """
        Explanation

        Arguments:
            triangle: Explanation

        Returns:
            None
        """
        n1, n2, n3 = triangle.get_nodes()

        # Calculate vectors relative to n1
        v1 = (n2.x - n1.x, n2.y - n1.y)
        v2 = (n3.x - n1.x, n3.y - n1.y)
        v3 = (n3.x - n2.x, n3.y - n2.y)

        # Compute angles using the dot product formula
        phi_12 = math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)))
        phi_23 = math.acos((-v1[0]*v3[0] - v1[1]*v3[1]) / (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v3[0]**2 + v3[1]**2)))
        phi_13 = math.pi - phi_12 - phi_23

        return min(phi_12, phi_23, phi_13)

    # Task 4
    def list_determinants(self, triangles):
        """
        Explanation

        Arguments:
            List of triangles

        Returns:
            None
        """
        determinants = []
        for t in triangles:
            if self.minimum_angle(t) < (2*math.pi/(360)):
                raise ValueError('Too small angle in triangle.')
            determinants.append(self.determinant(t))
        return determinants

    # Task 5
    def total_integral(self, function):
        """
        Explanation

        Arguments:
            function: Explanation

        Returns:
            None
        """
        total_integral = 0
        for i, t in enumerate(self.triangels):
            n1, n2, n3 = t.get_nodes()
            integral_triangle = (1/6)*abs(self.determinants[i])*(function(n1.x, n1.y) + function(n2.x, n2.y) + function(n3.x, n3.y))
            total_integral += integral_triangle
        return total_integral

    # Task 6
    def total_area(self):
        """
        Explanation

        Arguments:
            None

        Returns:
            None
        """
        total_area = 0
        for i, t in enumerate(self.triangels):
            total_area += 0.5*abs(self.determinants[i])

        return total_area

    # Task 7
    def draw(self):
        """
        Explanation

        Arguments:
            None

        Returns:
            None
        """
        fig, axes = plt.subplots(figsize=(10, 10))

        scale = 0.3

        for t in self.triangels:
            n1, n2, n3 = t.get_nodes()
            com_x = (n1.x + n2.x + n3.x)/3
            com_y = (n1.y + n2.y + n3.y)/3

            # Area of a triangle: 
            # 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|.
            # https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/

            #area = 1/2 * abs(n1.x*(n2.y - n3.y) + n2.y*(n3.y - n1.y) + n3.x*(n1.y - n2.y))

            scale = scale

            axes.plot([n1.x+(com_x-n1.x)*scale, n2.x+(com_x-n2.x)*scale, n3.x+(com_x-n3.x)*scale, n1.x+(com_x-n1.x)*scale], [n1.y+(com_y-n1.y)*scale, n2.y+(com_y-n2.y)*scale, n3.y+(com_y-n3.y)*scale, n1.y+(com_y-n1.y)*scale], label='scaled')

            #axes.plot([n1.x, n2.x, n3.x, n1.x], [n1.y, n2.y, n3.y, n1.y], label='unscaled')
            # axes.plot([n1.x], [n1.y], '.', label='1')
            # axes.plot([n2.x], [n2.y], '.', label='2')
            # axes.plot([n3.x], [n3.y], '.', label='3')
            #axes.scatter(com_x, com_y, c='black')

        plt.show()

if __name__ == "__main__":
    coordinates = np.loadtxt('meshes/coordinates_dolfin_coarse.txt')
    nodes = np.loadtxt('meshes/nodes_dolfin_coarse.txt')
    #coordinates = np.loadtxt('meshes_examples/coord_one.txt')
    #nodes = np.loadtxt('meshes_examples/elementnode_one.txt')
    m = Mesh(coordinates=coordinates, nodes=nodes)
    m.draw()

    print(m.total_integral(lambda x, y: x+y))
    print(m.total_integral(lambda x, y: x*y))
    print(m.total_integral(lambda x, y: x**2+y**2))
    print(m.total_integral(lambda x, y: 1))
    print(m.total_area())
