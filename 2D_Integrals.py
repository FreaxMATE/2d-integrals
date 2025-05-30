# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:47:56 2025

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def _list(self):
        return (self.x, self.y)
    def _np_array(self):
        return np.array([self.x, self.y])
    
    def __add__(self, VECTOR):
        return Vector2D(self.x + VECTOR.x, self.y + VECTOR.y)
    def __radd__(self, VECTOR):
        return self - VECTOR
        
    def __sub__(self, VECTOR):
        return Vector2D(self.x - VECTOR.x, self.y - VECTOR.y)
    
    def __rsub__(self, VECTOR):
        return self - VECTOR
    
    # Floating point denominator D
    def __truediv__(self, D):
        return Vector2D(self.x / D, self.y / D)
    
    def dot(self, VECTOR):
        return self.x * VECTOR.x + self.y * VECTOR.y
    
    def length(self):
        return np.sqrt(self.x**2 + self.y**2)
    
    def unit_vector(self):
        _length = self.length()
        
        # If length == 0, do not normalize
        if _length == 0:
            return self
        
        return Vector2D(self.x/_length, self.y/_length)
    
    # Returns angle between self and another vector
    def angle(self, VECTOR):
        cosine_angle = self.dot(VECTOR) / (self.length()*VECTOR.length())
        # Clamp the values
        cosine_angle = max(-1.0, min(1.0, cosine_angle))
        return np.arccos(cosine_angle) # Radians
    
    # Printable
    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"
        
    x = 0.0
    y = 0.0
    
# UNIT TRIANGLE

UNIT_TRIANGLE = [Vector2D(0.0, 0.0), Vector2D(0.0, 1.0), Vector2D(1.0, 0.0)]

MIN_ANGLE_THRESHOLD = 1 # Degrees

# Transformation function, maps an arbitrary triangle (P1, P2, P3) onto a defined unit triangle
def Map2_Unit_Triangle(UNIT_TRIANGLE_POINT, P1, P2, P3):
    x = P1.x + (P2.x - P1.x)*UNIT_TRIANGLE_POINT.x - (P3.x - P1.x)*UNIT_TRIANGLE_POINT.x
    y = P1.y + (P2.y - P1.y)*UNIT_TRIANGLE_POINT.y - (P3.y - P1.y)*UNIT_TRIANGLE_POINT.y
    
    return Vector2D(x, y)
    
class Mesh:
    
    # List of integers (Nodes)
    elements = []
    # List of Vector2D's
    coordinates = []
    # List of floats
    jacobian_determinants = []
    
    mesh_center_point = Vector2D(0, 0)
    
    # MESH = Tuple of two matrices (coordinates, elements)
    def __init__(self, MESH):
        self.coordinates = MESH[0]
        self.elements = MESH[1]
        
        self.compute_Jacobian_Determinants()
                
        # Calculate mean point e.g. the center
        for position in self.coordinates:
            self.mesh_center_point += position
        self.mesh_center_point = self.mesh_center_point / len(self.coordinates)
                
    # Jacobian Determinant based off of the transformation matrix formed from the 'Map2_Unit_Triangle' function
    def compute_J_determinant(self, A, B, C):
        return (A.x - C.x)*(B.y - A.y) - (A.y - C.y)*(B.x - A.x)
        
    def compute_Jacobian_Determinants(self):
        triangles = [(a-1, b-1, c-1) for a, b, c in self.elements]
        
        for T_i in triangles:
            if(self.compute_min_angle(T_i) < MIN_ANGLE_THRESHOLD):
                self.jacobian_determinants.append(0)
                continue
                #raise ValueError(f"Triangle {T_i} has an angle smaller than 1°. This triangle is too thin for numerical stability.")
            
            determinant = self.compute_J_determinant(self.coordinates[T_i[0]], self.coordinates[T_i[1]], self.coordinates[T_i[2]])
            self.jacobian_determinants.append(determinant)
            
    # Compute minimum angle in a given element/triangle
    def compute_min_angle(self, ELEMENT):
        N1, N2, N3 = ELEMENT
        
        A = self.coordinates[N1 - 1]
        B = self.coordinates[N2 - 1]
        C = self.coordinates[N3 - 1]
        
        # Compute edges of triangle
        EDGE_AB = B - A
        EDGE_AC = C - A
        
        EDGE_BA = A - B
        EDGE_BC = C - B
        
        EDGE_CA = A - C
        EDGE_CB = B - C
        
        AngleA = EDGE_AB.angle(EDGE_AC)
        AngleB = EDGE_BA.angle(EDGE_BC)
        AngleC = EDGE_CA.angle(EDGE_CB)
        
        # Triangle Angles
        #print(np.degrees(AngleA))
        #print(np.degrees(AngleB))
        #print(np.degrees(AngleC))
        
        return min(np.degrees(AngleA), np.degrees(AngleB), np.degrees(AngleC))
    
    # Approximation of surface integral for functions defined as f(x, y) (surface = unit triangle) 
    def compute_unit_integral(self, FUNCTION, P1, P2, P3):
        return ((FUNCTION(P1.x, P1.y) + FUNCTION(P2.x, P2.y) + FUNCTION(P3.x, P3.y)))/6 # (1/3)*(1/2) = (1/6)
    
    # Given any function, compute the surface integral for that function (surface = mesh)
    def compute_surface_integral(self, FUNCTION):        
        integral_sum = 0
        
        triangles = [(a-1, b-1, c-1) for a, b, c in self.elements]
        
        for i in range(0, len(triangles)):
            det_J = self.jacobian_determinants[i]
            integral_sum += det_J*self.compute_unit_integral(FUNCTION, UNIT_TRIANGLE[0], UNIT_TRIANGLE[1], UNIT_TRIANGLE[2])
            
        return integral_sum
        
    # Area computed from triangle vertices
    def compute_surface_area(self):
        Area = 0
    
        triangles = [(a-1, b-1, c-1) for a, b, c in self.elements]
        
        for T_i in triangles:
            A = self.coordinates[T_i[0]]
            B = self.coordinates[T_i[1]]
            C = self.coordinates[T_i[2]]
            
            Area += abs(A.x*(B.y - C.y) + B.x*(C.y - A.y) + C.x*(A.y - B.y))/2
            
        return Area

    # Area computed from Jacobian determinants
    def compute_surface_area_J(self):
        Area = 0
            
        for det_J in self.jacobian_determinants:
            Area += abs(det_J)/2
            
        return Area
            
    
# returns a mesh class filled with the data listed in the files
def load_mesh_data(FILE_PATH_COORDINATES, FILE_PATH_ELEMENTS):
    _coordinates = []
    _elements = []
    
    # Load coordinates
    with open(FILE_PATH_COORDINATES, 'r') as mesh_file_coordinates:
        lines = mesh_file_coordinates.readlines()
        if len(lines) != 2:
            raise ValueError("File does not contain two rows.")
            
        # Reads from first and second row in file, and converts the strings into a list of floats
        x_coordinates = list(map(float, lines[0].strip().split()))
        y_coordinates = list(map(float, lines[1].strip().split()))
        
        if len(x_coordinates) != len(y_coordinates):
            raise ValueError("X and Y coordinate lists are not the same length.")
        
        # Create list of Vector2Ds from the lists 
        _coordinates = [Vector2D(x, y) for x, y in zip(x_coordinates, y_coordinates)]
    
    # Load elements
    with open(FILE_PATH_ELEMENTS, 'r') as mesh_file_elements:
        lines = mesh_file_elements.readlines()
        if len(lines) != 3:
            raise ValueError("File does not contain three rows.")
            
        # Load all nodes from each row and convert to integers (scientific notation → float → integer)
        P1_Element_indices = [int(float(s)) for s in lines[0].strip().split()]
        P2_Element_indices = [int(float(s)) for s in lines[1].strip().split()]
        P3_Element_indices = [int(float(s)) for s in lines[2].strip().split()]
                
        # Check so that each row are the same length (same nr. of triangles)
        if not len(P1_Element_indices) == len(P2_Element_indices) == len(P3_Element_indices):
            raise ValueError("Rows are not same length")
            
        
        # Create list of tuples containg (node_1, node_2, node_3)
        _elements = list(zip(P1_Element_indices, P2_Element_indices, P3_Element_indices))
        
    # Returns a Mesh containing coordinates and elements
    mesh = Mesh([_coordinates, _elements])
    return mesh

def scale_triangle(TRIANGLE_COORDINATES, MESH_CENTER_POINT, SCALING_FACTOR):
    centroid = np.mean(TRIANGLE_COORDINATES, axis=0)
    
    return centroid + (TRIANGLE_COORDINATES - centroid) * SCALING_FACTOR

# Render mesh for debugging
def Render_Mesh(mesh):    
    for triangle in mesh.elements:
        coordinates = np.array([mesh.coordinates[triangle[0] - 1]._np_array(),
                                mesh.coordinates[triangle[1] - 1]._np_array(),
                                mesh.coordinates[triangle[2] - 1]._np_array()])
        
        pts_scaled = scale_triangle(coordinates, mesh.mesh_center_point, 0.8)
        
        pts = np.vstack([pts_scaled, pts_scaled[0]])
        
        plt.plot(pts[:, 0], pts[:, 1], linewidth=0.8)
   
    plt.gca().set_aspect('equal')
    plt.title('Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

mesh = load_mesh_data(
    "/home/kunruh/Documents/Studium/Physik/Master/3/Computational_programming/Assignments/Project/2d-integrals/meshes_examples/coord1.txt",
    "/home/kunruh/Documents/Studium/Physik/Master/3/Computational_programming/Assignments/Project/2d-integrals/meshes_examples/elementnode1.txt")

def constant_fun(x, y):
    return 1

print(mesh.compute_surface_area())
print(mesh.compute_surface_area_J())
print(mesh.compute_surface_integral(constant_fun))

Render_Mesh(mesh)
