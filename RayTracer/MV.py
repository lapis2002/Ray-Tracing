import numpy as np
'''
=========================================================================================================
=========================================== VECTOR ======================================================
=========================================================================================================
'''
def vec3 (e0: float, e1: float, e2: float):
    return np.array([e0, e1, e2])

def homoToCausian (vec4):
    return vec3(vec4[0][0], vec4[1][0], vec4[2][0])

def transformPoint(mat: np.ndarray, point: np.ndarray):
    pointHomo = np.array([[point[0]], [point[1]], [point[2]], [1]])
    pointHomo = mult(mat, pointHomo)
    pointTransformed = homoToCausian(pointHomo)
    return pointTransformed

def transformVector(mat: np.ndarray, vec: np.ndarray):
    vecHomo = np.array([[vec[0]], [vec[1]], [vec[2]], [0]])
    vecHomo = mult(mat, vecHomo)
    vecTransformed = homoToCausian(vecHomo)
    return vecTransformed

def normalize(v: np.ndarray, excludeLastComp = False):
    if (excludeLastComp):
        v = np.delete(v, v.size-1)

    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def cross (u: np.ndarray, v: np.ndarray):
    return np.cross(u, v)

def dot (u: np.ndarray, v: np.ndarray):
    return np.dot(u, v)

def mix (u: np.ndarray, v: np.ndarray, s: np.ndarray):
    assert(u.size == v.size)

    result = []
    for i in range(u.size):
        result.append(s * u[i] + (1-s)*v[i])

    return np.array(result)

def scalev (u: np.ndarray, s: float):
    return u*s
'''
=========================================================================================================
=========================================== MATRIX ======================================================
=========================================================================================================
'''

def identity_mat(n: np.ndarray):
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i][i] = 1

    return mat
    
def mult (matA: np.ndarray, matB: np.ndarray):
    return np.matmul(matA, matB)

def invert(mat: np.ndarray):
    return np.linalg.inv(mat)

def transpose(mat: np.ndarray):
    return mat.transpose()

def flatten(mat: np.ndarray):
    return mat.flatten()

'''
=========================================================================================================
===================================== MATHEMATIC OPERATIONS =============================================
=========================================================================================================
'''    
def to_radians(degree: float): 
    return degree * np.pi / 180.0

def equal (u: np.ndarray, v:np.ndarray):
    return np.array_equal(u, v)

def add (u: np.ndarray, v: np.ndarray):
    return np.add(u, v)

def substract(u: np.ndarray, v: np.ndarray):
    return np.subtract(u,v)

def translate(x: float, y: float, z: float):
    mat4 = np.zeros((4, 4))
    mat4[0][0] = 1
    mat4[1][1] = 1
    mat4[2][2] = 1
    mat4[3][3] = 1

    mat4[0][3] = x
    mat4[1][3] = y
    mat4[2][3] = z

    return mat4

def scale (x:float, y: float, z: float):
    mat4 = np.zeros((4, 4))
    mat4[0][0] = x
    mat4[1][1] = y
    mat4[2][2] = z
    mat4[3][3] = 1

    return mat4

def inverseScale (x: float, y: float, z: float):
    return scale(1/x, 1/y, 1/z)

def inverseTranslate (x: float, y: float, z: float):
    return translate(-x, -y, -z)

def inverseTransposeTranslate (x: float, y: float, z: float):
    return inverseTranslate(x, y, z).transpose()

def inverseTransposeScale (x: float, y: float, z: float):
    return inverseScale(x, y, z)
'''
Return rotate matrix
@ param angle in degree
@ param axis np array of axis
'''
def rotate(angle: float, axis: float):
    v = normalize(axis)

    x = v[0]
    y = v[1]
    z = v[2]

    c = np.cos(to_radians(angle))
    omc = 1.0 - c
    s = np.sin(to_radians(angle))

    r1 = np.array([x*x*omc + c,   x*y*omc - z*s, x*z*omc + y*s, 0.0])
    r2 = np.array([x*y*omc + z*s, y*y*omc + c,   y*z*omc - x*s, 0.0])
    r3 = np.array([x*z*omc - y*s, y*z*omc + x*s, z*z*omc + c,   0.0])
    r4 = np.zeros(4)
    result = np.array([r1, r2, r3, r4])

    return result

def look_at (eye: np.ndarray, at: np.ndarray, up: np.ndarray):
    assert(eye.size == 3)
    assert(at.size == 3)
    assert(up.size == 3)

    if equal(eye, at):
        return np.zeros((4, 4))

    v = normalize(substract(at, eye))   # view direction vector
    n = normalize(cross(v, up))         # perpendicular vector
    u = normalize(cross(n, v))          # "new" up vector

    v = (-v)

    result = np.array(
        np.append(n, -dot(n, eye)),
        np.append(u, -dot(u, eye)),
        np.append(v, -dot(v, eye)),
        np.zeros(4)
    )

    return result

def ortho (left: float, right: float, 
           top: float, bottom: float, 
           near: float, far: float):
    assert(left != right)
    assert(bottom != top)
    assert(near != far)

    w = right - left
    h = top - bottom
    d = far - near

    result = np.zeros((4, 4))
    result[0][0] = 2.0 / w
    result[1][1] = 2.0 / h 
    result[2][2] = -2.0 / d
    result[0][3] = -(left + right) / w
    result[1][3] = -(top + bottom) / h
    result[2][3] = -(near + far) / d

    return result



