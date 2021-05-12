#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt

import MV
import ppm

MAX_DEPTH = 5

class Ray:
    def __init__ (self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction

class Light:
    def __init__ (self, 
                  name: str, 
                  pos_x: float, pos_y: float, pos_z: float, 
                  i_r:float, i_g: float, i_b: float
                ):
        self.name = name
        self.pos = MV.vec3(pos_x, pos_y, pos_z)
        self.intensity = MV.vec3(i_r, i_g, i_b)

class Sphere:
    def __init__ (self, 
                  name: str, 
                  pos_x: float, pos_y: float, pos_z: float, 
                  scl_x: float, scl_y: float, scl_z: float, 
                  r: float, g: float, b: float, 
                  k_a: float, k_d: float, k_s: float, k_r: float, n: int,
                ):
        self.name = name
        self.pos = np.array([pos_x, pos_y, pos_z])
        self.scale = np.array([scl_x, scl_y, scl_z])
        self.color = np.array([r, g, b])
        self.k_a = k_a
        self.k_s = k_s
        self.k_d = k_d
        self.k_r = k_r
        self.specular = n
        self.inverseMat = self.getInverseMat()
        self.transposeInverseMat = self.inverseMat.transpose()

    def getTransformationMatrix (self) -> np.ndarray:
        scaleMat = MV.scale(self.scale[0], self.scale[1], self.scale[2])
        translateMat = MV.translate(self.pos[0], self.pos[1], self.pos[2])
        # translate then scale
        return MV.mult(scaleMat, translateMat)
    
    def getInverseMat (self) -> np.ndarray:
        return (MV.mult(MV.inverseScale(self.scale[0], self.scale[1], self.scale[2]), 
                        MV.inverseTranslate(self.pos[0], self.pos[1], self.pos[2])))
    
    def isInside (self, point: np.ndarray) -> bool:
        point_SphereCoord = MV.transformPoint(self.inverseMat, point)
        return (MV.dot(point_SphereCoord, point_SphereCoord) <= 1)

    def intersectRay (self, ray: Ray, isEyeRay:bool=False) -> np.ndarray:
        rayLength = MV.dot(ray.direction, ray.direction) ** 0.5
        ray.direction = MV.normalize(ray.direction)

        rayOrigin_SphereCoord = MV.transformPoint(self.inverseMat, ray.origin)
        rayDir_SphereCoord = MV.transformVector(self.inverseMat, ray.direction)
        a = MV.dot(rayDir_SphereCoord, rayDir_SphereCoord)
        b = MV.dot(rayDir_SphereCoord, rayOrigin_SphereCoord)
        c = MV.dot(rayOrigin_SphereCoord, rayOrigin_SphereCoord) - 1**2
        
        d = b * b - a * c
        
        if d < 0:
            return np.inf
        if d >= 0:
            t1 = (-b + np.sqrt(d)) / a
            t2 = (-b - np.sqrt(d)) / a
            if (isEyeRay):
                tMin = min (t1, t2)
                tMax = max (t1, t2)
                if (tMin > rayLength):
                    return tMin
                elif (tMax > rayLength):
                    return tMax
                else:
                    return np.inf
            answer = min(t1, t2)
            if answer < 0:
                return np.inf
            else:
                return answer

class Scene:
    def __init__ (self, 
                  ambient: np.ndarray, 
                  backgroundColor: np.ndarray,
                  camPoint: np.ndarray,
                  atVector: np.ndarray,
                  objects: list, 
                  lightSources: list
                  ):
        self.ambient = ambient
        self.backgroundColor = backgroundColor
        self.camPoint = camPoint
        self.atVector = atVector
        self.objects = objects
        self.lightSources = lightSources

'''
=========================================================================================================
========================================== PARSING ======================================================
=========================================================================================================
'''
def parse_data(filename: str):
    spheres = []
    lights = []
    input_data = open(filename, "r")

    for line in input_data:
        inp = line.split()
        if inp:
            if inp[0] == "SPHERE":
                spheres.append(Sphere(
                    inp[1], 
                    float(inp[2]), float(inp[3]), float(inp[4]),                      # position: pos_x, pos_y, pos_z
                    float(inp[5]), float(inp[6]), float(inp[7]),                      # scale: scl_x, scl_y, scl_z
                    float(inp[8]), float(inp[9]), float(inp[10]),                     # color: r, g, b
                    float(inp[11]), float(inp[12]), float(inp[13]), float(inp[14]),   # k
                    float(inp[15])                                                    # specular exponent: n
                ))

            elif inp[0] == "LIGHT":
                lights.append(Light(
                    inp[1], 
                    float(inp[2]), float(inp[3]), float(inp[4]),          # position: pos_x, pos_y, pos_z
                    float(inp[5]), float(inp[6]), float(inp[7])           # light: l_r, l_g, l_b
                ))

            elif inp[0] == "BACK":
                backgroundColor = np.array([float(e) for e in inp[1:]])

            elif inp[0] == "AMBIENT":
                ambient = np.array([float(e) for e in inp[1:]])

            elif inp[0] == "OUTPUT":
                output_filename =  inp[1]

            elif inp[0] == "NEAR":
                near = np.array(float(inp[1]))

            elif inp[0] == "LEFT":
                left = float(inp[1])

            elif inp[0] == "RIGHT":
                right = float(inp[1])

            elif inp[0] == "BOTTOM":
                bottom = float(inp[1])

            elif inp[0] == "TOP":
                top = float(inp[1])
            
            elif inp[0] == "RES":
                res = np.array([int(e) for e in inp[1:]])

    return lights, spheres, backgroundColor, ambient, near, left, right, bottom, top, res, output_filename

'''
=========================================================================================================
============================================== RAY TRACING ==============================================
=========================================================================================================
'''
def rayTracer (ray: Ray, scene: Scene):
    # find the intersection of the ray with all the object
    t = np.inf
    for i, obj in enumerate(scene.objects):
        # if the ray is eye-ray, check whether t < 1, 
        # if yes, omit it.
        if (np.array_equal(ray.origin, scene.camPoint)):
            tObj = obj.intersectRay(ray, True)
        else:
            tObj = obj.intersectRay(ray)
        if tObj < t:
            t, obj_idx = tObj, i
    # if there is not any intersection, return
    if t == np.inf:
        return
    
    obj = scene.objects[obj_idx]

    # converse the intersection point to Sphere Coordinate Sytem
    # then calculate the normal vector at the intersection point
    # finally converse it back to the World Coordinate System
    P = ray.origin + ray.direction * t
    P_SphereCoord = MV.transformPoint(obj.inverseMat, P)

    N_SphereCoord = P_SphereCoord - np.zeros(3) # sphere origin is (0, 0, 0) in sphere coordinate system

    # N = P - obj.pos
    # N_SphereCoord = MV.transformVector(obj.inverseMat, N)

    N = MV.transformVector(obj.transposeInverseMat, N_SphereCoord)
    N = MV.normalize(N)

    V = MV.normalize(-P)
    color = obj.color.copy()

    '''
    PIXEL_COLOR[c] = Ka*Ia[c]*O[c] +

    for each point light (p) { Kd*Ip[c]*(N dot L)*O[c]+Ks*Ip[c]*(R dot V)n } 
    
    + Kr*(Color returned from reflection ray)

    O is the object color (<r> <g> <b>)
    '''
    Ka = scene.ambient * obj.k_a 
    Kd = np.zeros(3)
    Ks = np.zeros(3)
    for light in scene.lightSources:
        L = MV.normalize(light.pos - P)
        # Shadow: find if the point is shadowed or not.
        ray = Ray(P + N * .000001, L)
        l = [sphere.intersectRay(ray) 
        for k, sphere in enumerate(scene.objects) if k != obj_idx]
        if l and min(l) < np.inf:
            continue
        if (obj.isInside(scene.atVector)):
            if (obj.isInside(light.pos)):
                N = -N
            else:
                continue
        
        R = MV.normalize(2 * MV.dot(N, L) * N - L)

        # Lambert shading (diffuse).
        Kd += obj.k_d * max(MV.dot(N, L), 0.)  *  ( light.intensity )
        # Blinn-Phong shading (specular)
        Ks += obj.k_s * ( max(np.dot(R, V), 0.) ** obj.specular ) * ( light.intensity )
    pixelColor = Ka * color + Kd * color + Ks
    return obj, P + N * .000001, N, pixelColor 
'''
=========================================================================================================
=========================================================================================================
=========================================================================================================
'''
parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str, help="input file")

args = parser.parse_args()
if not args.infile:
    print("Need an infile...")
    exit()

lights, spheres, backgroundColor, ambient, near, left, right, bottom, top, res, output_filename = parse_data(args.infile)

h = res[0]
w = res[1]
img = np.full((h, w, 3), backgroundColor)

atVector = MV.vec3((right+left)/2, (top+bottom)/2 , -near) 
camPoint = MV.vec3(0., 0., 0.) 

MAX_DEPTH = 6

scene = Scene(ambient, backgroundColor, camPoint, atVector, spheres, lights)
r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r, 1., 1. / r )
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col = np.zeros(3)
        scene.atVector[:2] = (x, y)
        D = scene.atVector - scene.camPoint
        depth = 0
        ray = Ray(scene.camPoint, D)
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < MAX_DEPTH:
            traced = rayTracer(ray, scene)
            if not traced:
                # check whether the ray if eye-ray or not, 
                # if yes, then return the color of the background
                # if no, it is a reflected ray then return nothing
                if depth == 0:
                    col = scene.backgroundColor.copy()
                break
            obj, M, N, col_ray = traced

            # Calculate reflected color
            ray.origin, ray.direction = M, MV.normalize(ray.direction - 2 * np.dot(ray.direction, N) * N)
            col += reflection * col_ray
            reflection *= obj.k_r

            # If k_r = 0, then do not need to consider reflected ray 
            if reflection == 0:
                break

            depth += 1

        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave(output_filename[:-4] + '.png', img)

ppm.savePpmP6(output_filename, img)