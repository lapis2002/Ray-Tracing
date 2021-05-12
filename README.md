# Ray-Tracing
## Python raytracer

These external packages need to be installed:
<ul>
<li> Pillow is a fork of the PIL package. It provides the Image module for this application to make it easier for saving images.

    pip install pillow=8.1.2

<li> Numpy is a scientific package that helps with mathematical functions.

    pip install numpy=1.20.2
</ul>
Internal module:

    argparse - for parsing file from the command

The system can handle the rendering of ellipsoids, with a fixed camera situated at the origin in a right-handed coordinate system, looking down the negative z-axis. It also handles hollow spheres, which are “cut” open by the near plane as well as lights inside spheres. Local illumination, reflections, and shadows is also implemented.

The program takes a single argument, which is the name of the file to be parsed, and can be run with command:

    > python RayTracer.py testCase1.txt  

## Input File Format

The content and syntax of the file is as follows: <br>
### Content

    The near plane**, left**, right**, top**, and bottom**

    The resolution of the image nColumns* X nRows*

    The position** and scaling** (non-uniform), color***, Ka***, Kd***, Ks***, Kr*** and the specular exponent n* of a sphere

    The position** and intensity*** of a point light source

    The background colour ***

    The scene’s ambient intensity***

    The output file name (this should be limited to 20 characters with no spaces)

### Syntax
    * int         ** float          *** float between 0 and 1

    NEAR <n>

    LEFT <l>

    RIGHT <r>

    BOTTOM <b>

    TOP <t>

    RES <x> <y>

    SPHERE <name> <pos x> <pos y> <pos z> <scl x> <scl y> <scl z> <r> <g> <b> <Ka> <Kd> <Ks> <Kr> <n>

    … // up to 14 additional sphere specifications

    LIGHT <name> <pos x> <pos y> <pos z> <Ir> <Ig> <Ib>

    … // up to 9 additional light specifications

    BACK <r> <g > <b>

    AMBIENT <Ir> <Ig> <Ib>

    OUTPUT <name>

All names should be limited to 20 characters, with no spaces. All fields are separated by spaces. There will be no angle brackets in the input file. The ones above are used to indicate the fields. <br>

    Local illumination model used:
        PIXEL_COLOR[c] = Ka*Ia[c]*O[c] +

    for each point light (p) { 
        Kd*Ip[c]*(N dot L)*O[c] +
        Ks*Ip[c]*(R dot V)n } +
        Kr*(Color returned from reflection ray)

    O is the object color (<r> <g> <b>)
    [c] means that the variable has three different color component, so the value may vary depending on whether the red, green, or blue color chanel is being calculated

    The maximum number of reflection ray is 3.

![Ray Tracer](/TestCases/rt.png "Ray Tracer")
