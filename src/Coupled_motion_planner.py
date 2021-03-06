# Importing libraries

from time import time
import numpy as np
from numpy import array
from numpy import dot
import math
from scipy import signal
from scipy import ndimage
from scipy import interpolate
import sys
import cv2
import FastMarching.FastMarching as FM
import FastMarching.FastMarching3D as FM3D

# Constants
epsilon = sys.float_info.epsilon



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # # FUNCTIONS
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def surface_normal(resolution, size, z):

    # Obtain surface normals in every point of a meshgrid

    xm = np.linspace(0, size, int(round(size/resolution)))
    ym = np.linspace(0, size, int(round(size/resolution)))
    x,y = np.meshgrid(xm, ym)
    xx = x
    yy = y
    zz = z

    m,n = x.shape
    m = int(m)
    n = int(n)

    stencil1 = array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])/2
    stencil2 = array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])/2

    xx = np.vstack((3*xx[0, :] - 3*xx[1, :] + xx[2, :], xx, 3*xx[m - 1, :] - 3*xx[m - 2, :] + xx[m - 3, :]))
    xx = np.hstack((array([3*xx[:, 0] - 3*xx[:, 1] + xx[:, 2]]).T, xx, array([3*xx[:, n - 1] - 3*xx[:, n - 2] + xx[:, n - 3]]).T))
    yy = np.vstack((3*yy[0, :] - 3*yy[1, :] + yy[2, :], yy, 3*yy[m - 1, :] - 3*yy[m - 2, :] + yy[m - 3, :]))
    yy = np.hstack((array([3*yy[:, 0] - 3*yy[:, 1] + yy[:, 2]]).T, yy, array([3*yy[:, n - 1] - 3*yy[:, n - 2] + yy[:, n - 3]]).T))
    zz = np.vstack((3*zz[0, :] - 3*zz[1, :] + zz[2, :], zz, 3*zz[m - 1, :] - 3*zz[m - 2, :] + zz[m - 3, :]))
    zz = np.hstack((array([3*zz[:, 0] - 3*zz[:, 1] + zz[:, 2]]).T, zz, array([3*zz[:, n - 1] - 3*zz[:, n - 2] + zz[:, n - 3]]).T))

    ax = -signal.convolve2d(xx, np.flipud(stencil1), mode = 'valid')
    ay = -signal.convolve2d(yy, np.flipud(stencil1), mode = 'valid')
    az = -signal.convolve2d(zz, np.flipud(stencil1), mode = 'valid')

    bx = signal.convolve2d(xx, np.flipud(stencil2), mode = 'valid')
    by = signal.convolve2d(yy, np.flipud(stencil2), mode = 'valid')
    bz = signal.convolve2d(zz, np.flipud(stencil2), mode = 'valid')

    nx = -(ay*bz - az*by)
    ny = -(az*bx - ax*bz)
    nz = -(ax*by - ay*bx)

    mag = np.sqrt(nx*nx + ny*ny + nz*nz)
    mag[np.where(mag == 0)] = epsilon
    nxout = nx/mag
    nyout = ny/mag
    nzout = nz/mag

    return nxout, nyout, nzout

def image_filling(im):

    # Fill gaps in an image (uint8)

    h,w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_floodfill = im.copy()
    cv2.floodFill(im_floodfill, mask, (0, 0), 1)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill) - 254

    im_out = im|im_floodfill_inv

    return im_out

def structural_disk(r):

    # Create a binary circle of a certain radius

    se = np.zeros((2*r + 1, 2*r + 1), np.uint8)
    for i in range(0, 2*r + 1):
        for j in range (0, 2*r + 1):
            d = math.sqrt((r - i)**2 + (r - j)**2)
            if d <= r:
                se[i][j] = 1

    return se

def trans_rot(pos, ang):

    # Create a trasnformation matrix in function of a translation and a rotation 

    xb = pos[0]
    yb = pos[1]
    zb = pos[2]

    alpha = ang[2]
    beta = ang[1]
    gamma = ang[0]

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    Tr = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, xb],
          [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, yb],
          [-sb,   cb*sg,            cb*cg,            zb],
          [0,     0,                0,                1]]

    return Tr

def posTransform(initialPosition, heading, distance):

    # Obtains the cartesian position in function of the distance and heading to a reference frame 

    Tob = trans_rot(initialPosition, heading)

    Tbc = trans_rot(distance, [0, 0, 0])

    Toc = dot(Tob, Tbc)

    finalPosition = [Toc[0][3], Toc[1][3], Toc[2][3]]

    return finalPosition

def distTransform(initialPosition, heading, finalPosition):

    # Obtains the distance between reference frames in function of their position and heading 

    Tob = trans_rot(initialPosition, heading)

    Tof = trans_rot(finalPosition, [0, 0, 0])

    Tbf = dot(np.linalg.inv(Tob), Tof)

    distance = [Tbf[0,3], Tbf[1,3], Tbf[2,3]]

    return distance

def basePos2wheelsPos(basePosition, heading, Zs, resolution):

    # Returns wheels position in function of the arm base position and heading

    Tob = trans_rot(basePosition,heading)

    Tbw1 = [[1, 0, 0, 0.14],
            [0, 1, 0, 0.51],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tbw2 = [[1, 0, 0, -0.2],
            [0, 1, 0, 0.51],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tbw3 = [[1, 0, 0, -0.54],
            [0, 1, 0, 0.51],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tbw4 = [[1, 0, 0, -0.54],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tbw5 = [[1, 0, 0, -0.2],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tbw6 = [[1, 0, 0, 0.14],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

    Tow1 = dot(Tob, Tbw1)
    Tow2 = dot(Tob, Tbw2)
    Tow3 = dot(Tob, Tbw3)
    Tow4 = dot(Tob, Tbw4)
    Tow5 = dot(Tob, Tbw5)
    Tow6 = dot(Tob, Tbw6)

    Tow1[2][3] = Zs[int(round(Tow1[1][3]/resolution)), int(round(Tow1[0][3]/resolution))]
    Tow2[2][3] = Zs[int(round(Tow2[1][3]/resolution)), int(round(Tow2[0][3]/resolution))]
    Tow3[2][3] = Zs[int(round(Tow3[1][3]/resolution)), int(round(Tow3[0][3]/resolution))]
    Tow4[2][3] = Zs[int(round(Tow4[1][3]/resolution)), int(round(Tow4[0][3]/resolution))]
    Tow5[2][3] = Zs[int(round(Tow5[1][3]/resolution)), int(round(Tow5[0][3]/resolution))]
    Tow6[2][3] = Zs[int(round(Tow6[1][3]/resolution)), int(round(Tow6[0][3]/resolution))]

    wheel1Pos = [Tow1[0][3], Tow1[1][3], Tow1[2][3]]
    wheel2Pos = [Tow2[0][3], Tow2[1][3], Tow2[2][3]]
    wheel3Pos = [Tow3[0][3], Tow3[1][3], Tow3[2][3]]
    wheel4Pos = [Tow4[0][3], Tow4[1][3], Tow4[2][3]]
    wheel5Pos = [Tow5[0][3], Tow5[1][3], Tow5[2][3]]
    wheel6Pos = [Tow6[0][3], Tow6[1][3], Tow6[2][3]]

    return wheel1Pos, wheel2Pos, wheel3Pos, wheel4Pos, wheel5Pos, wheel6Pos

def wheelSlope(wheel1Pos, wheel2Pos, wheel3Pos, wheel4Pos, wheel5Pos, wheel6Pos):

    # Obtains the rover inclination in function of the position of the wheels

    x1 = wheel1Pos[0]
    x2 = wheel2Pos[0]
    x3 = wheel3Pos[0]
    x4 = wheel4Pos[0]
    x5 = wheel5Pos[0]
    x6 = wheel6Pos[0]

    y1 = wheel1Pos[1]
    y2 = wheel2Pos[1]
    y3 = wheel3Pos[1]
    y4 = wheel4Pos[1]
    y5 = wheel5Pos[1]
    y6 = wheel6Pos[1]

    z1 = wheel1Pos[2]
    z2 = wheel2Pos[2]
    z3 = wheel3Pos[2]
    z4 = wheel4Pos[2]
    z5 = wheel5Pos[2]
    z6 = wheel6Pos[2]

    xa = (x1 + x6)/2
    xb = (x5 + x4)/2
    xc = (x3 + x2)/2

    ya = (y1 + y6)/2
    yb = (y5 + y4)/2
    yc = (y3 + y2)/2

    za = (z1 + z6)/2
    zb = (z5 + z4)/2
    zc = (z3 + z2)/2

    B = ((za - zc)*(ya - yb) - (za - zb)*(ya - yc))/((xa - xc)*(ya - yb) - (xa - xb)*(ya - yc))
    A = ((za - zb) - B*(xa - xb))/(ya - yb)

    if za >= (zb + zc)/2:
        B = -B

    return A, B

def roverHeading(basePath, heading, resolution, Zs):

    # Function that obtains the heading of the rover (pitch, yaw, roll) for each waypoint
    # of the path in function of the position of the wheels

    Rx = np.zeros(heading.shape)
    Ry = np.zeros(heading.shape)
    Rz = np.zeros(heading.shape)

    Rz = heading

    for i in range(1, len(basePath)):
        w1Pos, w2Pos, w3Pos, w4Pos, w5Pos, w6Pos = basePos2wheelsPos(basePath[i, :], [Rx[i - 1], Ry[i - 1], Rz[i]], Zs, resolution)

        A, B = wheelSlope(w1Pos, w2Pos, w3Pos, w4Pos, w5Pos, w6Pos)
        Rx[i] = np.tan(A)
        Ry[i] = np.tan(B)

    rHeading = np.vstack((Rx, Ry, Rz)).T

    return rHeading

def deploymentStage(qid, qend, index, qinitial):

    # Function that returns a vector with the configurations of the arm during the deployment stage

    qid[0] = math.pi*0.99999/2

    # Initialazing the joint's positions matrix in the deployment stage
    joints = np.zeros((index, len(qinitial)))

    # Final and initial configurations of the joints
    joints[0, :] = qinitial
    joints[-1, :] = qid

    # The deployment takes place linearly from the initial to the final
    # configurations, clockwise or counter clock wise, trying to travel the
    # shortest angle.
    # Clockwise -> Decrease (Negative inc)
    # Counter clockwise -> Increase (Positive inc)
    inc = np.ones((len(qinitial), 1))

    for i in range(0, len(qinitial)):
        angleDiff = qid[i] - qinitial[i]
        inc[i] = angleDiff/(index - 1)

    for i in range(1, index - 1):
        joints[i, :] = (inc.T + joints[i - 1, :])

    return joints


def GetObstMap(ZsMap, resX, resY, resZ, sX, sY, sZ, newObstMap, xm, ym):

    # Set cost map of obstacles and ground

    obstMap = np.ones((sX, sY, sZ))
    groundMap =  np.ones((sX, sY, sZ))

    m, n = ZsMap.shape

    # For each position in the map
    for i in range(0, n):
        for j in range(0, m):
            if resX*i != xm and resY*j != ym:

                # If it is not the sample position, we obtain the Z index
                iz = int(round(ZsMap[j, i]/resZ))
                if i < sX and j < sY and iz < sZ:

                    # If the node is inside the map...
                    if newObstMap[j, i] == 1:

                        # If it is an obstacle, we asign it to the obstacles map
                        obstMap[j, i, iz] = np.inf

                    else:

                        # If it is just ground, we asign it to the ground map
                        groundMap[j, i, iz] = np.inf

    finalMap = obstMap + groundMap

    # Edge point's cost must be infinite
    finalMap[:, 0, :] = np.inf
    finalMap[:, -1, :] = np.inf
    finalMap[0, :, :] = np.inf
    finalMap[-1, :, :] = np.inf
    finalMap[:, :, 0] = np.inf
    finalMap[:, :, -1] = np.inf

    return finalMap, obstMap, groundMap

def trans2pos(T, elbow):

    # Obtaining the vector of orientation (Euler ZYZ) and position in
    # function of the elbow configuration and a transformation matrix

    # Positions
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    # Orientations in Euler ZYZ angles
    cbeta = T[2, 2]
    sbeta1 = math.sqrt(T[0, 2]**2 + T[1, 2]**2)
    sbeta2 = -math.sqrt(T[0, 2]**2 + T[1, 2]**2)

    beta1 = math.atan2(sbeta1, cbeta)
    beta2 = math.atan2(sbeta2, cbeta)

    # Choosing configuration in function of the elbow
    if elbow == 1:
        beta = beta1
        sbeta = sbeta1
    else:
        beta = beta2
        sbeta = sbeta2

    if sbeta == 0:
        if cbeta == 1:
            alpha = 0
            gamma = math.atan2(T[1, 0], T[1, 0]) - alpha
        elif cbeta == -1:
            alpha = 0
            gamma = math.atan2(T[0, 1], T[1, 1]) + alpha
    else:
        salpha = T[1, 2]/sbeta
        calpha = T[0, 2]/sbeta
        sgamma = T[2, 1]/sbeta
        cgamma = -T[2, 0]/sbeta

        alpha = math.atan2(salpha, calpha)
        gamma = math.atan2(sgamma, cgamma)

    # Saving final position-orientation vector
    r = [x, y, z, alpha, beta, gamma]

    return r

def MCD(joints):

    # Direct kinematics model of the 5 DOF manipulator

    # Saving joint values
    theta1 = joints[0]
    theta2 = joints[1]
    theta3 = joints[2]
    theta4 = joints[3]
    theta5 = joints[4]

    # Manipulator parameters
    d1 = 0.0895
    a2 = 0.206
    a3 = 0.176
    d5 = 0.0555

    # Generating transformation matrices for each joint
    T01 = dot(traslation([0, 0, d1]), dot(rotZ(theta1), dot(traslation([0, 0, 0]), rotX(-math.pi/2))))
    T12 = dot(traslation([0, 0, 0]), dot(rotZ(theta2), dot(traslation([a2, 0, 0]), rotX(0))))
    T23 = dot(traslation([0, 0, 0]), dot(rotZ(theta3), dot(traslation([a3, 0, 0]), rotX(0))))
    T34 = dot(traslation([0, 0, 0]), dot(rotZ(theta4), dot(traslation([0, 0, 0]), rotX(math.pi/2))))
    T45 = dot(traslation([0, 0, d5]), dot(rotZ(theta5), dot(traslation([0, 0, 0]), rotX(0))))

    # The transformation matrix from base to end-effector
    T = dot(T01, np.dot(T12, dot(T23, dot(T34, T45))))

    # Filtering very low values as zeros
    m, n = T.shape
    for j in range(0, m):
        for i in range(0, n):
            if abs(T[j, i]) < 1e-4:
                    T[j, i] = 0

    return T

def traslation(p):

    # Traslation transformation matrix

    D = [[1, 0, 0, p[0]],
         [0, 1, 0, p[1]],
         [0, 0, 1, p[2]],
         [0, 0, 0, 1]]

    return D

def rotZ(theta):

    # Z rotation transformation matrix

    s = math.sin(theta)
    c = math.cos(theta)

    if abs(c) < 0.000000001: c = 0
    if abs(s) < 0.000000001: s = 0

    RZ = [[c, -s, 0, 0],
          [s, c,  0, 0],
          [0, 0,  1, 0],
          [0, 0,  0, 1]]

    return RZ

def rotY(theta):

    # Y rotation transformation matrix

    s = math.sin(theta)
    c = math.cos(theta)

    if abs(c) < 0.000000001: c = 0
    if abs(s) < 0.000000001: s = 0

    RY = [[c,  0, s, 0],
          [0,  1, 0, 0],
          [-s, 0, c, 0],
          [0,  0, 0, 1]]

    return RY

def rotX(theta):

    # X rotation transformation matrix

    s = math.sin(theta)
    c = math.cos(theta)

    if abs(c) < 0.000000001: c = 0
    if abs(s) < 0.000000001: s = 0

    RX = [[1, 0, 0,  0],
          [0, c, -s, 0],
          [0, s, c,  0],
          [0, 0, 0,  1]]

    return RX

def TunnelCost(rlim, rO, rm, gamma2D, sX, sY, sZ, resX, resY, resZ, finalBaseHeading, finalWayPointArm, initialWayPointArm):

    # Generate tunnel extruding a circle

    # Asign a cost to each point of the map
    Cmap = 10*np.ones((sY, sX, sZ))

    # Gradient of the cost inside the tunnel
    gradient = 15

    # Tunnel characteristics
    tunnelRad = rlim + 2*resX
    tunnelSizeX = int(round(2*tunnelRad/resX) + 1)
    tunnelSizeZ = int(round(2*tunnelRad/resZ) + 1)

    # Inside tunnel a gradual cost is asigned

    m, n = gamma2D.shape

    for j in range(0, m):

        # Orientation of the arm's base
        alpha = finalBaseHeading[j, 2] - math.pi/2 	# Rz
        beta = finalBaseHeading[j, 1]			# Ry
        gamma = finalBaseHeading[j, 0] 		 	# Rx

        ca = math.cos(alpha)
        cb = math.cos(beta)
        cg = math.cos(gamma)
        sa = math.sin(alpha)
        sb = math.sin(beta)
        sg = math.sin(gamma)

        # Transformation from the origin to the arm's base
        Toa = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, gamma2D[j, 0]],
               [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, gamma2D[j, 1]],
               [-sb,   cb*sg,            cb*cg,            gamma2D[j, 2]],
               [0,     0,                0,                1]]

        # For every point inside the tunnel
        for i in np.linspace(-tunnelRad, tunnelRad, tunnelSizeX, endpoint=True):
            for k in np.linspace(-tunnelRad, tunnelRad, tunnelSizeZ, endpoint=True):

                # Transformation from the arm's base to the point
                Tap = [[1, 0, 0, i],
                       [0, 1, 0, 0],
                       [0, 0, 1, k],
                       [0, 0, 0, 1]]

                # Transformation from the origin to the point
                Top = dot(Toa, Tap)

                # X-Y-Z indeces of the point
                ix = int(round(Top[0][3]/resX))
                iy = int(round(Top[1][3]/resY))
                iz = int(round(Top[2][3]/resZ))
                norm = math.sqrt(i**2 + k**2)

                # If the point is inside the map
                if ix >= 0 and iy >= 0 and iz >= 0 and ix < sX and iy < sY and iz < sZ:

                    # If the point is inside the reachability area
                    if norm < rlim:

                        # If the point has not been assigned a cost
                        if Cmap[iy, ix, iz] == 10:

                            # A cost in function of distance to optimal radius
                            Cmap[iy, ix, iz] = gradient*(norm - (rO + rm)/2)**2 + 2 + 4*(i + rlim + 2*resZ)

                    else:

                        # If the point is not inside the reachability area, infinite cost
                        # Also ensuring this point is not the sample point
                        if (ix != finalWayPointArm[0] or iy != finalWayPointArm[1] or iz != finalWayPointArm[2]) and (ix != initialWayPointArm[0] or iy != initialWayPointArm[1] or iz != initialWayPointArm[2]):
                            Cmap[iy, ix, iz] = np.inf

                # We repeat the algorithm for the neighbour forward points, so
                # we ensure every point is touched at least once
                Tap = [[1, 0, 0, i],
                       [0, 1, 0, resY],
                       [0, 0, 1, k],
                       [0, 0, 0, 1]]

                Top = dot(Toa, Tap)

                ix = int(round(Top[0][3]/resX))
                iy = int(round(Top[1][3]/resY))
                iz = int(round(Top[2][3]/resZ))

                if ix >= 0 and iy >= 0 and iz >= 0 and ix < sX and iy < sY and iz < sZ:
                    if Cmap[iy, ix, iz] == 10:
                        if norm < rlim:
                            Cmap[iy, ix, iz] = gradient*(norm - (rO + rm)/2)**2 + 2 + 4*(i + rlim + 2*resZ)

    # We repeat the previous algorithm but just in the first position of the
    # arm's base path, closing the tunnel so the computation will be
    # significantly reduced

    # Orientation of the arm's base
    alpha = finalBaseHeading[0, 2] - math.pi/2 	# Rz
    beta = finalBaseHeading[0, 1]		# Ry
    gamma = finalBaseHeading[0, 0] 		# Rx

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    # Transformation from the origin to the arm's base
    Toa = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, gamma2D[0, 0]],
           [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, gamma2D[0, 1]],
           [-sb,   cb*sg,            cb*cg,            gamma2D[0, 2]],
           [0,     0,                0,                1]]

    # For every point inside the tunnel
    for i in np.linspace(-tunnelRad, tunnelRad, tunnelSizeX, endpoint=True):
        for k in np.linspace(-tunnelRad, tunnelRad, tunnelSizeZ, endpoint=True):
            Tap = [[1, 0, 0, i],
                   [0, 1, 0, -resY],
                   [0, 0, 1, k],
                   [0, 0, 0, 1]]

            Top = dot(Toa, Tap)
            ix = int(round(Top[0, 3]/resX))
            iy = int(round(Top[1, 3]/resY))
            iz = int(round(Top[2, 3]/resZ))

            norm = math.sqrt(i**2 + k**2)

            if ix >= 0 and iy >= 0 and iz >= 0 and ix < sX and iy < sY and iz < sZ:

                # If the point is inside the reachability area
                if norm < rlim:
                    if (ix != finalWayPointArm[0] or iy != finalWayPointArm[1] or iz != finalWayPointArm[2]) and (ix != initialWayPointArm[0] or iy != initialWayPointArm[1] or iz != initialWayPointArm[2]):
                        Cmap[iy, ix, iz] = np.inf

    # We extrude a semi-sphere in the last position of the
    # arm's base path, closing the tunnel so the computation will be
    # significantly reduced

    # Orientation of the arm's base
    m, n = finalBaseHeading.shape
    alpha = finalBaseHeading[m-1, 2]		# Rz
    beta = finalBaseHeading[m-1, 1] 		# Ry
    gamma = finalBaseHeading[m-1, 0] 		# Rx

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    m, n = gamma2D.shape

    # Transformation from the origin to the arm's base
    Toa = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, gamma2D[m-1, 0]],
           [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, gamma2D[m-1, 1]],
           [-sb,   cb*sg,            cb*cg,            gamma2D[m-1,2]],
           [0,     0,                0,                1]]

    # For angles between -100º and 100º, ensuring a closed tunnel
    for i in range(-100, 100, 2):
        theta = math.pi*i/180

        # For angles between -90º and 90º, ensuring a closed tunnel
        for j in range(-90, 90,2):
            sigma = math.pi*j/180

            # For radius between 0 and rlim
            for k in np.linspace(0, tunnelRad, round(tunnelSizeZ/2) + 1, endpoint=True):
                ct = math.cos(theta)
                st = math.sin(theta)
                cs = math.cos(sigma)
                ss = math.sin(sigma)

                # From arm's base to workspace sphere point
                Tas = [[1, 0, 0, k*ct*cs],
                       [0, 1, 0, k*ct*ss],
                       [0, 0, 1, k*st],
                       [0, 0, 0, 1]]

                # From global system reference to workspace sphere point
                Tos= dot(Toa, Tas)

                # Workspace sphere point position
                ix = int(round(Tos[0, 3]/resX))
                iy = int(round(Tos[1, 3]/resY))
                iz = int(round(Tos[2, 3]/resZ))

                if ix >= 0 and iy >= 0 and iz >= 0 and ix < sX and iy < sY and iz < sZ:
                    if Cmap[iy, ix, iz] == 10:
                        Cmap[iy, ix, iz] = gradient*(k - (rO + rm)/2)**2 + 2

            ct = math.cos(theta)
            st = math.sin(theta)
            cs = math.cos(sigma)
            ss = math.sin(sigma)

            # From arm's base to workspace sphere point
            Tas = [[1, 0, 0, (rlim + 2*resZ)*ct*cs],
                   [0, 1, 0, (rlim + 2*resZ)*ct*ss],
                   [0, 0, 1, (rlim + 2*resZ)*st],
                   [0, 0, 0, 1]]

            # From global system reference to workspace sphere point
            Tos= dot(Toa, Tas)

            # Workspace sphere point position
            ix = int(round(Tos[0, 3]/resX))
            iy = int(round(Tos[1, 3]/resY))
            iz = int(round(Tos[2, 3]/resZ))

            if ix >= 0 and iy >= 0 and iz >= 0 and ix < sX and iy < sY and iz < sZ:
                if (ix != finalWayPointArm[0] or iy != finalWayPointArm[1] or iz != finalWayPointArm[2]) and (ix != initialWayPointArm[0] or iy != initialWayPointArm[1] or iz != initialWayPointArm[2]):
                    Cmap[iy, ix, iz] = np.inf

    return Cmap

def MCI(position, orientation, elbow):

    # Variable to differentiate if the objective is reachable
    unreachable = 0

    # Parameters of the arm
    d1 = 0.0895
    a2 = 0.206
    a3 = 0.176
    d5 = 0.0555

    # Objective's position
    x = position[0]
    y = position[1]
    z = position[2]

    # X-Y distance to the objective
    r = math.sqrt(x**2 + y**2)


    # If the objective is just in the Z-axis, any angle for the first joint is viable
    if r == 0: theta1 = orientation[0]
    # If it is not, then the angle is a certain one
    else: theta1 = math.atan2(y/r, x/r)

    # The Y Euler angle is an argument
    alpha = orientation[1]

    # Position of the wrist
    rw = r - d5*math.sin(alpha)
    zw = z - d5*math.cos(alpha)


    # Distance between the second joint axis and the wrist
    l = math.sqrt(rw**2 + (zw - d1)**2)

    # Several angles which define the configuration
    try:
	    beta = math.atan2((zw - d1)/l, rw/l)
	    gamma = np.abs(math.acos((a2**2 + l**2 - a3**2)/(2*a2*l)))
	    delta = np.abs(math.acos((a2**2 + a3**2 - l**2)/(2*a2*a3)))

    except:
	    beta = math.pi
	    gamma = math.pi
	    delta = math.pi
	    unreachable = 1

    # Choosing the configuration in function of the elbow
    if elbow == 1:
        theta2 = 2*math.pi - (beta + gamma)
        theta3 = 2*math.pi - (math.pi + delta)
    else:
        theta2 = gamma - beta
        theta3 = 2*math.pi - (math.pi - delta)

    # Wrist joint is obtained in function of the position ones
    theta4 = (alpha - theta2 - theta3)

    # Last joint angle is just the last Z Euler angle
    theta5 = orientation[2]


    # Robustness

    # Generating transformation matrices for each joint
    T01 = dot(traslation([0, 0, d1]), dot(rotZ(theta1), dot(traslation([0, 0, 0]), rotX(-math.pi/2))))
    T12 = dot(traslation([0, 0, 0]), dot(rotZ(theta2), dot(traslation([a2, 0, 0]), rotX(0))))
    T23 = dot(traslation([0, 0, 0]), dot(rotZ(theta3), dot(traslation([a3, 0, 0]), rotX(0))))
    T34 = dot(traslation([0, 0, 0]), dot(rotZ(theta4), dot(traslation([0, 0, 0]), rotX(math.pi/2))))
    T45 = dot(traslation([0, 0, d5]), dot(rotZ(theta5), dot(traslation([0, 0, 0]), rotX(0))))

    T02 = dot(T01, T12)
    T03 = dot(T02, T23)
    T04 = dot(T03, T34)
    T05 = dot(T04, T45)

    # General facts

    # If this distance is greater than the maximum possible one
    if l > (a2 + a3):
        unreachable = 1
        print('The wrist position is too far')

    # If this distance is lower than the minimum possible one
    elif l < (a2 - a3):
        unreachable = 1
        print('The wrist position is too close')

    # First joint facts

    # If the objective is very close to the first joint
    elif (np.abs(x) <= 0.01) and (np.abs(y) <= 0.01) and (z <= d1) + 0.01:
        unreachable = 1
        print('The objective is very close to the first joint')

    # Checking first joint limits
    elif theta1 < math.pi/2 - 100*math.pi/180 or theta1 > math.pi/2 + 20*math.pi/180:
        unreachable = 1
        print('The first joint limit has been reached, theta 1:' + str(theta1))

    # Second joint facts

    # If the second joint is touching the rovers body
    elif T02[2][3] <= 0:
        unreachable = 1
        print('The second joint will crash with the rovers body')

    # If the second joint is touching the ground
    elif T02[0][3] >= 0.22:
        unreachable = 1
        print('The second joint will crash with the ground')

    # If the second joint is crashing with the wheel
    elif (T02[2][3] <= 0.21) and (T02[1][3] <= -0.17):
        unreachable = 1
        print('The second joint will crash with the right-front wheel')

    # Third joint facts

    # If the third joint is touching the rovers body
    elif T03[2][3] <= 0:
        unreachable = 1
        print('The third joint will crash with the rovers body')

    #If the third joint is touching the ground
    elif T03[0][3] >= 0.22:
        unreachable = 1
        print('The third joint will crash with the ground')

    # If the third joint is crashing with the wheel
    elif (T03[2][3] <= 0.11) and (T03[1][3] <= -0.05):
        unreachable = 1
        print('The third joint will crash with the right-front wheel')

    # If the third joint is crashing with the other wheel
    elif (T03[2][3] <= 0.11) and (T03[1][3] >= 0.35):
        unreachable = 1
        print('The third joint will crash with the left-front wheel')

    # Fourth joint facts

    # If the wrist is too close to the first joint
    elif (np.abs(rw) <= 0.01) and (zw <= d1 + 0.01):
        unreachable = 1
        print('Wrist will crash with the first joint')

    # Fifth joint facts

    # If the fifth joint is touching the rovers body
    elif T05[2][3] <= 0:
        unreachable = 1
        print('The fifth joint will crash with the rovers body')

    # If the fifth joint is touching the ground
    elif T05[0][3] >= 0.22:
        unreachable = 1
        print('The fifth joint will crash with the ground')


    # If the fifth joint is crashing with the wheel
    elif (T05[2][3] <= 0.11) and (T05[1][3] <= -0.05):
        unreachable = 1
        print('The fifth joint will crash with the right-front wheel')


    # If the fifth joint is crashing with the other wheel
    elif (T05[2][3] <= 0.11) and (T05[1][3] >= 0.35):
        unreachable = 1
        print('The fifth joint will crash with the left-front wheel')

    # If the objective is reachable, we save the configuration
    if unreachable == 0:
        joints = [theta1, theta2, theta3, theta4, theta5]
    else:
        joints = [np.nan, np.nan, np.nan, np.nan, np.nan]

    return joints

def assign(gamma, path, rO, Rm, Rlim):
    
    # Function to relate end effector and arm base paths in function of
    # the characteristics of the arm
   
    assignment1 = -1*np.ones([len(gamma)])
    assignment2 = -1*np.ones([len(gamma)])
    assignment3 = -1*np.ones([len(gamma)])

    # Comfortable configurations for the arm are checked first
    for i in range(len(path) - 1, -1, -1):
        for j in range(0, len(gamma)):
            if np.linalg.norm(gamma[j, :] - path[i, :]) < (rO + Rm)/2:
                assignment1[j] = i

    # If some of the waypoints are further, more extended configs for the arm are also considered
    for i in range(len(path) - 1, -1, -1):
        for j in range(0, len(gamma)):
            if np.linalg.norm(gamma[j, :] - path[i, :]) < Rm:
                assignment2[j] = i

    # If there is still some waypoints unassigned, maybe we need the arm to be fully extended
    for i in range(len(path) - 1, -1, -1):
        for j in range(0, len(gamma)):
            if np.linalg.norm(gamma[j, :] - path[i, :]) < Rlim:
                assignment3[j] = i

    assignment = assignment1
    for i in range(0, len(assignment)):
        if assignment[i] == -1:
            assignment[i] = assignment2[i]

        if assignment[i] == -1:
            assignment[i] = assignment3[i]

    # Ensuring first and last configs are correct
    assignment[0] = 0
    assignment[-1] = len(path) - 1


    # Ensuring the assignment vector is always growing
    for i in range(len(assignment) - 1, 0, -1):
        if assignment[i] == -1:
            assignment[i] = len(path) - 1
        if assignment[i] < assignment[i - 1]:
            assignment[i - 1] = assignment[i]

    return np.uint32(assignment)


def PathInverseKinematics(assignment, gamma3D, finalBasePath, finalBaseHeading, qi, Rini, elbow, ryf):

    # Function to obtain the configuration of the arm at every waypoint of the paths using the inverse kinematics model
    # of the manipulator

    # Initial Y Euler rotation of the end effector
    ryi = Rini[4] % 2*math.pi

    # Initialization of the vector which saves all the configs of the arm
    joints = np.zeros([len(gamma3D), len(qi)])
    joints[0, :] = qi

    # For every waypoint in the path
    for i in range(1, len(gamma3D)):

        # First of all, we obtain the distance and oriention of the end effector reference frame with respect to the arm base reference frame
        xv = finalBasePath[assignment[i], 0]
        yv = finalBasePath[assignment[i], 1]
        zv = finalBasePath[assignment[i], 2]

        alphab = finalBaseHeading[assignment[i], 2]
        betab = math.pi/2 + finalBaseHeading[assignment[i], 1]
        gammab = finalBaseHeading[assignment[i], 0]

        ca = math.cos(alphab)
        cb = math.cos(betab)
        cg = math.cos(gammab)
        sa = math.sin(alphab)
        sb = math.sin(betab)
        sg = math.sin(gammab)

        Tob = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, xv],
               [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, yv],
               [-sb,   cb*sg,            cb*cg,            zv],
               [0,     0,                0,                1]]

        xp = gamma3D[i, 0]
        yp = gamma3D[i, 1]
        zp = gamma3D[i, 2]

        Top = [[1, 0, 0, xp],
               [0, 1, 0, yp],
               [0, 0, 1, zp],
               [0, 0, 0, 1]]

        Tbp =  dot(np.linalg.inv(Tob), Top)

        xc = Tbp[0][3]
        yc = Tbp[1][3]
        zc = Tbp[2][3]

        rz1 = math.atan2(yc, xc)
        ry = ((i - 1)/(len(gamma3D) - 1))*(ryf - ryi) + ryi
        rz2 = 0

        # The inverse kinematics will provide the config of the arm
        joints[i, :] = MCI([xc, yc, zc], [rz1, ry, rz2], elbow)

        # Ensuring position valuesfor the joints are correct
        unreachable = 0
        for j in range(0,len(qi)):
            if joints[i, j] > math.pi:
                joints[i, j] = joints[i, j] - 2*math.pi
            elif joints[i, j] < -math.pi:
                joints[i, j] = joints[i, j] + 2*math.pi
            if np.isnan(joints[i, j]):
                unreachable = 1

        # If the config is not viiable, we maintain the previous one
        if unreachable == 1:
            print('Not reachable position and orientation of the arm at waypoint ' + str(i))
            if i == (len(gamma3D) - 1):
                print('The sample waypoint is not reachable')
                joints[:, :] = np.nan
                break
            joints[i, :] = joints[i - 1, :]

    return joints

def angularSmooth(angles):

    # Function that removes jumps between 0 and 2pi and viceversa,
    # making continuous any angular vector

    m, n = angles.shape
    for i in range(1, m):
        for j in range(0, n):
            if angles[i, j] - angles[i-1, j] > 5:
                if i == m:
                    angles[-1, j] = angles[-1, j] - 2*math.pi
                else:
                    angles[range(i, m), j] = angles[range(i, m), j] - 2*math.pi

            elif angles[i, j] - angles[i-1, j] < -5:
                if i == m:
                    angles[-1, j] = angles[-1, j] + 2*math.pi
                else:
                    angles[range(i, m), j] = angles[range(i, m), j] + 2*math.pi

    return angles

def smoothJoints(joints):

    # Smoothing function for several angular vectors

    m, n = joints.shape

    # Removing jumps in the vector
    ajoints = angularSmooth(joints)
    fJoints = np.zeros_like(ajoints)

    # Smoothing algorithm with a gaussian filter
    for i in range(0, n):
        fJoints[:, i] = ndimage.gaussian_filter1d(ajoints[:, i], 4)

    # Ensuring the initial and final configuration are still the same
    fJoints[0, :] = ajoints[0, :]
    fJoints[-1, :] = ajoints[-1, :]


    return fJoints
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # MAIN
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def main(xm, ym, xr, yr, initialHeading, mapDirectory, resolution, size):      
		            
    # Initial execution time
    startTime = time()

    # Loading DEM 
    with open(str(mapDirectory) + 'PRL_DEM.txt','r') as file:
        Zs = array([[float(num) for num in line.split(',')] for line in file])
    
    Zs = Zs - np.min(Zs)

    # Obtaining normal vectors to the DEM
    [Nx, Ny, Nz] = surface_normal(resolution, size, Zs)

    # Scene data
    pxm = int(round(xm/resolution - 1))
    pym = int(round(ym/resolution - 1))
    zm = Zs[pym, pxm]
    sampleNode = [pxm, pym]

    # Rover data
    roverPos = [xr, yr]
    qinitial = [math.pi/2, 0, -2.21, 0, 0]
    rxm = int(round(xr/resolution - 1))
    rym = int(round(yr/resolution - 1))
    roverNode = [rxm, rym]

    # Rover arm parameters
    Rm = 0.4241   	  	# Maximum optimal radius
    rm = 0.1105     		# Minimum optimal radius
    rO = (Rm + rm)/2  		# Optimal radius
    Rlim = 0.527    		# Maximum reachability radius
    rlim = 0.05     		# Minimum reachability radius

    # Rover configurable parameters

    xa = 0.26       		# X axes distance from rover reference system to arm base
    ya = -0.15      		# Y axes distance from rover reference system to arm base
    za = 0.16       		# Heigth from rover reference system to arm base
    zp = 0.07       		# Heigth from floor to rover reference system
    zbase = 0.23    		# Heigth from floor to arm base
    diagonal = 0.9  		# Diagonal length of the rover body
    v = 0.1         		# Rover speed

    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     STEP 1: Rover path planning to reach de sample
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # Slope matrix in every node
    slope = np.arccos(Nz)

    # =============================================================================
    #     Obstacle and cost map obtention
    # =============================================================================

    obstMap = np.zeros(Zs.shape)

    # Slope threshold to recognize obstacles
    obstMap[slope > 0.20] = 1  # 0.419 for ET-UMA

    # Map limits are not considered as obstacles yet to allow next image processing steps
    obstMap[0, :] = 0
    obstMap[-1, :] = 0
    obstMap[:, 0] = 0
    obstMap[:, -1] = 0

    # Filling gaps
    obstMap = np.uint8(obstMap)
    obstMap = image_filling(obstMap)

    # Eroding small obstacles
    se = structural_disk(10)
    obstMap = cv2.erode(obstMap, se, iterations = 1)
    obstMap = cv2.dilate(obstMap, se, iterations = 1)

    # Eroding narrow corridors
    obstDilatation = diagonal/2
    se = structural_disk(int(round(obstDilatation/resolution)))

    obstMap = cv2.dilate(obstMap, se, iterations = 1)
    obstMap = image_filling(obstMap)
    obstMap = cv2.erode(obstMap, se, iterations = 1)

    # Now, map limits are considered as obstacles
    obstMap[0, :] = 1
    obstMap[-1, :] = 1
    obstMap[:, 0] = 1
    obstMap[:, -1] = 1
    obstMap = np.float64(obstMap)

    # High cost for obstacles
    obstacleHighCost = obstMap*300

    # Dilated cost map based on distance to obstacles
    obstExpansion = 1
    se = structural_disk(int(round(obstExpansion/resolution)))
    dilatedObstMap = cv2.dilate(obstMap, se, iterations = 1)

    obstDistance = resolution*ndimage.distance_transform_edt(obstMap==0)

    obstDist = dilatedObstMap*(1 - obstDistance/(np.max(obstDistance)))
    minDist = np.min(obstDist[obstDist > 0])
    obstDist[obstDist > 0] = obstDist[obstDist > 0] - minDist

    gradient = 10

    obstDilatedHighCost = obstDist*gradient

    # Final cost map
    cMap = 1 + (obstacleHighCost + obstDilatedHighCost).T

    # Smoothing the cost map to avoid abrupt changes
    h = np.ones((50, 50))/50**2

    cMap = signal.convolve2d(cMap, np.flipud(h), mode='same', fillvalue=300)

    # Map limits must have infinite cost
    cMap[0, :] = np.inf
    cMap[-1, :] = np.inf
    cMap[:, 0] = np.inf
    cMap[:, -1] = np.inf

    # =============================================================================
    #     Fast marching method
    # =============================================================================

    goal = sampleNode
    start = roverNode

    # Computing time map
    TmapG, TmapS, nodeJoin = FM.biComputeTmap(cMap.T, goal, start)

    # Obtaining the path
    pathG = FM.getPathGDM(TmapG, nodeJoin, goal, 0.5)
    pathS = FM.getPathGDM(TmapS, nodeJoin, start, 0.5)

    roverPath = np.vstack((np.flipud(pathS), pathG[1:, :]))

    roverPath = dot(resolution, roverPath + 1)

    # Deleting waypoints close to the sample position or the initial rover position
    count = 0
    i = 0
    totalSize = len(roverPath)
    while count < totalSize:
        if np.linalg.norm(roverPath[i, :] - roverPos) < 0.1:
            roverPath = np.delete(roverPath, i, axis=0)
        elif np.linalg.norm(roverPath[i,:] - [xm, ym]) < 0.1:
            roverPath = np.delete(roverPath, i, axis=0)
        else:
            i = i + 1
        count = count + 1

    # Including z coordinate in the rover path
    roverPath = np.vstack((roverPath.T, zp + Zs[np.uint32(np.round(roverPath[:, 1]/resolution)), np.uint32(np.round(roverPath[:, 0]/resolution))])).T

    # Obtaining rover heading
    dX = np.diff(roverPath[:, 0])
    dY = np.diff(roverPath[:, 1])
    heading = np.hstack((initialHeading, np.arctan2(dY, dX))).T

    # Obtaining arm base path and heading
    realBasePath = np.zeros(roverPath.shape)

    for i in range(0, len(roverPath)):
        realBasePath[i, :] = posTransform(roverPath[i, :], [0, 0, heading[i]], [xa, ya, za])

    realBaseHeading = roverHeading(realBasePath, heading, resolution, Zs)

    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     STEP 2: Choosing the best fetching position for the rover
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # Extracting waypoints inside sampling area
    subPath = np.zeros([0, 3])
    subPathHeading = np.zeros([0, 3])

    for i in range(0, len(realBasePath)):
        if np.linalg.norm(realBasePath[i, :] - [xm, ym, zm]) < Rlim:
            subPath = np.vstack((subPath, realBasePath[i, :]))
            subPathHeading = np.vstack((subPathHeading, realBaseHeading[i, :]))


    # =============================================================================
    #     Cost in function of the reachability
    # =============================================================================
   
    distCost = np.zeros(len(subPath))

    for i in range(0, len(subPath)):
        d = np.linalg.norm(subPath[i, :] - [xm, ym, zm])
        baseOrientation = np.arctan2((ym - subPath[i, 1])/d, (xm - subPath[i, 0])/d)

	# Checking if the samples are reachable or not
        diff = baseOrientation - subPathHeading[i, 2]
        if diff < -math.pi: diff = diff + 2*math.pi
        if diff > math.pi: diff = diff - 2*math.pi

        if diff < math.pi/2 and diff > 0:
            if d < rlim:
                distCost[i] = np.inf
            elif d > Rlim:
                distCost[i] = np.inf
            else:
                distCost[i] = 1/((rO + Rm)/2)**2*(d - ((rO + Rm)/2))**2
        else:
            distCost[i] = np.inf

    # Normalizing the cost between [0, 1]
    if not np.all(distCost == np.inf):
        distCost[distCost != np.inf] = distCost[distCost != np.inf]/np.max(distCost[distCost != np.inf])

    # =============================================================================
    #     Cost in function of the position of the wheels
    # =============================================================================

    inclCost = np.zeros(len(subPath))

    for i in range(0, len(subPath)):
        w1, w2, w3, w4, w5, w6 = basePos2wheelsPos(subPath[i, :], subPathHeading[i, :], Zs, resolution)

	# Checking if any wheel could step into an obstacle
        w1Cost = obstMap[int(round(w1[1]/resolution)), int(round(w1[0]/resolution))]
        w2Cost = obstMap[int(round(w2[1]/resolution)), int(round(w2[0]/resolution))]
        w3Cost = obstMap[int(round(w3[1]/resolution)), int(round(w3[0]/resolution))]
        w4Cost = obstMap[int(round(w4[1]/resolution)), int(round(w4[0]/resolution))]
        w5Cost = obstMap[int(round(w5[1]/resolution)), int(round(w5[0]/resolution))]
        w6Cost = obstMap[int(round(w6[1]/resolution)), int(round(w6[0]/resolution))]

        inclCost[i] = w1Cost + w2Cost + w3Cost + w4Cost + w5Cost + w6Cost

        if inclCost[i] > 0:
            inclCost[i] = np.inf


    # =============================================================================
    #     Cost in function of the distance to obstacles
    # =============================================================================

    obstCost = np.zeros(len(subPath))

    for i in range(0, len(subPath)):
        obstCost[i] = obstDistance[int(round(subPath[i, 1]/resolution)), int(round(subPath[i, 0]/resolution))]

    if np.max(obstCost) == 0: obstCost = np.inf + obstCost
    else:
        obstCost = 1 - obstCost/np.max(obstCost)

	# Checking positions really close to obstacles
        for i in range(0, len(subPath)):
            if obstCost[i] >= 0.99:
                obstCost[i] = np.inf


    # =============================================================================
    #     Cost in function of the heading
    # =============================================================================

    headingCost = np.zeros(len(subPath))

    for i in range(0, len(subPath)):
        roverCenterPos = roverPath[-len(subPath) + i,:]
        roverCenterHeading = heading[-len(subPath) + i]

	# Considering as the best orientation if the rover heads right towards the sample
        d = np.linalg.norm([roverCenterPos[0], roverCenterPos[1]] - array([xm, ym]))
        bestOrientation = np.arctan2((ym - roverCenterPos[1])/d, (xm - roverCenterPos[0])/d)

        headingCost[i] = np.abs(roverCenterHeading - bestOrientation)
        if headingCost[i] > math.pi: headingCost[i] = headingCost[i] - 2*math.pi
        if headingCost[i] < math.pi: headingCost[i] = headingCost[i] + 2*math.pi

    headingCost = headingCost/np.max(headingCost)

    # =============================================================================
    #     Total Cost
    # =============================================================================

    # Distance (reachability) cost is the most important when computing the total cost
    totalCost = 2*distCost + inclCost + obstCost + headingCost


    # =============================================================================
    #     Best fetching position
    # ============================================================================

    minCost = min(totalCost)
    index = np.int32(np.where(totalCost == minCost))

    # If there is no possible fetching position, then the sample is not reachable
    if np.isinf(minCost):
        index = np.inf
        print('ERROR: The sample is not reachable')
        return 0

    # Final paths
    finalIndex = len(realBasePath) - len(subPath) + index[0, 0]

    finalBasePath = realBasePath[range(0, finalIndex + 1), :]
    finalBaseHeading = realBaseHeading[range(0, finalIndex + 1), :]

    # Global variables can be returned as outputs of the py function
    global finalRoverPath
    finalRoverPath = roverPath[range(0, finalIndex + 1), :]

    global finalRoverHeading
    finalRoverHeading = heading[range(0, finalIndex + 1)]

    # The prepared-to-fetch end effector position is considered to be 10cm above the sample
    zm = zm + 0.10

    # =============================================================================
    #     Final elbow position (up/down)
    # =============================================================================

    if zm > finalBasePath[-1, 2] - zbase:
        ryf = np.arctan2(Nz[sampleNode[1], sampleNode[0]], np.linalg.norm([Nx[sampleNode[1], sampleNode[0]], Ny[sampleNode[1], sampleNode[0]]]))
    else:
        ryf = (math.pi-np.arctan2(Nz[sampleNode[1], sampleNode[0]], np.linalg.norm([Nx[sampleNode[1], sampleNode[0]], Ny[sampleNode[1], sampleNode[0]]]))) % math.pi

    d = np.linalg.norm([finalBasePath[-1, 0], finalBasePath[-1, 1]] - array([xm, ym]))
    zproy = finalBasePath[-1, 2] - d*np.tan(finalBaseHeading[-1, 1])

    if zproy > zm:
        elbow = 1 					# up elbow
        qid = [-math.pi/2, -1.6422, 1.459, 0.6, 0]  
    else:
        elbow = 0					# down elbow
        qid = [-math.pi/2, -0.6721, -0.5, -0.3456, 0]
        ryf = -ryf % (2*math.pi)


    # =============================================================================
    #     Final arm configuration
    # =============================================================================

    baseSampleDistance = distTransform(finalBasePath[-1, :], [finalBaseHeading[-1, 0], finalBaseHeading[-1, 1] + math.pi/2, finalBaseHeading[-1, 2]],[xm, ym, zm])
    baseSampleOrientation = [np.arctan2(baseSampleDistance[1], baseSampleDistance[0]), ryf, 0]


    qend = MCI(baseSampleDistance, baseSampleOrientation, elbow)
    if np.isnan(qend[0]):
        print('ERROR: The final arm configuration is not viable. The sample is not reachable')
        return 0

    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     STEP 3: End effector path planning and arm motion planning 
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # Select end-effector path planning start point (1 meter before reaching the sample)
    for i in range(len(finalRoverPath) - 1, -1, -1):
        aux = [finalBasePath[i, 0] - xm, finalBasePath[i, 1] - ym, finalBasePath[i, 2] - zm]
        if np.linalg.norm(aux) >= 1:
            index = i
            break

    # =============================================================================
    # Deployment stage
    # =============================================================================

    joints = deploymentStage(qid, qend, index, qinitial)

    # =============================================================================
    # End effector's path planning area
    # =============================================================================

    # Arm's base path and heading inside the end-effector planning area
    effectorBasePath = array(finalBasePath[range(index, len(finalBasePath)), :])
    effectorBaseHeading = array(finalBaseHeading[range(index, len(finalBaseHeading)), :])

    # Maximum and minimum positions of the arm's base path inside the area
    xmin = np.min(effectorBasePath[:, 0]) - 1.4*Rlim
    xmax = np.max(effectorBasePath[:, 0]) + 1.4*Rlim
    ymin = np.min(effectorBasePath[:, 1]) - 1.4*Rlim
    ymax = np.max(effectorBasePath[:, 1]) + 1.4*Rlim

    # Center of the area
    xi = (xmin + xmax)/2
    yi = (ymin + ymax)/2
    pxi = int(round(xi/resolution))
    pyi = int(round(yi/resolution))

    # The size of the area is obtained in funtion of the arm's base path
    dx = xmax - xmin
    dy = ymax - ymin
    if dx >= dy:
        halfMap = dx/2
    else:
        halfMap = dy/2

    indHalf = int(round(halfMap/resolution))

    # Indices of the maximum and minimum positions inside the map
    ixmin = pxi - indHalf
    ixmax = pxi + indHalf
    iymin = pyi - indHalf
    iymax = pyi + indHalf

    # Ensuring there is no point outside the real map
    maxind = int(round(size/resolution))
    if ixmin <= 0:
        ixmin = 1
    elif ixmin > maxind:
        ixmin = maxind
    if ixmax <= 0:
        ixmax = 1
    elif ixmax > maxind:
        ixmax = maxind
    if iymin <= 0:
        iymin = 1
    elif iymin > maxind:
        iymin = maxind
    if iymax <= 0:
        iymax = 1
    elif iymax > maxind:
        iymax = maxind


    # New reduced size map that defines the end-effector's path planning area
    ZsMap = np.zeros([iymax - iymin, ixmax - ixmin])
    for j in range(iymin, iymax):
        for i in range(ixmin, ixmax):
            ZsMap[j - iymin, i - ixmin] = Zs[j, i]


    # =============================================================================
    # New map parameters for the end-effector path planner
    # =============================================================================

    # Map real size
    aX = resolution*(ixmax - 1) - resolution*ixmin
    aY = resolution*(iymax - 1) - resolution*iymin
    aZ = np.max(ZsMap) - np.min(ZsMap) + 0.5

    # Number of nodes for each axis
    sX, sY = ZsMap.shape
    sZ = int(round(aZ/0.02))

    # Resolution for each axis
    resX = aX/sX
    resY = aY/sY
    resZ = 0.02

    # Putting map reference frame into [0, 0, 0]
    Xmin = resolution*ixmin
    Ymin = resolution*iymin
    Zmin = np.min(ZsMap)

    ZsMap = ZsMap - Zmin

    # Putting arm's base path into the new reference frame
    effectorBasePath[:, 0] = effectorBasePath[:, 0] - Xmin
    effectorBasePath[:, 1] = effectorBasePath[:, 1] - Ymin
    effectorBasePath[:, 2] = effectorBasePath[:, 2] - Zmin

    # Putting sample position into the new reference frame
    xmNew = xm - Xmin
    ymNew = ym - Ymin
    zmNew = zm - Zmin

    # =============================================================================
    # Create obstacles cost map
    # =============================================================================

    # Taking the obstacle map inside the new map
    newObstMap = np.zeros([iymax - iymin, ixmax - ixmin])
    for i in range(ixmin, ixmax):
        newObstMap[:, i - ixmin] = obstMap[range(iymin, iymax), i]


    # Obtaining obstacles, ground and total terrain cost map
    Cmap1, obstMap3, groundMap = GetObstMap(ZsMap, resX, resY, resZ, sX, sY, sZ, newObstMap, xm, ym)

    # =============================================================================
    # Set initial and final waypoints
    # =============================================================================

    # Vector with the position and orientation from the arm's base to the
    # end-effector's initial position
    qi = joints[-1, :]

    # Transformation from the arm's base to the end-effector's initial position
    Tbp = MCD(qi)
    Rini = trans2pos(Tbp, elbow)	 # Vector with the position and orientation of the end effector

    # Orientation of the arm's base
    alpha = effectorBaseHeading[0, 2]		 # Rz
    beta = math.pi/2+effectorBaseHeading[0, 1]	 # Ry
    gamma = effectorBaseHeading[0, 0] 		 # Rx

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    # Transformation from the origin to the arm's base
    Tob = [[ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, effectorBasePath[0, 0]],
           [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, effectorBasePath[0, 1]],
           [-sb,   cb*sg,            cb*cg,            effectorBasePath[0, 2]],
           [0,     0,                0,                1]]


    # Transformation from the origin to the end-effector's initial position
    Top = dot(Tob, Tbp)

    # Initial position of the end-effector
    xini = Top[0, 3]
    yini = Top[1, 3]
    zini = Top[2, 3]

    # Nodes with the initial and final end-effector's positions
    initialWayPointArm = np.uint32(np.round([xini/resX, yini/resY, zini/resZ]))
    finalWayPointArm = np.uint32(np.round([xmNew/resX, ymNew/resY, zmNew/resZ]))

    # =============================================================================
    # Generate tunnel extruding a circle
    # =============================================================================

    Cmap2 = TunnelCost(Rlim, rO, rm, effectorBasePath, sX, sY, sZ, resX, resY, resZ, effectorBaseHeading, finalWayPointArm, initialWayPointArm)

    # Total cost map is the sum of ground, obstacles and tunnel
    Cmap = Cmap1*Cmap2



    # =============================================================================
    # Fast Marching 3D
    # =============================================================================

    # Computing time map
    Tmap3D = FM3D.computeTmap(Cmap, finalWayPointArm, initialWayPointArm)

    # Obtaining the end effector path
    gamma3D = array(FM3D.getPathGDM(Tmap3D, initialWayPointArm, finalWayPointArm, 0.5))

    gamma3D[:, 0] = gamma3D[:, 0]*resX
    gamma3D[:, 1] = gamma3D[:, 1]*resY
    gamma3D[:, 2] = gamma3D[:, 2]*resZ

    # Smoothing the path to avoid abrupt changes
    gamma3D[:, 0] = signal.savgol_filter(gamma3D[:, 0], 11, 3)
    gamma3D[:, 1] = signal.savgol_filter(gamma3D[:, 1], 11, 3)
    gamma3D[:, 2] = signal.savgol_filter(gamma3D[:, 2], 11, 3)

    # Returning to the global reference frame
    effectorBasePath[:, 0] = effectorBasePath[:, 0] + Xmin
    effectorBasePath[:, 1] = effectorBasePath[:, 1] + Ymin
    effectorBasePath[:, 2] = effectorBasePath[:, 2] + Zmin

    gamma3D[:, 0] = gamma3D[:, 0] + Xmin
    gamma3D[:, 1] = gamma3D[:, 1] + Ymin
    gamma3D[:, 2] = gamma3D[:, 2] + Zmin

    # Ensuring the sample pose is the last node of the path
    gamma3D[-1, :] = [xm, ym, zm]

    # Ensuring both paths have the same size
    resizedGamma3D = np.zeros([len(finalBasePath) - index + 1, 3])

    fx = interpolate.interp1d(range(0, len(gamma3D)), gamma3D[:, 0])
    fy = interpolate.interp1d(range(0, len(gamma3D)), gamma3D[:, 1])
    fz = interpolate.interp1d(range(0, len(gamma3D)), gamma3D[:, 2])

    resizedGamma3D[:, 0] = fx(np.linspace(0, len(gamma3D) - 1, len(finalBasePath) - index + 1, endpoint = True))
    resizedGamma3D[:, 1] = fy(np.linspace(0, len(gamma3D) - 1, len(finalBasePath) - index + 1, endpoint = True))
    resizedGamma3D[:, 2] = fz(np.linspace(0, len(gamma3D) - 1, len(finalBasePath) - index + 1, endpoint = True))


    # =============================================================================
    #     Relation between end effector path and rover path
    # =============================================================================

    global assignment
    assignment = assign(resizedGamma3D, effectorBasePath, rO, Rm, Rlim)

    # An inverse kinematics model is used to obtain the configuration of the arm for every waypoint in both paths
    effectorJoints = PathInverseKinematics(assignment, resizedGamma3D, effectorBasePath, effectorBaseHeading, qi, Rini, elbow, ryf)

    # Putting together the deployment stage with the end effector path planning stage
    joints = np.vstack((joints[range(0, index - 1), :], effectorJoints))
    assignment = np.hstack((array(range(0, index - 1)), assignment + index - 1))

    # Smoothing the final joints vector to avoid abrupt changes
    global finalJoints
    finalJoints = smoothJoints(joints)

    # Changing from int64 to int32 as C++ uses this last one for int    
    assignment = np.int32(assignment)	
    
    # Ending algorithm
    elapsedTime = time() - startTime
    print("Elapsed execution time: " + str(elapsedTime))
    
    return finalRoverPath, finalRoverHeading, finalJoints, assignment




    
