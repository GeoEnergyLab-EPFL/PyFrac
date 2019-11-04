import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self,name,x,y):
        self.name = name
        self.x = x
        self.y = y


def distance(p1, p2):
    # Compute the euclidean distance between two points
    return np.sqrt((-p1.x + p2.x)**2 + (-p1.y + p2.y)**2)


def pointtolinedistance(p1, p2, p0):
    # Compute the minimum distance from a point of coordinates (x0,y0) to a the line passing through 2 points.
    # The function works only for planar problems.
    x0, x1, x2 = [p0.x, p1.x, p2.x]
    y0, y1, y2 = [p0.y, p1.y, p2.y]
    mac_precision = 10 * np.finfo(float).eps
    dist=0
    if np.abs(x2 - x1)/np.maximum(np.abs(x2),np.abs(x1)) < mac_precision:  # the front is a vertical line
        dist=np.abs(x0 - x1)
    elif np.abs(y2 - y1)/np.maximum(np.abs(y2),np.abs(y1)) < mac_precision:  # the front is an horizontal line
        dist = np.abs(y0 - y1)
    else: #general case
        dist = np.abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/distance(p1, p2)
    return dist


def elements(typeindex, nodeindex, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID):
    if typeindex[nodeindex] == 0:  # the node is one the edge of a cell
        cellOfNodei = connectivityedgeselem[edgeORvertexID[nodeindex]]
    else: # the node is one vertex of a cell
        cellOfNodei = Connectivitynodeselem[edgeORvertexID[nodeindex]]
    return cellOfNodei


def findcommon(nodeindex0, nodeindex1, typeindex, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID):
    """
    given two points we return the cells that are in common between them

    :param nodeindex0: position of the node 0 inside the list of the found intersections that defines the front
    :param nodeindex1: position of the node 1 inside the list of the found intersections that defines the front
    :param typeindex: array that specify if a node at the front is an existing vertex or an intersection with the edge
    :param connectivityedgeselem: given an edge number, it will return all the elements that have the given edge
    :param Connectivitynodeselem: given a node number, it will return all the elements that have the given node
    :param edgeORvertexID: list that contains for each node at the front the number of the vertex or of the edge where it lies
    :return: list of elements
    """
    cellOfNodei = elements(typeindex, nodeindex0, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)
    cellOfNodeip1 = elements(typeindex, nodeindex1, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)
    diff = np.setdiff1d(cellOfNodei,cellOfNodeip1)  # Return the unique values in cellOfNodei that are not in cellOfNodeip1.
    common = np.setdiff1d(cellOfNodei, diff)  # Return the unique values in cellOfNodei that are not in diff.
    return common


def filltable(nodeVScommonelementtable, nodeindex, common, sgndDist_k, column):
    if len(common) == 1:
        nodeVScommonelementtable[nodeindex, column]=common[0]
    elif len(common) > 1:
        """
        situation with two common elements
        ___|______|____           ___|_*__*_|____
           |      |                  |/    \|         
        ___*______*____           ___*______*____
           |      |                 /|      |\  
        ___|______|____           _/_|______|_\___
           |      |                  |      |
        In this situation take the i with LS<0 as tip
        (...if you choose LS>0 as tip you will not find zero vertexes then...)
        """
        #nodeVScommonelementtable[nodeindex, column] = common[np.argmax(sgndDist_k[common])]
        nodeVScommonelementtable[nodeindex,column]=common[np.argmin(sgndDist_k[common])]
    elif len(common) == 0:
        print('ERROR: two consecutive nodes does not belongs to a common cell --->PLEASE STOP THE PROGRAM')
    return nodeVScommonelementtable


def ISinsideFracture(i,mesh,sgndDist_k):
    if i == 1258:
        print("")
    """
    you are in cell i
    you want to know if points 0,1,2,3 are inside or outside of the fracture
    -extrapolate the level set at there points by taking the level set (LS) at the center of the neighbors cells
    -if at the point the LS is < 0 then the point is inside
      _   _   _   _   _   _
    | _ | _ | _ | _ | _ | _ |
    | _ | _ | _ | _ | _ | _ |
    | _ | e | a | f | _ | _ |
    | _ | _ 3 _ 2 _ | _ | _ |
    | _ | d | i | b | _ | _ |
    | _ | _ 0 _ 1 _ | _ | _ |
    | _ | h | c | g | _ | _ |
    | _ | _ | _ | _ | _ | _ |
    """
    #                         0     1      2      3
    #NeiElements[i]->[left, right, bottom, up]
    [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]

    a = mesh.NeiElements[i, top_elem]
    b = mesh.NeiElements[i, right_elem]
    c = mesh.NeiElements[i, bottom_elem]
    d = mesh.NeiElements[i, left_elem]
    e = mesh.NeiElements[d, top_elem]
    f = mesh.NeiElements[b, top_elem]
    g = mesh.NeiElements[b, bottom_elem]
    h = mesh.NeiElements[d, bottom_elem]

    hcid_mean = np.mean(np.asarray(sgndDist_k[[h, c, i, d]]))
    cgbi_mean = np.mean(np.asarray(sgndDist_k[[c, g, b, i]]))
    ibfa_mean = np.mean(np.asarray(sgndDist_k[[i, b, f, a]]))
    diae_mean = np.mean(np.asarray(sgndDist_k[[d, i, a, e]]))
    answer_on_vertexes = [hcid_mean<0, cgbi_mean<0, ibfa_mean<0, diae_mean<0]
    return answer_on_vertexes


def findangle(x1, y1, x2, y2, x0, y0):
    """
    Compute the angle with respect to the horizontal direction between the segment from a point of coordinates (x0,y0)
    and orthogonal to a the line passing through 2 points. The function works only for planar problems.

    Args:
    :param x0: coordinate x point
    :param y0: coordinate y first
    :param x1: coordinate x first point that defines the line
    :param y1: coordinate y first point that defines the line
    :param x2: coordinate x second point that defines the line
    :param y2: coordinate y second point that defines the line

    Returns:
    :return: angle, xintersections, yintersections

    """
    mac_precision = 10*np.finfo(float).eps
    if np.abs(x2 - x1)/np.maximum(np.abs(x2),np.abs(x1)) < mac_precision:  # the front is a vertical line
        x = x2
        y = y0
        angle = 0.
    elif np.abs(y2 - y1)/np.maximum(np.abs(y2),np.abs(y1)) < mac_precision:  # the front is an horizontal line
        angle = np.pi/2
        x = x0
        y = y2
    else:
        # m and q1 are the coefficients of the line defined by (x1,y1) and (x2,y2): y = m * x + q1
        # q2 is the coefficients of the line defined by (x,y) and (x0,y0): y = -1/m * x + q2
        m = (y2 - y1) / (x2 - x1)
        q1 = y2 - m * x2
        q2 = y0 + x0 / m
        x = (q2 - q1) * m / (m * m + 1)
        y = m * x + q1
        angle = np.arctan(np.abs((y-y0))/np.abs((x-x0)))

    return angle, x, y


def find_first_cell(i, mesh, anularegion, sgndDist_k):
    checked_cells = 0
    total_cells = len(anularegion)
    found_cell = False

    #                         0     1      2      3
    #NeiElements[i]->[left, right, bottom, up]
    [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]

    while found_cell is False and checked_cells < total_cells:
        """
        Consider the following cells around the cell "i":
        In cell i the level set is negative
        Find the cell where LS > 0 or change the current i
         _ _ _ _ _ _
        |_|_|_|_|_|_|
        |_|_|_|_|_|_|
        |_|e|a|f|_|_|
        |_|d|i|b|_|_|
        |_|h|c|g|_|_|
        |_|_|_|_|_|_|

                                 0     1      2      3
        NeiElements[i]->[left, right, bottom, up]
        """

        a = mesh.NeiElements[i, top_elem]
        b = mesh.NeiElements[i, right_elem]
        c = mesh.NeiElements[i, bottom_elem]
        d = mesh.NeiElements[i, left_elem]
        e = mesh.NeiElements[d, top_elem]
        f = mesh.NeiElements[b, top_elem]
        g = mesh.NeiElements[b, bottom_elem]
        h = mesh.NeiElements[d, bottom_elem]
        abcdefgh = [a, b, c, d, e, f, g, h]
        elements = [i, i, c, d, d, i, c, h]
        LS = sgndDist_k[abcdefgh]

        if not np.all(LS < 0):
            """
            the front is passing for the big set of elements abcdefgh
            reduce to a fictitious cell where at least one node has LS<0 and  at least one node has LS>0
            include always i because we know it belongs to the ribbon cells
            """
            index = np.argmax(LS)  # index of the i with positive LS
            i = elements[index]
            found_cell = True

        elif np.all(LS < 0):
            """
            the front is not passing for the big set of elements abcdefgh
            take a new set of elements centered in the i of the current set with the higher value of LS
            in this way we are moving closer to the front.                
            """
            index = np.argmax(LS)
            i = elements[index]
            checked_cells = checked_cells + 1
    if checked_cells > len(anularegion):
        raise SystemExit('ERROR: unable to find the first cell where the front is passing trough, LS non correct')
    else:
        return i   


def find_next_cell(i, mesh, sgndDist_k):

    #                         0     1      2      3
    # NeiElements[i]->[left, right, bottom, up]
    [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]
    """
    you are in the cell i --> take the cell a,b,c
    the front is crossing the cell because at least one vertex has LS<0 and at least one has LS>0
    there are always 2 edges where the front enter the fictitious cell and where it exits the cell.
    The front can't exit the fictitious cell from a vertex because we set LS -(machine precision) where it was 0
     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    remembrer the usage of NeiElements[i]->[left, right, bottom, up].
    """
    a = mesh.NeiElements[i, top_elem]
    c = mesh.NeiElements[i, right_elem]
    b = mesh.NeiElements[i + 1, top_elem]
    icba = [i, c, b, a]
    LS = sgndDist_k[icba]

    """
    take the vertical and horizontal lines passing through the center of the fictitious cell
    2 intersections with edges----> just take the first
    LS[m]*LS[n] if both with the same sign then the product will be always positive
    """
    first_point=Point(0,0.,0.)
    if (LS[0] * LS[1]) < 0:
        c1 = mesh.CenterCoor[i, 0]
        c2 = mesh.CenterCoor[c, 0]
        first_point.x = c1 + np.abs(LS[0]) * (c2 - c1) / (np.abs(LS[0]) + np.abs(LS[1]))
        first_point.y = mesh.CenterCoor[i, 1]
        # intersection on the lower fictitious edge i-c
        # =>change the current fictitious i to the bottom i
        i = mesh.NeiElements[i, bottom_elem]
    elif (LS[1] * LS[2]) < 0:
        c1 = mesh.CenterCoor[c, 1]
        c2 = mesh.CenterCoor[b, 1]
        first_point.x = mesh.CenterCoor[c, 0]
        first_point.y = c1 + np.abs(LS[1]) * (c2 - c1) / (np.abs(LS[1]) + np.abs(LS[2]))
        # intersection on the right fictitious edge c-b
        # =>change the current fictitious i to the right i
        i = c
    elif (LS[2] * LS[3]) < 0:
        c1 = mesh.CenterCoor[a, 0]
        c2 = mesh.CenterCoor[b, 0]
        first_point.x = c1 + np.abs(LS[3]) * (c2 - c1) / (np.abs(LS[2]) + np.abs(LS[3]))
        first_point.y = mesh.CenterCoor[a, 1]
        # intersection on the top fictitious edge a-b
        # =>change the current fictitious i to the upper i
        i = a
    elif (LS[3] * LS[0]) < 0:
        c1 = mesh.CenterCoor[i, 1]
        c2 = mesh.CenterCoor[a, 1]
        first_point.x = mesh.CenterCoor[i, 0]
        first_point.y = c1 + np.abs(LS[0]) * (c2 - c1) / (np.abs(LS[3]) + np.abs(LS[0]))
        # intersection on the left fictitious edge i-a
        # =>change the current fictitious i to the i below
        i = mesh.NeiElements[i, left_elem]
    else:
        raise SystemExit('ERROR: unable to find the the cell where the front is exiting from the 1st cell')
    return i, first_point


def reconstruct_front_continuous(sgndDist_k, anularegion, Ribbon, eltsChannel, mesh):

        # sgndDist_k vector that contains the distance from the center of the cell to the front. They are negative if the point is inside the fracture otherwise they are positive
        # anularegion cells where we expect to be the front
        # Ribbon ribbon elements
        # mesh obj

        """
        description of the function.

        Args:
            mesh (object):               -- descriptions.
            distances (float):           -- descriptions.
            ribboncells (integers):      -- descriptions.
            tipcells (integers):         -- descriptions.

        Returns:
            tipcells (integers):         -- descriptions.
            nextribboncells (integers):  -- descriptions.
            vertexes (integers):         -- descriptions.
            alphas (float):              -- descriptions.
            orthogonaldistances (float): -- descriptions.

        """
        float_precision = np.float64
        integer_precision = int
        mac_precision = 100*np.finfo(float).eps
        """
        0) Set all the LS==0 to -mac_precision
        In this way we avoid to deal with the situations where the front is crossing exactly a vertex of an i     
        """
        zerovertexes = np.where(sgndDist_k == 0)[0]
        if len(zerovertexes) > 0:
            sgndDist_k[zerovertexes] = -mac_precision

        """
        1) arrays initialization:
        """
        icbaMat = np.empty([2, 2], dtype=integer_precision)
        alphaX = np.empty(4, dtype=float_precision)
        alphaY = np.empty(4, dtype=float_precision)

        # only for plotting purposes:
        orderallx = np.asarray([1, 0, 3, 2], dtype=integer_precision)
        orderally = np.asarray([3, 2, 1, 0], dtype=integer_precision)
        xintersection = []
        yintersection = []
        elementstorage = []
        typeindex = []  # it's telling you if we have the intersection on a vertex (1) or on a edge (0)
        edgeORvertexID = []  # is telling you the vertex or the edge ID of the intersected point
        xred = []
        yred = []
        xgreen = []
        ygreen = []
        xblack = []
        yblack = []
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        """
        --------------------------------------------------------------------
        2)Find the first fictitius cell where the front is passing.
        The Boundary Element Method discretizes the domain in subcells.
        A fictitius cell is generated by the union of the centers of 4 cells
        --------------------------------------------------------------------     
        """
        # start from a cell in the ribbon, so we are sure that in cell i the LS < 0
        i = Ribbon[0]
        i = find_first_cell(i, mesh, anularegion, sgndDist_k)
        i,first_point = find_next_cell(i, mesh, sgndDist_k)
        x_controlnode = first_point.x
        y_controlnode = first_point.y
        """
        ------------------
        3) Go outside from the previous fictitious cell i,
           find all the intersections,
           follow the front and find the next fictitious cell:
        ------------------
        """

        #                         0     1      2      3
        # NeiElements[i]->[left, right, bottom, up]
        [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]

        Front_not_Completed = True  # if the front has been completed
        counter = 0
        maxiter = 2*mesh.NumberOfElts
        while Front_not_Completed and counter < maxiter:
            counter = counter+1
            elementstorage.append(i)
            up = mesh.NeiElements[i, top_elem]
            right = mesh.NeiElements[i, right_elem]
            rightUp = mesh.NeiElements[i+1, top_elem]
            icba = np.asarray([i, right, rightUp, up], dtype=integer_precision)
            LS = sgndDist_k[icba]

            icbaMat[0, :] = [i, right]
            icbaMat[1, :] = [up, rightUp]
            """
            icbaMat:
             _               _
            |                 |
            |    i   ; right  | 
            |                 |
            |   up   ; rightup|
            |_               _|
            
            """

            if not(np.all(LS > 0) or np.all(LS < 0)):
                # the front is crossing the cell.
                # take the vertical and horizontal line passing trough the center of the cell
                centernode = mesh.Connectivity[i, 2]
                xgrid = mesh.VertexCoor[centernode, 0]
                ygrid = mesh.VertexCoor[centernode, 1]
                allx = np.asarray([mesh.CenterCoor[right, 0], mesh.CenterCoor[i, 0], mesh.CenterCoor[up, 0], mesh.CenterCoor[rightUp, 0]], dtype=float_precision)
                ally = np.asarray([mesh.CenterCoor[up, 1], mesh.CenterCoor[rightUp, 1], mesh.CenterCoor[right, 1], mesh.CenterCoor[i, 1]], dtype=float_precision)

                # only for plotting purposes
                # vvvvvvvvvvvvvvvvvvvvvvvv
                counternegative = 0
                counterpositive = 0
                for j in range(0, 4):
                    if LS[j] > 0:
                        xred.append(allx[orderallx][j])
                        yred.append(ally[orderally][j])
                        counterpositive = counterpositive+1
                    elif LS[j] < 0:
                        xgreen.append(allx[orderallx][j])
                        ygreen.append(ally[orderally][j])
                        counternegative = counternegative+1
                    elif LS[j] == 0:
                        xblack.append(allx[orderallx][j])
                        yblack.append(ally[orderally][j])
                        counternegative = counternegative+1
                        counterpositive = counterpositive+1
                if counterpositive == 4 or counternegative == 4:
                    print('error')
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                appendYcandidate = 'no'
                appendXcandidate = 'no'

                # if the sum of LS=0 then the front is passing in the middle of the cell
                if np.sum(LS) == 0:
                    edgeORvertexID.append(centernode)
                    xintersection.append(mesh.VertexCoor[centernode,0])
                    yintersection.append(mesh.VertexCoor[centernode,1])
                    typeindex.append(1)  #intersecting with a vertex
                    print('point naturally on the vertex')

                else:  # the front can intersect the edges

                    # compute the alphaX and alphaY vectors
                    # xgrid=mesh.CenterCoor[i,0]+mesh.hx/2
                    # ygrid=mesh.CenterCoor[i,1]+mesh.hy/2
                    alphaX[[0,2]] = (xgrid - allx[[0, 2]])
                    alphaX[[1,3]] = -(xgrid - allx[[1, 3]])
                    alphaY[[0,2]] = (ygrid - ally[[0, 2]])
                    alphaY[[1,3]] = -(ygrid - ally[[1, 3]])

                    # compute the intersection with the mesh cells
                    if not (np.inner(LS, alphaX)) == 0:
                        # intersection between the front and the Vertical line
                        yCandidate = np.inner(ally, LS*alphaX)/(np.inner(LS, alphaX))

                        # i) check if the intersection with the front and the vertical line is inside the grid
                        maxabmax = np.maximum(np.abs(yCandidate), np.abs(np.max(ally)))
                        maxabmin = np.maximum(np.abs(yCandidate), np.abs(np.min(ally)))

                        if (yCandidate - np.max(ally)) / maxabmax <= +mac_precision / maxabmax and (
                                yCandidate - np.min(ally)) / maxabmin >= -mac_precision / maxabmin:
                            """
                            if yCandidate <= np.max(ally) and yCandidate >= np.min(ally) => the point is inside the cell
                            ii) find if the point is between i and right <OR> up and rightup

                             ____________/_______
                            |          |/        |
                            |      up  o right up|
                            |_________/|_________|
                            |        / |         |
                            |  i    /  | right   |
                            |______/___|_________|
                                  /
                             ______________/_____
                            |         |   /      |
                            |     up  |right  up |
                            |_________|_/________|
                            |         |/         |
                            | i       o   right  |
                            |________/|__________|
                                    /
                                       icbaMat[0,:]=[i,right]
                                       icbaMat[1,:]=[up,rightUp]              
                            """
                            if (yCandidate - ygrid) / np.maximum(np.abs(yCandidate), np.abs(ygrid)) < 0:  # i and right
                                Yposition = 0  # down
                                appendYcandidate = 'yes'  # save the coordinates
                            elif (yCandidate - ygrid) / np.maximum(np.abs(yCandidate), np.abs(ygrid)) > 0:  # rightup and up
                                Yposition = 1  # up
                                appendYcandidate = 'yes'  # save the coordinates
                        else:
                            """
                            no intersections with vertical axis
                             ____|________________
                            |    |     |          |
                            |    | up  |right  up |
                            |____|_____|__________|
                            |    |     |          |
                            | i  |   right        |
                            |____|_____|__________|
                                 |
                            checking flags
                            """
                            appendYcandidate = 'no'
                            Yposition = 'none'

                    if not (np.inner(LS, alphaY)) == 0:
                        # intersection between the front and the Horizontal line
                        xCandidate = np.inner(allx, LS*alphaY)/(np.inner(LS, alphaY))

                        # check if the intersection with the front and the horizontal line is inside the grid
                        maxabmax = np.maximum(np.abs(xCandidate), np.abs(np.max(allx)))
                        maxabmin = np.maximum(np.abs(xCandidate), np.abs(np.min(allx)))
                        if (xCandidate - np.max(allx)) / maxabmax <= +mac_precision / maxabmax and (
                                xCandidate - np.min(allx)) / maxabmin >= -mac_precision / maxabmin:
                            """
                            if xCandidate  <= np.max(allx) and xCandidate >= np.min(allx) =>the point is inside the cell
                            ii) find if the point is between i and up <OR> right and rightup

                             _____/______________
                            |    /     |         |
                            |   /  up  | right up|
                            |__o_______|_________|
                            | /        |         |
                            |/      i  | right   |
                            |__________|_________|

                             _________________/__
                            |         |      /   |
                            |     up  |right/ up |
                            |_________|____o_____|
                            |         |   /      |
                            |     i   |  /right  |
                            |_________|_/________|
                                       /
                            icbaMat[0,:]=[i,right]
                            icbaMat[1,:]=[up,rightUp]                            
                            """

                            if (xCandidate - xgrid) / np.maximum(np.abs(xCandidate), np.abs(xgrid)) < 0:
                                # i and up
                                Xposition = 0  # left
                                appendXcandidate = 'yes'  # save the coordinates
                            elif (xCandidate - xgrid) / np.maximum(np.abs(xCandidate), np.abs(xgrid)) > 0:
                                # right and up
                                Xposition = 1  # right
                                appendXcandidate = 'yes'  # save the coordinates
                        else:
                            """
                            no intersections with horizontal axis
                               ____________________
                              |         |          |
                              |     up  |right  up |
                              |_________|__________|
                            __|_________|__________|_______
                              |     i   |   right  |
                              |_________|__________|

                            checking flags                            
                            """

                            appendXcandidate = 'no'
                            Xposition = 'none'
                    """
                    1 or 2 intersections exist 
                    check the i if ia s ribbon cell.
                    
                    icbaMat:
                     _               _
                    |                 |
                    |    i ;   right  | 
                    |                 |
                    |   up   ; rightup|
                    |_               _|
                    
                    Yposition <=> down (0) / up (1)
                    Xposition <=> left (0) / right (1)
                    
                     _               _
                    |        |        |
                    |    i   | right  | 
                    |________|________|
                    |   up   | rightup|
                    |_       |       _|
                    
                    """
                    if appendYcandidate == 'yes' and appendXcandidate == 'yes' and \
                            (np.any(np.isin(Ribbon, icbaMat[Yposition,Xposition])) \
                             or np.all(np.isin(Ribbon, np.setdiff1d(icba,icbaMat[Yposition,Xposition])))):
                        """
                            regarding the last bolean number:
                            if the first is true: for sure the front is inside a ribbon
                            if the second is true you migth have more than one node per cell
                            in any case : move the nodes to the center
                        """
                        if (np.any(np.isin(Ribbon, icbaMat[Yposition,Xposition]))):
                            print('point inside the ribbon')
                        elif np.all(np.isin(Ribbon, np.setdiff1d(icba,icbaMat[Yposition,Xposition]))):
                            print('more than one node per cell')
                        print('point forced to be on the vertex because the front is going inside the ribbon cell')
                        typeindex.append(1)
                        edgeORvertexID.append(centernode)
                        xintersection.append(mesh.VertexCoor[centernode, 0])
                        yintersection.append(mesh.VertexCoor[centernode, 1])
                    elif appendYcandidate == 'yes' or appendXcandidate == 'yes':
                        # both intersections exists
                        if appendXcandidate == 'yes' and appendYcandidate == 'yes':
                            # append first the closest to the controlnode
                            p_atfront=Point(0,x_controlnode, y_controlnode)
                            p_y_intersect=Point(0,xCandidate, ygrid)
                            p_x_intersect = Point(0, xgrid, yCandidate)
                            dXTOcontrolnode=distance(p_y_intersect, p_atfront)
                            dYTOcontrolnode=distance(p_x_intersect, p_atfront)

                            if dXTOcontrolnode < dYTOcontrolnode:  # write first x intersection
                                xintersection.append(xCandidate)
                                yintersection.append(ygrid)
                                typeindex.append(0)
                                if Xposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[i], mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                                elif Xposition==1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                xintersection.append(xgrid)
                                yintersection.append(yCandidate)
                                typeindex.append(0)
                                if Yposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[i], mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                elif Yposition == 1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                            else: # write first y intersection
                                xintersection.append(xgrid)
                                yintersection.append(yCandidate)
                                typeindex.append(0)
                                if Yposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[i], mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                elif Yposition == 1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                                xintersection.append(xCandidate)
                                yintersection.append(ygrid)
                                typeindex.append(0)
                                if Xposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[i],
                                                          mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                                elif Xposition == 1:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[rightUp],
                                                          mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                        else:
                            # only one intersection exists
                            if appendXcandidate=='yes':
                                # check if the intersection is in the ribbon cell
                                if (np.any(np.isin(Ribbon, icbaMat[:, Xposition]))):
                                    print('point forced to be on the vertex because the front is going inside the ribbon cell')
                                    typeindex.append(1)
                                    edgeORvertexID.append(centernode)
                                    xintersection.append(mesh.VertexCoor[centernode, 0])
                                    yintersection.append(mesh.VertexCoor[centernode, 1])
                                else:
                                    xintersection.append(xCandidate)
                                    yintersection.append(ygrid)
                                    typeindex.append(0)
                                    if Xposition == 0:
                                        edge = np.intersect1d(mesh.Connectivityelemedges[i], mesh.Connectivityelemedges[up])
                                        edgeORvertexID.append(edge[0])
                                    elif Xposition==1:
                                        edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[right])
                                        edgeORvertexID.append(edge[0])
                            if appendYcandidate=='yes':
                                # check if the intersection is in the ribbon cell
                                if (np.any(np.isin(Ribbon, icbaMat[Yposition,:]))):
                                    print('point forced to be on the vertex because the front is going inside the ribbon cell')
                                    typeindex.append(1)
                                    edgeORvertexID.append(centernode)
                                    xintersection.append(mesh.VertexCoor[centernode, 0])
                                    yintersection.append(mesh.VertexCoor[centernode, 1])
                                else:
                                    xintersection.append(xgrid)
                                    yintersection.append(yCandidate)
                                    typeindex.append(0)
                                    if Yposition == 0:
                                        edge = np.intersect1d(mesh.Connectivityelemedges[i], mesh.Connectivityelemedges[right])
                                        edgeORvertexID.append(edge[0])
                                    elif Yposition == 1:
                                        edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[up])
                                        edgeORvertexID.append(edge[0])

                """
                vocabulary:
                
                xintersection:= x coordinates
                yintersection:= y coordinates
                typeindex:= 0 if node intersecting an edge, 1 if intersecting an existing vertex of the mesh
                edgeORvertexID:= index of the vertex or index of the edge 
                controlnode:= is the last node of the previous cell
                """
                """
                compute the new controlnode and the new fictitious i where the front is going and define i
                 _ _ _ _ _ _
                |_|_|_|_|_|_|
                |_|_|_|_|_|_|
                |_|_|a|b|_|_|
                |_|_|i|c|_|_|
                |_|_|_|_|_|_|
                |_|_|_|_|_|_|
                NeiElements[i]->[left, right, bottom, up].
                
                2 intersections with the edges of the fictitius cell, TAKE THE ONE THAT IS NOT ALREADY SELECTED
                """

                found_next_controlnode='no'
                if (LS[0]*LS[1] < 0) and found_next_controlnode == 'no':
                    # intersection on the lower fictitius edge i-c
                    c1 = mesh.CenterCoor[i, 0]
                    c2 = mesh.CenterCoor[right, 0]
                    x_controlnode_temp = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[0])+np.abs(LS[1]))
                    y_controlnode_temp = mesh.CenterCoor[i, 1]
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius i
                        i = mesh.NeiElements[i, bottom_elem]
                        found_next_controlnode = 'yes'
                if (LS[1]*LS[2]<0) and found_next_controlnode=='no':
                    # intersection on the right fictitius edge c-b
                    c1 = mesh.CenterCoor[right, 1]
                    c2 = mesh.CenterCoor[rightUp, 1]
                    x_controlnode_temp = mesh.CenterCoor[right, 0]
                    y_controlnode_temp = c1+np.abs(LS[1])*(c2-c1)/(np.abs(LS[1])+np.abs(LS[2]))
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius i
                        i = right
                        found_next_controlnode = 'yes'
                if (LS[2]*LS[3] < 0) and found_next_controlnode=='no':
                    # intersection on the top fictitius edge a-b
                    c1 = mesh.CenterCoor[up, 0]
                    c2 = mesh.CenterCoor[rightUp, 0]
                    x_controlnode_temp = c1+np.abs(LS[3])*(c2-c1)/(np.abs(LS[2])+np.abs(LS[3]))
                    y_controlnode_temp = mesh.CenterCoor[up, 1]
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius i
                        i = up
                        found_next_controlnode = 'yes'
                if (LS[0]*LS[3] < 0) and found_next_controlnode=='no':
                    # intersection on the left fictitius edge i-a
                    c1 = mesh.CenterCoor[i, 1]
                    c2 = mesh.CenterCoor[up, 1]
                    x_controlnode_temp = mesh.CenterCoor[i, 0]
                    y_controlnode_temp = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[3])+np.abs(LS[0]))
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius i
                        i = mesh.NeiElements[i, left_elem]
                        found_next_controlnode = 'yes'
                if found_next_controlnode == 'no':
                    print('ERROR: next fictitius i not found. INFINITE LOOP STARTED! Please Stop me (1)')

                #make the if condition for closing the loop
                if first_point.y==y_controlnode and first_point.x==x_controlnode:
                    Front_not_Completed = False
                # A = np.full(mesh.NumberOfElts, np.nan)
                # A[anularegion] = sgndDist_k[anularegion]
                # from visualization import plot_fracture_variable_as_image
                # figure = plot_fracture_variable_as_image(A, mesh)
                # ax = figure.get_axes()[0]
                # xtemp = xintersection
                # ytemp = yintersection
                # # plt.plot(mesh.VertexCoor[mesh.Connectivity[Ribbon, 0], 0],
                # #         mesh.VertexCoor[mesh.Connectivity[Ribbon, 0], 1],
                # #         '.', color='violet')
                # plt.plot(xtemp, ytemp, '-o')
                # plt.plot(xred, yred, '.', color='red')
                # plt.plot(xgreen, ygreen, '.', color='yellow')
                # plt.plot(xblack, yblack, '.', color='black')
                # # ---Plot the ribbon cells with orange---
                # plt.plot(mesh.CenterCoor[Ribbon, 0] + mesh.hx / 10, mesh.CenterCoor[Ribbon, 1] + mesh.hy / 10, '.',
                #          color='orange')
                # plt.plot([xCandidate], [ygrid], 'ob')
                # plt.plot([xgrid], [yCandidate], 'ob')
            else:
                print('ERROR: wrong front direction')

        if counter > maxiter:
            raise SystemExit('ERROR: unable to find the last point of the front')

        """
        ------------------
        4)  Make a 2D table where to store info for each node found at the front. 
            The 1st column contains the TIPcell's name common with the 
            previous node in the list of nodes at the front while the second column the cell's name common with the next node.
            The nodes that have to be deleted will have same value in both columns
            
         ___|__________|___   
            |   in     |
            |         /|
            |\       / |
         ___|_\_____/__|___
            |          |
            |    out   |
            
            
        ------------------  
                  
          
        """
        nodeVScommonelementtable=np.zeros([len(xintersection), 3],dtype=int)
        for nodeindex in range(0,len(xintersection)):
            # commonbackward contains the unique values in cellOfNodei that are in cellOfNodeim1.
            if nodeindex== 0:
                commonbackward = findcommon(nodeindex, len(xintersection)-1, typeindex, mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
            else:
                commonbackward = findcommon(nodeindex, (nodeindex - 1), typeindex, mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
            # commonforward contains the unique values in cellOfNodei that are in cellOfNodeip1.
            if nodeindex== len(xintersection)-1:
                 commonforward = findcommon(nodeindex, 0,typeindex,mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
            else:
                 commonforward = findcommon(nodeindex, (nodeindex + 1), typeindex, mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)

            column=0
            nodeVScommonelementtable=filltable(nodeVScommonelementtable,nodeindex,commonbackward,sgndDist_k,column)
            column=1
            nodeVScommonelementtable=filltable(nodeVScommonelementtable,nodeindex,commonforward,sgndDist_k,column)

        listofTIPcells = []
        # remove the nodes in the cells with more than 2 nodes and keep the first and the last node
        counter = 0
        n=len(xintersection)
        for nodeindex in range(0, len(xintersection)):
            if nodeVScommonelementtable[nodeindex][1] == nodeVScommonelementtable[nodeindex][0]:
                # plot before removing
                # A = np.full(mesh.NumberOfElts, np.nan)
                # A[anularegion] = sgndDist_k[anularegion]
                # from visualization import plot_fracture_variable_as_image
                # figure = plot_fracture_variable_as_image(A, mesh)
                # ax = figure.get_axes()[0]
                # xtempppp = xintersection
                # ytempppp = yintersection
                # xtempppp.append(xtempppp[0]) # close the front
                # ytempppp.append(ytempppp[0]) # close the front
                # plt.plot(xtempppp, ytempppp, '-o')
                # plt.plot( xintersection[nodeindex-counter - 1:nodeindex-counter + 1], yintersection[nodeindex-counter - 1:nodeindex-counter + 1], '-r')
                # plt.plot(xblack, yblack, '.',color='black')
                # plt.plot(mesh.CenterCoor[Ribbon,0], mesh.CenterCoor[Ribbon,1], '.',color='g')
                # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10, mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')

                del xintersection[nodeindex-counter]
                del yintersection[nodeindex-counter]
                del typeindex[nodeindex-counter]
                del edgeORvertexID[nodeindex-counter]

                nodeVScommonelementtable[nodeindex][2]=1 # to remember that the node has been deleted

                #plot after removing
                # A = np.full(mesh.NumberOfElts, np.nan)
                # A[anularegion] = sgndDist_k[anularegion]
                # from visualization import plot_fracture_variable_as_image
                # figure = plot_fracture_variable_as_image(A, mesh)
                # ax = figure.get_axes()[0]
                # xtempppp = xintersection
                # ytempppp = yintersection
                # xtempppp.append(xtempppp[0])  # close the front
                # ytempppp.append(ytempppp[0])  # close the front
                # plt.plot(xtempppp, ytempppp, '-o')
                # plt.plot(xintersection[nodeindex-counter - 1:nodeindex-counter + 1], yintersection[nodeindex-counter - 1:nodeindex-counter + 1], '-r')
                # plt.plot(xblack, yblack, '.', color='black')
                # plt.plot(mesh.CenterCoor[Ribbon, 0], mesh.CenterCoor[Ribbon, 1], '.', color='g')
                # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10,
                #          mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')
                counter = counter + 1
            elif nodeVScommonelementtable[nodeindex][0] == nodeVScommonelementtable[(nodeindex+1)%n][1]:
                # plot before removing
                # A = np.full(mesh.NumberOfElts, np.nan)
                # A[anularegion] = sgndDist_k[anularegion]
                # from visualization import plot_fracture_variable_as_image
                # figure = plot_fracture_variable_as_image(A, mesh)
                # ax = figure.get_axes()[0]
                # xtempppp = xintersection
                # ytempppp = yintersection
                # xtempppp.append(xtempppp[0]) # close the front
                # ytempppp.append(ytempppp[0]) # close the front
                # plt.plot(xtempppp, ytempppp, '-o')
                # plt.plot( xintersection[nodeindex-counter - 1:nodeindex-counter + 1], yintersection[nodeindex-counter - 1:nodeindex-counter + 1], '-r')
                # plt.plot(xblack, yblack, '.',color='black')
                # plt.plot(mesh.CenterCoor[Ribbon,0], mesh.CenterCoor[Ribbon,1], '.',color='g')
                # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10, mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')
                del xintersection[nodeindex-counter]
                del yintersection[nodeindex-counter]
                del typeindex[nodeindex-counter]
                del edgeORvertexID[nodeindex-counter]
                nodeVScommonelementtable[nodeindex][2]=1 # to remember that the node has been deleted
                counter = counter + 1
                del xintersection[(nodeindex-counter+1)%len(xintersection)]
                del yintersection[(nodeindex-counter+1)%len(xintersection)]
                del typeindex[(nodeindex-counter+1)%len(xintersection)]
                del edgeORvertexID[(nodeindex-counter+1)%len(xintersection)]
                nodeVScommonelementtable[(nodeindex+1)%n][2]=1 # to remember that the node has been deleted
                counter = counter + 1
                # plot after removing
                # A = np.full(mesh.NumberOfElts, np.nan)
                # A[anularegion] = sgndDist_k[anularegion]
                # from visualization import plot_fracture_variable_as_image
                # figure = plot_fracture_variable_as_image(A, mesh)
                # ax = figure.get_axes()[0]
                # xtempppp = xintersection
                # ytempppp = yintersection
                # xtempppp.append(xtempppp[0])  # close the front
                # ytempppp.append(ytempppp[0])  # close the front
                # plt.plot(xtempppp, ytempppp, '-o')
                # plt.plot(xintersection[nodeindex-counter - 1:nodeindex-counter + 1], yintersection[nodeindex-counter - 1:nodeindex-counter + 1], '-r')
                # plt.plot(xblack, yblack, '.', color='black')
                # plt.plot(mesh.CenterCoor[Ribbon, 0], mesh.CenterCoor[Ribbon, 1], '.', color='g')
                # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10,
                #          mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')
            else:
                listofTIPcells.append(nodeVScommonelementtable[nodeindex][0])

        if np.unique(np.asarray(listofTIPcells)).size != len(listofTIPcells):
            duplicates=np.abs(np.unique(np.asarray(listofTIPcells)).size-len(listofTIPcells))
            raise SystemExit('ERROR: the front has cells that are duplicates:', duplicates)

        """
        ------------------
        5)  Define the correct node from where compute the distance to the front
            that node should have the larger distance from the front and being inside the fracture but belonging to the tip cell
            define the angle
        ------------------  
        """
        vertexpositionwithinthecell=[0 for i in range(len(listofTIPcells))]
        vertexID = [0 for i in range(len(listofTIPcells))]
        distances = [0 for i in range(len(listofTIPcells))]
        angles = [0 for i in range(len(listofTIPcells))]
        xintersectionsfromzerovertex = []
        yintersectionsfromzerovertex = []

        for nodeindex in range(0, len(xintersection)):
            if nodeindex == len(xintersection)-1:
                nodeindexp1 = 0
            else:
                nodeindexp1 = nodeindex+1
            localvertexID = []
            localdistances = []
            localvertexpositionwithinthecell = []
            p = Point(0,0.,0.)
            i=listofTIPcells[nodeindexp1]
            answer_on_vertexes = ISinsideFracture(i, mesh, sgndDist_k)
            for j in range(0,4):
                p.name = mesh.Connectivity[i][j]
                p.x = mesh.VertexCoor[p.name][0]
                p.y = mesh.VertexCoor[p.name][1]
                if answer_on_vertexes[j]:
                    localvertexID.append(p.name)
                    localvertexpositionwithinthecell.append(j)
                    p1 = Point(0,xintersection[nodeindex], yintersection[nodeindex])
                    p2 = Point(0,xintersection[nodeindexp1], yintersection[nodeindexp1])
                    localdistances.append(pointtolinedistance(p1, p2, p))

            # take the largest distance from the front
            if len(localdistances)==0:
                raise SystemExit('ERROR: there are no nodes in the given tip cell that are inside the fracture')
            index = np.argmax(np.asarray(localdistances))
            if index.size>1:
                index = index[0]
            vertexID[nodeindexp1]=localvertexID[index]
            vertexpositionwithinthecell[nodeindexp1]=localvertexpositionwithinthecell[index]
            distances[nodeindexp1]=localdistances[index]

            # compute the angle
            x = mesh.VertexCoor[localvertexID[index]][0]
            y = mesh.VertexCoor[localvertexID[index]][1]
            [angle, xint, yint] = findangle(xintersection[nodeindex], yintersection[nodeindex], xintersection[nodeindexp1], yintersection[nodeindexp1], x, y)
            angles[nodeindexp1]=angle
            xintersectionsfromzerovertex.append(xint)
            yintersectionsfromzerovertex.append(yint)

        listofTIPcellsONLY=np.asarray(listofTIPcells,dtype=int)
        vertexpositionwithinthecellTIPcellsONLY = np.asarray(vertexpositionwithinthecell,dtype=int)
        distancesTIPcellsONLY=np.copy(distances)
        anglesTIPcellsONLY=np.copy(angles)
        vertexIDTIPcellsONLY=np.copy(vertexID)

        # find the cells that have been passed completely by the front
        # you can find them by this reasoning:
        # [cells where LS<0] - [cells at the previous channell (meaning ribbon+fracture)] - [tip cells]
        #
        temp = sgndDist_k[range(0,len(sgndDist_k))]
        temp[temp > 0] = 0
        fullyfractured = np.nonzero(temp)
        fullyfractured = np.setdiff1d(fullyfractured,eltsChannel)
        fullyfractured = np.setdiff1d(fullyfractured,listofTIPcells)
        if len(fullyfractured) > 0:
            fullyfractured_angle=[]
            fullyfractured_distance = []
            fullyfractured_vertexID = []
            fullyfractured_vertexpositionwithinthecell = []
            # loop over the fullyfractured cells
            for fullyfracturedcell in range(0, len(fullyfractured)):
                i = fullyfractured[fullyfracturedcell]
                """
                you are in cell i
                take the level set at the center of the neighbors cells 
                  _   _   _   _   _   _
                | _ | _ | _ | _ | _ | _ |
                | _ | _ | _ | _ | _ | _ |
                | _ | e | a | f | _ | _ |
                | _ | _ 3 _ 2 _ | _ | _ |              
                | _ | d | i | b | _ | _ |
                | _ | _ 0 _ 1 _ | _ | _ |
                | _ | h | c | g | _ | _ |
                | _ | _ | _ | _ | _ | _ |
                
                                        0     1      2      3
                NeiElements[i]->[left, right, bottom, up]
                """

                a = mesh.NeiElements[i, top_elem]
                b = mesh.NeiElements[i, right_elem]
                c = mesh.NeiElements[i, bottom_elem]
                d = mesh.NeiElements[i, left_elem]
                e = mesh.NeiElements[d, top_elem]
                f = mesh.NeiElements[b, top_elem]
                g = mesh.NeiElements[b, bottom_elem]
                h = mesh.NeiElements[d, bottom_elem]

                hcid = sgndDist_k[[h, c, i, d]]
                cgbi = sgndDist_k[[c, g, b, i]]
                ibfa = sgndDist_k[[i, b, f, a]]
                diae = sgndDist_k[[d, i, a, e]]
                LS = [hcid, cgbi, ibfa, diae]
                hcid_mean = np.mean(np.asarray(sgndDist_k[[h, c, i, d]]))
                cgbi_mean = np.mean(np.asarray(sgndDist_k[[c, g, b, i]]))
                ibfa_mean = np.mean(np.asarray(sgndDist_k[[i, b, f, a]]))
                diae_mean = np.mean(np.asarray(sgndDist_k[[d, i, a, e]]))
                LS_means = [hcid_mean, cgbi_mean, ibfa_mean, diae_mean]
                localvertexpositionwithinthecell = np.argmin(np.asarray(LS_means))
                fullyfractured_vertexpositionwithinthecell.append(localvertexpositionwithinthecell)
                fullyfractured_distance.append(np.abs(LS_means[localvertexpositionwithinthecell]))
                fullyfractured_vertexID.append(mesh.Connectivity[i, localvertexpositionwithinthecell])
                chosenLS=LS[localvertexpositionwithinthecell ]
                # compute the angle
                dLSdy = 0.5 * mesh.hy * (chosenLS[3] + chosenLS[2] - chosenLS[1] - chosenLS[0])
                dLSdx = 0.5 * mesh.hx * (chosenLS[2] + chosenLS[1] - chosenLS[3] - chosenLS[0])
                if   dLSdy ==0. and dLSdx !=0. :
                    fullyfractured_angle.append(0.)
                elif dLSdy !=0. and dLSdx ==0 :
                    fullyfractured_angle.append(np.pi())
                elif dLSdy != 0. and dLSdx != 0:
                    fullyfractured_angle.append(np.arctan(np.abs(dLSdy)/np.abs(dLSdx)))
                else:
                    print("ERROR minimum of the function has been found, not expected")
            # finally append these informations to what computed before

            listofTIPcells=listofTIPcells+np.ndarray.tolist(fullyfractured)
            distances=distances+fullyfractured_distance
            angles=angles+fullyfractured_angle

            vertexpositionwithinthecell=vertexpositionwithinthecell+fullyfractured_vertexpositionwithinthecell
            vertexID=vertexID+fullyfractured_vertexID


        # find the new ribbon cells
        newRibbon = np.unique(np.ndarray.flatten(mesh.NeiElements[listofTIPcellsONLY, :]))
        temp = sgndDist_k[newRibbon]
        temp[temp > 0] = 0
        newRibbon = newRibbon[np.nonzero(temp)]
        newRibbon = np.setdiff1d(newRibbon, np.asarray(listofTIPcellsONLY))

        if len(xintersection)==0:
            raise SystemExit('ERROR: front not reconstructed')

        # A = np.full(mesh.NumberOfElts, np.nan)
        # A[anularegion] = sgndDist_k[anularegion]
        # from visualization import plot_fracture_variable_as_image
        # figure = plot_fracture_variable_as_image(A, mesh)
        # ax = figure.get_axes()[0]
        # xtemp = xintersection
        # ytemp = yintersection
        # xtemp.append(xtemp[0]) # close the front
        # ytemp.append(ytemp[0]) # close the front
        # # plt.plot(mesh.VertexCoor[mesh.Connectivity[Ribbon,0],0], mesh.VertexCoor[mesh.Connectivity[Ribbon,0],1], '.',color='violet')
        # plt.plot(xtemp, ytemp, '-o')
        # n=len(xintersectionsfromzerovertex)
        # for i in range(0,n) :
        #     plt.plot([mesh.VertexCoor[vertexID[(i+1)%n], 0], xintersectionsfromzerovertex[i]], [mesh.VertexCoor[vertexID[(i+1)%n], 1], yintersectionsfromzerovertex[i]], '-r')
        # # plt.plot(xred, yred, '.',color='red' )
        # # plt.plot(xgreen, ygreen, '.',color='yellow')
        # plt.plot(xblack, yblack, '.',color='black')
        # plt.plot(mesh.CenterCoor[newRibbon,0], mesh.CenterCoor[newRibbon,1], '.',color='orange')
        # #plt.plot(mesh.CenterCoor[Ribbon,0], mesh.CenterCoor[Ribbon,1], '.',color='b')
        # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10, mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')
        # plt.plot(mesh.VertexCoor[vertexID, 0], mesh.VertexCoor[vertexID, 1], '.', color='red')
        # plt.plot(xintersectionsfromzerovertex, yintersectionsfromzerovertex, '.', color='red')

        # from utility import plot_as_matrix
        # K = np.zeros((mesh.NumberOfElts,), )
        # K[listofTIPcells] = angles
        # plot_as_matrix(K, mesh)

        # from utility import plot_as_matrix
        # K = np.zeros((mesh.NumberOfElts,), )
        # K[listofTIPcells] = distances
        # plot_as_matrix(K, mesh)

        # from utility import plot_as_matrix
        # K = np.zeros((Fr_kplus1.mesh.NumberOfElts,), )
        # K[Fr_kplus1.EltTip] = Fr_kplus1.alpha
        # plot_as_matrix(K, Fr_kplus1.mesh)

        # from utility import plot_as_matrix
        # K = np.zeros((Fr_kplus1.mesh.NumberOfElts,), )
        # K[Fr_kplus1.EltTip] = Fr_kplus1.ZeroVertex
        # plot_as_matrix(K, Fr_kplus1.mesh)

        # from utility import plot_as_matrix
        # K = np.zeros((Fr_lstTmStp.mesh.NumberOfElts,), )
        # K[EltTip_k] = zrVertx_k
        # plot_as_matrix(K, Fr_lstTmStp.mesh)

        # Cells status list store the status of all the cells in the domain
        # update ONLY the position of the tip cells
        CellStatusNew = np.zeros(mesh.NumberOfElts, int)
        CellStatusNew[eltsChannel] = 1
        CellStatusNew[listofTIPcells] = 2
        CellStatusNew[Ribbon] = 3

        # mesh.identify_elements(listofTIPcellsONLY)
        # test=listofTIPcellsONLY
        # test1=listofTIPcellsONLY
        # for j in range(1,len(listofTIPcellsONLY)):
        #     element=listofTIPcellsONLY[j]
        #     test1[j]=mesh.Connectivity[element][vertexpositionwithinthecellTIPcellsONLY[j]]
        #     test[j]=vertexIDTIPcellsONLY[j]-mesh.Connectivity[element][vertexpositionwithinthecellTIPcellsONLY[j]]
        # from utility import plot_as_matrix
        # K = np.zeros((mesh.NumberOfElts,), )
        # K[listofTIPcellsONLY] = test1
        # plot_as_matrix(K, mesh)

        return np.asarray(listofTIPcells),np.asarray(listofTIPcellsONLY) , np.asarray(distances), np.asarray(angles), CellStatusNew, newRibbon, vertexpositionwithinthecell, vertexpositionwithinthecellTIPcellsONLY
        # return np.asarray(listofTIPcells),np.asarray(listofTIPcellsONLY) , np.asarray(distancesTIPcellsONLY), np.asarray(anglesTIPcellsONLY), np.asarray(CellStatusNew), np.asarray(newRibbon), vertexpositionwithinthecell, vertexpositionwithinthecellTIPcellsONLY


def UpdateListsFromContinuousFrontRec(newRibbon, listofTIPcells, sgndDist_k, zrVertx_k, mesh):

        EltChannel_k = np.setdiff1d(np.where(sgndDist_k<0)[0], listofTIPcells)
        EltTip_k = listofTIPcells
        EltCrack_k = np.concatenate((listofTIPcells, EltChannel_k))
        if np.unique(EltCrack_k).size != EltCrack_k.size:
            raise SystemExit('ERROR: the front is entering more than 1 time the same cell')
        EltRibbon_k = newRibbon

        # Cells status list store the status of all the cells in the domain
        # update ONLY the position of the tip cells
        CellStatus_k = np.zeros(mesh.NumberOfElts, int)
        CellStatus_k[EltChannel_k] = 1
        CellStatus_k[EltTip_k] = 2
        CellStatus_k[EltRibbon_k] = 3
        # from utility import plot_as_matrix
        # K = np.zeros((mesh.NumberOfElts,), )
        # plot_as_matrix(CellStatus_k, mesh)
        return   EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k, CellStatus_k