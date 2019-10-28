import numpy as np
import matplotlib.pyplot as plt


def distance(x1, y1, x2, y2):
    """
    Compute the euclidean distance between two points

    Args:
    :param x1: coordinate x first point
    :param y1: coordinate y first point
    :param x2: coordinate x second point
    :param y2: coordinate y second point

    Returns:
    :return: euclidean distance

    """
    return np.sqrt((-x1 + x2)**2 + (-y1 + y2)**2)


def pointtolinedistance(x1, y1, x2, y2, x0, y0):
    """
    Compute the minimum distance from a point of coordinates (x0,y0) to a the line passing through 2 points.
    The function works only for planar problems.

    Args:
    :param x0: coordinate x point from where to compute the distance to the line
    :param y0: coordinate y first from where to compute the distance to the line
    :param x1: coordinate x first point that defines the line
    :param y1: coordinate y first point that defines the line
    :param x2: coordinate x second point that defines the line
    :param y2: coordinate y second point that defines the line

    Returns:
    :return: distance

    """
    return np.abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/distance(x1, y1, x2, y2)


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
        ___|______|____
           |      |
        ___*______*____
           |      |   
        ___|______|____
           |      |
        In this situation take the element with LS>0 as tip
        """

        nodeVScommonelementtable[nodeindex,column]=common[np.argmin(sgndDist_k[common])]
    elif len(common) == 0:
        print('ERROR: two consecutive nodes does not belongs to a common cell --->PLEASE STOP THE PROGRAM')
    return nodeVScommonelementtable

def ISinsideFracture(x,y,xintersections,yintersections):
    """
    Given a polygon this routine will check if a given point is inside it

    :param x: coordinate of the point to be checked
    :param y: coordinate of the point to be checked
    :param xintersections: x coordinate of a point at the crack front -> the point are ordered proceding along the front
    :param yintersections: y coordinate of a point at the crack front -> the point are ordered proceding along the front
    :return: True or False depending if the point is inside the polygon or not
    """
    NumberOfIntersections = 0
    # Loop over edges
    for i in range(0, len(xintersections)):
        x1 = xintersections[i]
        y1 = yintersections[i]
        if i==(len(xintersections)-1):
            x2 = xintersections[0]
            y2 = yintersections[0]
        else:
            x2 = xintersections[i+1]
            y2 = yintersections[i+1]
        check = np.argmax([x1, x2])
        if x <= [x1, x2][check]:
            if x >= [x2, x1][check]:
                m = (y1-y2)/(x1-x2)
                if y <= (m*x+y1-m*x1):
                    if x == x1 or x == x2:
                        NumberOfIntersections = NumberOfIntersections + 0.5
                    else:
                        NumberOfIntersections = NumberOfIntersections + 1
    # check if is odd or even
    if not(NumberOfIntersections % 2):
        insideapolygon = False
    else:
        insideapolygon = True
    return insideapolygon

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
    fmachineprec = np.finfo(float).eps
    if np.abs(x2 - x1)/np.maximum(np.abs(x2),np.abs(x1)) < fmachineprec:  # the front is a vertical line
        x = x2
        y = y0
        angle = 0
    elif np.abs(y2 - y1)/np.maximum(np.abs(y2),np.abs(y1)) < fmachineprec:  # the front is an horizontal line
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
        fprecision = np.float64
        iprecision = int
        fmachineprec = 100*np.finfo(float).eps
        """
        ------------------
        0) Set all the LS==0 to -fmachineprec
        In this way we avoid to deal with the situations where the front is crossing exactly an element vertex
        ------------------        
        """

        zerovertexes = np.where(sgndDist_k == 0)[0]
        if len(zerovertexes) > 0:
            sgndDist_k[zerovertexes] = -fmachineprec

        """

        there is NO vertex of the fictitius cell where the LS is exactly zero!!!    
        all the nodes have LS either > or < 0                                       



        ------------------
        1) arrays initialization:
        ------------------
        """

        icba = np.empty(4, dtype=iprecision)
        icbaMat = np.empty([2, 2], dtype=iprecision)
        abcdefgh = np.empty(8, dtype=iprecision)
        alphaX = np.empty(4, dtype=fprecision)
        alphaY = np.empty(4, dtype=fprecision)

        # only for plotting purposes:
        orderallx = np.asarray([1, 0, 3, 2], dtype=iprecision)
        orderally = np.asarray([3, 2, 1, 0], dtype=iprecision)
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
        ------------------
        2)Find the first fictitius cell where the front is passing.
        The Boundary Element Method discretizes the domain in subcells.
        A fictitius cell is generated by the union of the centers of 4 cells
        ------------------        
        """

        # start from a cell in the ribbon, so we are sure that in cell i the LS < 0
        i = Ribbon[0]
        # count the number of trials
        examinatedcells = 0
        # if the first fictitius cell with positive and negative values of the level set has been found
        found = False
        [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]

        while found is False and examinatedcells < len(anularegion):
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
            NeiElements[element]->[left, right, bottom, up]
            """

            a = mesh.NeiElements[i, top_elem]
            b = mesh.NeiElements[i, right_elem]
            c = mesh.NeiElements[i, bottom_elem]
            d = mesh.NeiElements[i, left_elem]
            e = mesh.NeiElements[d, top_elem]
            f = mesh.NeiElements[b, top_elem]
            g = mesh.NeiElements[b, bottom_elem]
            h = mesh.NeiElements[d, bottom_elem]
            abcdefgh[:] = [a, b, c, d, e, f, g, h]
            elements = [i, i, c, d, d, i, c, h]
            LS = sgndDist_k[abcdefgh]

            if not np.all(LS < 0):
                """
                the front is passing for the big set of elements abcdefgh
                reduce to a fictitious cell where at least one node has LS<0 and  at least one node has LS>0
                include always i because we know it belongs to the ribbon cells
                """
                index = np.argmax(LS)  # index of the element with positive LS
                i = elements[index]
                found = True

            elif np.all(LS < 0):
                """
                the front is not passing for the big set of elements abcdefgh
                take a new set of elements centered in the element of the current set with the higher value of LS
                in this way we are moving closer to the front.                
                """

                index = np.argmax(LS)
                i = elements[index]
                examinatedcells = examinatedcells + 1
            if examinatedcells > len(anularegion):
                print('ERROR: first cell where the front is passing trough, not found')
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
        
        remembrer the usage of NeiElements[element]->[left, right, bottom, up].
        """
        a = mesh.NeiElements[i, top_elem]
        c = mesh.NeiElements[i, right_elem]
        b = mesh.NeiElements[i + 1, top_elem]
        icba[:] = [i, c, b, a]
        LS = sgndDist_k[icba]

        """
        take the vertical and horizontal lines passing through the center of the fictitious cell
        2 intersections with edges, just take the first
        LS[m]*LS[n] if both with the same sign then the product will be always positive
        """

        if (LS[0]*LS[1]) < 0:
            c1=mesh.CenterCoor[i, 0]
            c2=mesh.CenterCoor[c, 0]
            x_controlnode = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[0])+np.abs(LS[1]))
            y_controlnode = mesh.CenterCoor[i, 1]
            # intersection on the lower fictitious edge i-c
            # =>change the current fictitious element to the bottom element
            element = mesh.NeiElements[i, bottom_elem]
        elif (LS[1]*LS[2]) < 0 :
            c1=mesh.CenterCoor[c, 1]
            c2=mesh.CenterCoor[b, 1]
            x_controlnode = mesh.CenterCoor[c, 0]
            y_controlnode = c1+np.abs(LS[1])*(c2-c1)/(np.abs(LS[1])+np.abs(LS[2]))
            # intersection on the right fictitious edge c-b
            # =>change the current fictitious element to the right element
            element = c
        elif (LS[2]*LS[3]) < 0 :
            c1=mesh.CenterCoor[a, 0]
            c2=mesh.CenterCoor[b, 0]
            x_controlnode = c1+np.abs(LS[3])*(c2-c1)/(np.abs(LS[2])+np.abs(LS[3]))
            y_controlnode = mesh.CenterCoor[a, 1]
            # intersection on the top fictitious edge a-b
            # =>change the current fictitious element to the upper element
            element =a
        elif (LS[0]*LS[3]) < 0 :
            c1=mesh.CenterCoor[i, 1]
            c2=mesh.CenterCoor[a, 1]
            x_controlnode = mesh.CenterCoor[i, 0]
            y_controlnode = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[3])+np.abs(LS[0]))
            # intersection on the left fictitious edge i-a
            # =>change the current fictitious element to the element below
            element = mesh.NeiElements[i, left_elem]

        """
        ------------------
        3) Go outside from the previous fictitious cell i,
           find all the intersections,
           follow the front and find the next fictitious cell:
        ------------------
        """

        baseXcontrolnode = x_controlnode
        baseYcontrolnode = y_controlnode
        frontCompleted = 'no'  # if the front has been completed
        counter = 0
        while frontCompleted == 'no' and counter < 2*mesh.NumberOfElts:
            counter = counter+1
            elementstorage.append(element)
            up = mesh.NeiElements[element, top_elem]
            right = mesh.NeiElements[element, right_elem]
            rightUp = mesh.NeiElements[element+1, top_elem]
            icba = np.asarray([element, right, rightUp, up], dtype=iprecision)
            LS = sgndDist_k[icba]

            icbaMat[0, :] = [element, right]
            icbaMat[1, :] = [up, rightUp]
            """
            icbaMat:
             _               _
            |                 |
            |element ; right  | 
            |                 |
            |   up   ; rightup|
            |_               _|
            
            """

            if not(np.all(LS > 0) or np.all(LS < 0)):
                # the front is crossing the cell.
                # take the vertical and horizontal line passing trough the center of the cell
                centernode = mesh.Connectivity[element, 2]
                xgrid = mesh.VertexCoor[centernode, 0]
                ygrid = mesh.VertexCoor[centernode, 1]
                allx = np.asarray([mesh.CenterCoor[right, 0], mesh.CenterCoor[element, 0], mesh.CenterCoor[up, 0], mesh.CenterCoor[rightUp, 0]], dtype=fprecision)
                ally = np.asarray([mesh.CenterCoor[up, 1], mesh.CenterCoor[rightUp, 1], mesh.CenterCoor[right, 1], mesh.CenterCoor[element, 1]], dtype=fprecision)

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
                    # xgrid=mesh.CenterCoor[element,0]+mesh.hx/2
                    # ygrid=mesh.CenterCoor[element,1]+mesh.hy/2
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

                        if (yCandidate - np.max(ally)) / maxabmax <= +fmachineprec / maxabmax and (
                                yCandidate - np.min(ally)) / maxabmin >= -fmachineprec / maxabmin:
                            """
                            if yCandidate <= np.max(ally) and yCandidate >= np.min(ally) => the point is inside the cell
                            ii) find if the point is between element and right <OR> up and rightup

                             ____________/_______
                            |          |/        |
                            |      up  o right up|
                            |_________/|_________|
                            |        / |         |
                            |  element | right   |
                            |______/___|_________|
                                    /
                             ______________/_____
                            |         |   /      |
                            |     up  |right  up |
                            |_________|_/________|
                            |         |/         |
                            | element o   right  |
                            |________/|__________|
                                      /
                                       icbaMat[0,:]=[element,right]
                                       icbaMat[1,:]=[up,rightUp]              
                            """
                            if (yCandidate - ygrid) / np.maximum(np.abs(yCandidate), np.abs(ygrid)) < 0:  # element and right
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
                            | element  |   right  |
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
                        if (xCandidate - np.max(allx)) / maxabmax <= +fmachineprec / maxabmax and (
                                xCandidate - np.min(allx)) / maxabmin >= -fmachineprec / maxabmin:
                            """
                            if xCandidate  <= np.max(allx) and xCandidate >= np.min(allx) =>the point is inside the cell
                            ii) find if the point is between element and up <OR> right and rightup

                             _____/______________
                            |    /     |         |
                            |   /  up  | right up|
                            |__o_______|_________|
                            | /        |         |
                            |/ element | right   |
                            |__________|_________|

                             _________________/__
                            |         |      /   |
                            |     up  |right/ up |
                            |_________|____o_____|
                            |         |   /      |
                            | element |  /right  |
                            |_________|_/________|
                                       /
                            icbaMat[0,:]=[element,right]
                            icbaMat[1,:]=[up,rightUp]                            
                            """

                            if (xCandidate - xgrid) / np.maximum(np.abs(xCandidate), np.abs(xgrid)) < 0:
                                # element and up
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
                              | element |   right  |
                              |_________|__________|

                            checking flags                            
                            """

                            appendXcandidate = 'no'
                            Xposition = 'none'
                    """
                    1 or 2 intersections exist 
                    check the element if ia s ribbon cell.
                    
                    icbaMat:
                     _               _
                    |                 |
                    |element ; right  | 
                    |                 |
                    |   up   ; rightup|
                    |_               _|
                    
                    Yposition <=> down (0) / up (1)
                    Xposition <=> left (0) / right (1)
                    
                     _               _
                    |        |        |
                    |element | right  | 
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
                            dXTOcontrolnode=distance(xCandidate, ygrid, x_controlnode, y_controlnode)
                            dYTOcontrolnode=distance(xgrid, yCandidate, x_controlnode, y_controlnode)
                            #if (dXTOcontrolnode-dYTOcontrolnode)/np.maximum(dXTOcontrolnode,dYTOcontrolnode)< fmachineprec/np.maximum(dXTOcontrolnode,dYTOcontrolnode): #write first x intersection
                            if dXTOcontrolnode < dYTOcontrolnode:  # write first x intersection
                                xintersection.append(xCandidate)
                                yintersection.append(ygrid)
                                typeindex.append(0)
                                if Xposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[element], mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                                elif Xposition==1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                xintersection.append(xgrid)
                                yintersection.append(yCandidate)
                                typeindex.append(0)
                                if Yposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[element], mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                elif Yposition == 1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                            else: # write first y intersection
                                xintersection.append(xgrid)
                                yintersection.append(yCandidate)
                                typeindex.append(0)
                                if Yposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[element], mesh.Connectivityelemedges[right])
                                    edgeORvertexID.append(edge[0])
                                elif Yposition == 1:
                                    edge=np.intersect1d(mesh.Connectivityelemedges[rightUp],mesh.Connectivityelemedges[up])
                                    edgeORvertexID.append(edge[0])
                                xintersection.append(xCandidate)
                                yintersection.append(ygrid)
                                typeindex.append(0)
                                if Xposition == 0:
                                    edge = np.intersect1d(mesh.Connectivityelemedges[element],
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
                                        edge = np.intersect1d(mesh.Connectivityelemedges[element], mesh.Connectivityelemedges[up])
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
                                        edge = np.intersect1d(mesh.Connectivityelemedges[element], mesh.Connectivityelemedges[right])
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
                compute the new controlnode and the new fictitious element where the front is going and define element
                 _ _ _ _ _ _
                |_|_|_|_|_|_|
                |_|_|_|_|_|_|
                |_|_|a|b|_|_|
                |_|_|i|c|_|_|
                |_|_|_|_|_|_|
                |_|_|_|_|_|_|
                NeiElements[element]->[left, right, bottom, up].
                
                2 intersections with the edges of the fictitius cell, TAKE THE ONE THAT IS NOT ALREADY SELECTED
                """

                found_next_controlnode='no'
                if (LS[0]*LS[1] < 0) and found_next_controlnode == 'no':
                    # intersection on the lower fictitius edge i-c
                    c1 = mesh.CenterCoor[element, 0]
                    c2 = mesh.CenterCoor[right, 0]
                    x_controlnode_temp = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[0])+np.abs(LS[1]))
                    y_controlnode_temp = mesh.CenterCoor[element, 1]
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius element
                        element = mesh.NeiElements[element, bottom_elem]
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
                        # ---change the current fictitius element
                        element = right
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
                        # ---change the current fictitius element
                        element = up
                        found_next_controlnode = 'yes'
                if (LS[0]*LS[3] < 0) and found_next_controlnode=='no':
                    # intersection on the left fictitius edge i-a
                    c1 = mesh.CenterCoor[element, 1]
                    c2 = mesh.CenterCoor[up, 1]
                    x_controlnode_temp = mesh.CenterCoor[element, 0]
                    y_controlnode_temp = c1+np.abs(LS[0])*(c2-c1)/(np.abs(LS[3])+np.abs(LS[0]))
                    if x_controlnode_temp != x_controlnode or y_controlnode_temp != y_controlnode:
                        x_controlnode = x_controlnode_temp
                        y_controlnode = y_controlnode_temp
                        # ---change the current fictitius element
                        element = mesh.NeiElements[element, left_elem]
                        found_next_controlnode = 'yes'
                if found_next_controlnode == 'no':
                    print('ERROR: next fictitius element not found. INFINITE LOOP STARTED! Please Stop me (1)')

                #make the if condition for closing the loop
                if baseYcontrolnode==y_controlnode and baseXcontrolnode==x_controlnode:
                    frontCompleted = 'yes'
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
        """
        ------------------
        4)  Make a 2D table for each node found at the front: the first column contains the cell's name common with the 
            previous node while the second column the cell's name common with the next node.
            the nodes that have to be deleted will have same value in both column
        ------------------  
                  
          
        """
        nodeVScommonelementtable=np.empty([len(xintersection), 2],dtype=int)
        for nodeindex in range(0,len(xintersection)):
            # commonbackward contains the unique values in cellOfNodei that are in cellOfNodeim1.
            if nodeindex== 0:
                commonbackward = findcommon(nodeindex,len(xintersection)-1, typeindex, mesh.Connectivityedgeselem,mesh.Connectivitynodeselem, edgeORvertexID)
            else:
                commonbackward = findcommon(nodeindex, (nodeindex - 1), typeindex, mesh.Connectivityedgeselem,mesh.Connectivitynodeselem, edgeORvertexID)
            # commonforward contains the unique values in cellOfNodei that are in cellOfNodeip1.
            if nodeindex== len(xintersection)-1:
                commonforward=findcommon(nodeindex, 0,typeindex,mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
            else:
                commonforward = findcommon(nodeindex, (nodeindex + 1), typeindex, mesh.Connectivityedgeselem,mesh.Connectivitynodeselem, edgeORvertexID)

            column=0
            nodeVScommonelementtable=filltable(nodeVScommonelementtable,nodeindex,commonbackward,sgndDist_k,column)
            column=1
            nodeVScommonelementtable=filltable(nodeVScommonelementtable,nodeindex,commonforward,sgndDist_k,column)

        listofTIPcells = []
        # remove the nodes in the cells with more than 2 nodes and keep the first and the last node
        counter = 0
        for nodeindex in range(0, len(xintersection)):
            if nodeVScommonelementtable[nodeindex][1] == nodeVScommonelementtable[nodeindex][0]:
                del xintersection[nodeindex-counter]
                del yintersection[nodeindex-counter]
                del typeindex[nodeindex-counter]
                del edgeORvertexID[nodeindex-counter]
                counter = counter+1
            else:
                listofTIPcells.append(nodeVScommonelementtable[nodeindex][0])

        """
        This is another way of computing the tip cells
        - not used anymore -
        """
        # i=0
        # if len(typeindex) > 1:
        #     for i in range(0,len(typeindex)):
        #         cellscommontolastnodeadded = []
        #         cellscommontosecondlastnodeadded = []
        #         if (i+1)>len(typeindex)-1:
        #             j=0
        #         else:
        #             j=i+1
        #         # compute the tip cell defined by the last 2 points added in the list
        #         if typeindex[i] == 0:  # is an edge
        #             elms=np.unique(mesh.Connectivityedgeselem[edgeORvertexID[i]])
        #             cellscommontolastnodeadded.append(elms[0])
        #             cellscommontolastnodeadded.append(elms[1])
        #         elif typeindex[i] == 1:  # is a vertex
        #             elms = np.unique(mesh.Connectivitynodeselem[edgeORvertexID[i]])
        #             cellscommontolastnodeadded.append(elms[0])
        #             cellscommontolastnodeadded.append(elms[1])
        #             cellscommontolastnodeadded.append(elms[2])
        #             cellscommontolastnodeadded.append(elms[3])
        #         #
        #         if typeindex[j] == 0:  # is an edge
        #             elms = np.unique(mesh.Connectivityedgeselem[edgeORvertexID[j]])
        #             cellscommontosecondlastnodeadded.append(elms[0])
        #             cellscommontosecondlastnodeadded.append(elms[1])
        #         elif typeindex[j] == 1:  # is a vertex
        #             elms = np.unique(mesh.Connectivitynodeselem[edgeORvertexID[j]])
        #             cellscommontosecondlastnodeadded.append(elms[0])
        #             cellscommontosecondlastnodeadded.append(elms[1])
        #             cellscommontosecondlastnodeadded.append(elms[2])
        #             cellscommontosecondlastnodeadded.append(elms[3])
        #         cellscommontosecondlastnodeadded=np.asarray(cellscommontosecondlastnodeadded)
        #         cellscommontolastnodeadded = np.asarray(cellscommontolastnodeadded)
        #         listofTIPcells.append(np.intersect1d(cellscommontosecondlastnodeadded, cellscommontolastnodeadded)[0])
        """
        ------------------
        5)  Define the correct node from where compute the distance to the front
            that node should have the larger distance from the front and being inside the fracture but belonging to the tip cell
            define the angle
        ------------------  
        """
        vertexpositionwithinthecell=[]
        vertexID = []
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
            for i in range(0,4):
                node = mesh.Connectivity[listofTIPcells[nodeindexp1]][i]
                x = mesh.VertexCoor[node][0]
                y = mesh.VertexCoor[node][1]
                if ISinsideFracture(x, y, xintersection, yintersection):
                    localvertexID.append(node)
                    localvertexpositionwithinthecell.append(i)
                    localdistances.append(pointtolinedistance(xintersection[nodeindex], yintersection[nodeindex], xintersection[nodeindexp1], yintersection[nodeindexp1], x, y))

            # take the largest distance from the front
            index = np.argmax(np.asarray(localdistances))
            vertexID.append(localvertexID[index])
            vertexpositionwithinthecell.append(localvertexpositionwithinthecell[index])
            distances[nodeindexp1]=localdistances[index]

            # compute the angle
            x = mesh.VertexCoor[localvertexID[index]][0]
            y = mesh.VertexCoor[localvertexID[index]][1]
            [angle, xint, yint] = findangle(xintersection[nodeindex], yintersection[nodeindex], xintersection[nodeindexp1], yintersection[nodeindexp1], x, y)
            angles[nodeindexp1]=angle
            xintersectionsfromzerovertex.append(xint)
            yintersectionsfromzerovertex.append(yint)

        # find the new ribbon cells
        newRibbon = np.unique(np.ndarray.flatten(mesh.NeiElements[listofTIPcells, :]))
        temp = sgndDist_k[newRibbon]
        temp[temp > 0] = 0
        newRibbon = newRibbon[np.nonzero(temp)]
        newRibbon = np.setdiff1d(newRibbon, np.asarray(listofTIPcells), assume_unique=True)

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
        # for i in range(0,len(xintersectionsfromzerovertex)) :
        #     plt.plot([mesh.VertexCoor[vertexID[i], 0], xintersectionsfromzerovertex[i]], [mesh.VertexCoor[vertexID[i], 1], yintersectionsfromzerovertex[i]], '-r')
        # # plt.plot(xred, yred, '.',color='red' )
        # # plt.plot(xgreen, ygreen, '.',color='yellow')
        # plt.plot(xblack, yblack, '.',color='black')
        # plt.plot(mesh.CenterCoor[Ribbon,0], mesh.CenterCoor[Ribbon,1], '.',color='orange')
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

        # Cells status list store the status of all the cells in the domain
        # update ONLY the position of the tip cells
        CellStatusNew = np.zeros(mesh.NumberOfElts, int)
        CellStatusNew[eltsChannel] = 1
        CellStatusNew[listofTIPcells] = 2
        CellStatusNew[Ribbon] = 3
        
        return np.asarray(listofTIPcells), np.asarray(distances), np.asarray(angles), np.asarray(CellStatusNew), np.asarray(newRibbon), np.asarray(listofTIPcells), vertexpositionwithinthecell


def UpdateListsFromContinuousFrontRec(newRibbon, listofTIPcells, sgndDist_k, zrVertx_k, mesh):

        EltChannel_k = np.setdiff1d(np.where(sgndDist_k<0)[0], listofTIPcells)
        EltTip_k = listofTIPcells
        EltCrack_k = np.concatenate((listofTIPcells, EltChannel_k))
        EltRibbon_k = newRibbon

        # Cells status list store the status of all the cells in the domain
        # update ONLY the position of the tip cells
        CellStatus_k = np.zeros(mesh.NumberOfElts, int)
        CellStatus_k[EltChannel_k] = 1
        CellStatus_k[EltTip_k] = 2
        CellStatus_k[EltRibbon_k] = 3

        return   EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, zrVertx_k, CellStatus_k