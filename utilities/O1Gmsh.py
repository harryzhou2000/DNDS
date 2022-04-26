import numpy as np
import scipy
import scipy.sparse
import scipy.interpolate


def RBF(x, c):
    r = np.sqrt(np.sum(x*x, 1))/c
    # return numpy.exp(-(r*r))
    # return (1-r**4)*(4*r+1)*(r < 1.0)
    return (1-r)**2*(r < 1.0)

# base [N,3]


def getWXY(base, disp, c):
    # print(base)
    N = np.size(base, 0)
    dim = np.size(base, 1)
    dimdisp = np.size(disp, 1)
    Mat = np.zeros((N, N))
    for i in range(N):
        Mat[:, i] = RBF(base-np.reshape(base[i, :], (1, -1)), c)
    P = np.ones((N, 1))
    Mat = np.concatenate([Mat, P], 1)
    P = np.concatenate([P, np.zeros((1, 1))], 0)
    Mat = np.concatenate([Mat, np.transpose(P)], 0)
    WXY = np.linalg.solve(Mat, (
        np.concatenate([disp, np.zeros((1, dimdisp))], 0)))
    return WXY


def RBFInterp(base, WXY, x, c):
    N = np.size(base, 0)
    dim = np.size(base, 1)
    rads = base - np.reshape(x, (1, dim))
    rbfs = RBF(rads, c)
    rets = rbfs.transpose() @ WXY[0:N, :] + WXY[N, :]
    return rets


class MeshO12DModder:
    def __init__(self) -> None:
        pass

    def ReadFromGmshO1(self, fname):
        fin = open(fname, mode='r')
        lines = fin.readlines()
        fin.close()

        nline = len(lines)
        iline = 0
        while(iline < nline):
            words = lines[iline].split()
            if words[0] == "$MeshFormat":
                iline += 1
                words = lines[iline].split()
                while(words[0] != "$EndMeshFormat"):
                    MeshFormat = [float(item) for item in words]
                    # print(MeshFormat)
                    if(np.floor(MeshFormat[0]) != 2):
                        raise("Gmsh version not 2.x !!")
                    if(np.floor(MeshFormat[1]) != 0):
                        raise("Gmsh not ASCII !!")
                    iline += 1
                    words = lines[iline].split()

            if words[0] == "$Nodes":
                iline += 1
                words = lines[iline].split()
                self.nnodes = int(words[0])
                self.nodes = {}
                iline += 1
                words = lines[iline].split()
                while(words[0] != "$EndNodes"):
                    cnodes = [float(item) for item in words[1:]]
                    inode = int(words[0])
                    self.nodes[inode] = np.array(cnodes)
                    iline += 1
                    words = lines[iline].split()

            if words[0] == "$Elements":
                iline += 1
                words = lines[iline].split()
                self.nelems = int(words[0])
                self.elems = {}
                iline += 1
                words = lines[iline].split()
                while(words[0] != "$EndElements"):
                    cnodes = [int(item) for item in words[5:]]
                    inode = int(words[0])
                    tnode = int(words[1])
                    ntags = int(words[2])
                    if(ntags != 2):
                        raise("ntag not2 !! line %d" % (iline))
                    gnode0 = int(words[3])
                    gnode1 = int(words[4])
                    self.elems[inode] = (tnode, gnode0, gnode1, cnodes)
                    iline += 1
                    words = lines[iline].split()

            if words[0] == "$PhysicalNames":
                iline += 1
                words = lines[iline].split()
                self.nphy = int(words[0])
                self.phys = {}
                iline += 1
                words = lines[iline].split()
                while(words[0] != "$EndPhysicalNames"):
                    name = words[2][1:-1]
                    dim = int(words[0])
                    iphy = int(words[1])
                    self.phys[iphy] = [dim, name]
                    iline += 1
                    words = lines[iline].split()

            iline += 1
        self.nodes = dict(sorted(self.nodes.items(), key=lambda i: i[0]))

    def PrintO1As2dSCFVGrid(self, fname, ifPeriodic=False, ia=-1, ib=-1, offset=np.array((-1, -1))):
        fout = open(fname, mode='w')
        fout.write("%12d\n" % (1))
        # fout.write(" %10d  %10d  %10d\n" %(len(self.elems), len(self.elems), len(self.nodes), len(self.nodes)))
        itri = 0
        iquad = 0
        self.phycount = {}
        for ielem in self.elems:
            elem = self.elems[ielem]
            if(elem[0] == 1 and len(elem[3]) == 2):  # line 2
                pass
            elif(elem[0] == 2 and len(elem[3]) == 3):  # tri 3
                itri += 1
            elif(elem[0] == 3 and len(elem[3]) == 4):  # quad 4
                iquad += 1
            else:
                raise("ELEM WRONG")

            if(elem[2] in self.phycount.keys()):
                self.phycount[elem[2]].add(ielem)
            else:
                self.phycount[elem[2]] = {ielem}

        fout.write(" %10d  %10d  %10d  %10d\n" %
                   (itri + iquad, itri+iquad, len(self.nodes), len(self.nodes)))
        for ielem in self.elems:
            elem = self.elems[ielem]
            if(elem[0] == 1 and len(elem[3]) == 2):  # line 2
                continue
            elif(elem[0] == 2 and len(elem[3]) == 3):  # tri 3
                fout.write("  %10d  %10d  %10d  %10d\n" %
                           (3, elem[3][0], elem[3][1], elem[3][2]))
            elif(elem[0] == 3 and len(elem[3]) == 4):  # quad 4
                fout.write("  %10d  %10d  %10d  %10d %10d\n" %
                           (4, elem[3][0], elem[3][1], elem[3][2], elem[3][3]))
            else:
                raise("ELEM WRONG")
        for inode in self.nodes:
            node = self.nodes[inode]
            fout.write("  %22.16E  %22.16E\n" % (node[0], node[1]))

        bcs = {}
        bcsum = 0
        for iphys in self.phys:
            phy = self.phys[iphys]
            phynamewords = phy[1].split('-')
            if phynamewords[0] == 'bc' and phy[0] == 1:
                bcs[iphys] = (phynamewords[0], phynamewords[1],
                              len(self.phycount[iphys]))
                bcsum += len(self.phycount[iphys])

        fout.write("  %10d\n  %10d\n" % (len(bcs), bcsum))
        ibc = 1
        for iphys in bcs:
            fout.write("  %10d %10d\n" % (ibc, bcs[iphys][2]))
            ibc += 1
        ibc = 1
        for iphys in bcs:
            for bcelem in self.phycount[iphys]:
                fout.write("  %10d  %10d  %10d  %10d\n" %
                           (ibc, int(bcs[iphys][1]), self.elems[bcelem][3][0], self.elems[bcelem][3][1]))

            ibc += 1

        if(ifPeriodic):
            fout.write("  %10d\n" % (2))
            fout.write("  %10d  %10d  %.16g  %.16g\n" %
                       (ia, ib, offset[0], offset[1]))
            fout.write("  %10d  %10d  %.16g  %.16g\n" %
                       (ib, ia, -offset[0], -offset[1]))

        fout.close()

    def GetMediumTopo(self):
        self.faceDict = {}
        for ielem in self.elems:
            elem = self.elems[ielem]
            if(elem[0] == 1 and len(elem[3]) == 2):  # line 2
                edge = [elem[3][0], elem[3][1]]
                self.AddElem2FaceDict(edge, ielem)
            elif(elem[0] == 2 and len(elem[3]) == 3):  # tri 3
                edge = [elem[3][0], elem[3][1]]
                self.AddElem2FaceDict(edge, ielem)
                edge = [elem[3][1], elem[3][2]]
                self.AddElem2FaceDict(edge, ielem)
                edge = [elem[3][2], elem[3][0]]
                self.AddElem2FaceDict(edge, ielem)
            elif(elem[0] == 3 and len(elem[3]) == 4):  # quad 4
                edge = [elem[3][0], elem[3][1]]
                self.AddElem2FaceDict(edge, ielem)
                edge = [elem[3][1], elem[3][2]]
                self.AddElem2FaceDict(edge, ielem)
                edge = [elem[3][2], elem[3][3]]
                self.AddElem2FaceDict(edge, ielem)
                edge = [elem[3][3], elem[3][0]]
                self.AddElem2FaceDict(edge, ielem)
            else:
                raise("ELEM WRONG")
        iface = 1
        for edgeTuple in self.faceDict:
            self.faceDict[edgeTuple] = (iface, self.faceDict[edgeTuple])
            iface += 1
        self.faceDict = dict(
            sorted(self.faceDict.items(), key=lambda i: i[1][0]))

    def AddElem2FaceDict(self, edge: list, ielem):
        edge.sort()
        edge = tuple(edge)
        if edge in self.faceDict:
            self.faceDict[edge].append(ielem)
        else:
            self.faceDict[edge] = [ielem]

    def GetEdgeIface(self, edge: list):
        edge.sort()
        edge = tuple(edge)
        return self.faceDict[edge][0]

    def AddO2Points(self):
        iO2Edge = 1
        self.nodeO2Edge = {}
        for edge in self.faceDict:
            p1 = self.nodes[edge[0]]
            p2 = self.nodes[edge[1]]
            p = (p1 + p2) * 0.5
            self.nodeO2Edge[iO2Edge] = p
            iO2Edge += 1
        self.nnodesO2Edge = iO2Edge-1
        self.nodeO2Edge = dict(
            sorted(self.nodeO2Edge.items(), key=lambda i: i[0]))

        iO2Vol = 1
        self.nodeO2Vol = {}
        self.cell2volnode = {}
        for ielem in self.elems:
            elem = self.elems[ielem]
            if(elem[0] == 1 and len(elem[3]) == 2):  # line 2
                continue
            elif(elem[0] == 2 and len(elem[3]) == 3):  # tri 3
                continue
            elif(elem[0] == 3 and len(elem[3]) == 4):  # quad 4
                p1 = self.nodes[elem[3][0]]
                p2 = self.nodes[elem[3][1]]
                p3 = self.nodes[elem[3][2]]
                p4 = self.nodes[elem[3][3]]
                p = (p1 + p2 + p3 + p4) * (0.25)
                self.nodeO2Vol[iO2Vol] = p
                self.cell2volnode[ielem] = iO2Vol
                iO2Vol += 1
            else:
                raise("ELEM WRONG")
        self.nnodesO2Vol = iO2Vol-1
        self.nodeO2Vol = dict(
            sorted(self.nodeO2Vol.items(), key=lambda i: i[0]))
        self.nodesO2All = list(self.nodes.items())
        self.nodesO2All.extend([(item[0] + self.nnodes, item[1])
                               for item in self.nodeO2Edge.items()])
        self.nodesO2All.extend([(item[0] + self.nnodes + self.nnodesO2Edge, item[1])
                               for item in self.nodeO2Vol.items()])
        self.nodesO2All = dict(self.nodesO2All)

        self.elemO2s = {}
        for ielem in self.elems:
            elem = self.elems[ielem]

            if(elem[0] == 1 and len(elem[3]) == 2):  # line 2
                iface = self.GetEdgeIface([elem[3][0], elem[3][1]])
                self.elemO2s[ielem] = (8, elem[1], elem[2], [
                                       elem[3][0], elem[3][1], iface + self.nnodes])
            elif(elem[0] == 2 and len(elem[3]) == 3):  # tri 3
                iface1 = self.GetEdgeIface([elem[3][0], elem[3][1]])
                iface2 = self.GetEdgeIface([elem[3][1], elem[3][2]])
                iface3 = self.GetEdgeIface([elem[3][2], elem[3][0]])
                self.elemO2s[ielem] = (9, elem[1], elem[2], [
                                       elem[3][0], elem[3][1], elem[3][2],
                                       iface1+self.nnodes, iface2+self.nnodes, iface3+self.nnodes])

            elif(elem[0] == 3 and len(elem[3]) == 4):  # quad 4
                iface1 = self.GetEdgeIface([elem[3][0], elem[3][1]])
                iface2 = self.GetEdgeIface([elem[3][1], elem[3][2]])
                iface3 = self.GetEdgeIface([elem[3][2], elem[3][3]])
                iface4 = self.GetEdgeIface([elem[3][3], elem[3][0]])
                self.elemO2s[ielem] = (10, elem[1], elem[2], [
                                       elem[3][0], elem[3][1], elem[3][2], elem[3][3],
                                       iface1+self.nnodes, iface2+self.nnodes,
                                       iface3+self.nnodes, iface4+self.nnodes])
            else:
                raise("ELEM WRONG")

        for ielem in self.elemO2s:
            elem = self.elemO2s[ielem]
            if(elem[0] == 8 and len(elem[3]) == 3):  # line 3
                continue
            elif(elem[0] == 9 and len(elem[3]) == 6):  # tri 6
                continue

            elif(elem[0] == 10 and len(elem[3]) == 8):  # quad 9 incomplete
                self.elemO2s[ielem][3].append(
                    self.cell2volnode[ielem] + self.nnodes + self.nnodesO2Edge)
            else:
                raise("ELEM WRONG")

    def PrintO2(self, fname):
        fout = open(fname, mode='w')
        fout.write("$MeshFormat\n" + " 2.2 0 8\n" + "$EndMeshFormat\n")
        fout.write("$Nodes\n%d\n" %
                   (self.nnodes + self.nnodesO2Edge + self.nnodesO2Vol))
        # for i in self.nodes:
        #     fout.write("%6d %.16g %.16g %.16g\n" % (i,
        #                                             self.nodes[i][0], self.nodes[i][1], self.nodes[i][2]))
        # for i in self.nodeO2Edge:
        #     fout.write("%6d %.16g %.16g %.16g\n" % (i + self.nnodes,
        #                                             self.nodeO2Edge[i][0], self.nodeO2Edge[i][1], self.nodeO2Edge[i][2]))
        # for i in self.nodeO2Vol:
        #     fout.write("%6d %.16g %.16g %.16g\n" % (i + self.nnodes + self.nnodesO2Edge,
        #                                             self.nodeO2Vol[i][0], self.nodeO2Vol[i][1], self.nodeO2Vol[i][2]))
        for i in self.nodesO2All:
            fout.write("%6d %.16g %.16g %.16g\n" % (i,
                                                    self.nodesO2All[i][0], self.nodesO2All[i][1], self.nodesO2All[i][2]))
        fout.write("$EndNodes\n")

        fout.write("$Elements\n%d\n" % (self.nelems))
        for ielem in self.elemO2s:
            elem = self.elemO2s[ielem]
            fout.write("   %d %d %d %d %d " %
                       (ielem, elem[0], 2, elem[1], elem[2]))
            for inode in elem[3]:
                fout.write("%.6d " % inode)
            fout.write('\n')
        fout.write("$EndElements\n")

        fout.write("$PhysicalNames\n%d\n" % (self.nphy))
        for ip in self.phys:
            fout.write("%d %d \"%s\"\n" % (self.phys[ip][0], ip, self.phys[ip][1]))
        fout.write("$EndPhysicalNames\n")

    def SmoothEdge(self, bname, crbf, thetaMax):
        for ip in self.phys:
            if bname == self.phys[ip][1]:
                self.iphySmooth = ip
                if(self.phys[ip][0] != 1):
                    raise("Can Only smooth 1d phys group!")
                break
        else:
            raise("boundary %s not found! " % (bname))

        self.elemSmooth = {}
        self.smoothNodes = {}
        for ielem in self.elemO2s:
            if(self.elemO2s[ielem][1] != self.iphySmooth):
                continue
            self.elemSmooth[ielem] = self.elemO2s[ielem]
            elem = self.elemO2s[ielem]
            if(elem[3][0] in self.smoothNodes):
                self.smoothNodes[elem[3][0]].append(elem[3][2])
            else:
                self.smoothNodes[elem[3][0]] = [elem[3][2]]

            if(elem[3][1] in self.smoothNodes):
                self.smoothNodes[elem[3][1]].append(elem[3][2])
            else:
                self.smoothNodes[elem[3][1]] = [elem[3][2]]

            self.smoothNodes[elem[3][2]] = [elem[3][0], elem[3][1]]

        self.smoothedNodes = {}
        for inode in self.smoothNodes:
            if(inode <= self.nnodes):
                self.smoothedNodes[inode] = self.nodesO2All[inode]
                continue
            N1 = self.smoothNodes[inode][0]
            N2 = self.smoothNodes[inode][1]
            p1 = self.nodesO2All[N1]
            p2 = self.nodesO2All[N2]
            if(len(self.smoothNodes[N1]) > 1 and len(self.smoothNodes[N2]) > 1):
                if(self.smoothNodes[N1][0] != inode):
                    M1 = self.smoothNodes[N1][0]
                else:
                    M1 = self.smoothNodes[N1][1]
                if(self.smoothNodes[N2][0] != inode):
                    M2 = self.smoothNodes[N2][0]
                else:
                    M2 = self.smoothNodes[N2][1]
                if(self.smoothNodes[M1][0] != N1):
                    N3 = self.smoothNodes[M1][0]
                else:
                    N3 = self.smoothNodes[M1][1]
                if(self.smoothNodes[M2][0] != N2):
                    N4 = self.smoothNodes[M2][0]
                else:
                    N4 = self.smoothNodes[M2][1]
                p3 = self.nodesO2All[N3]
                p4 = self.nodesO2All[N4]
                L12 = p2-p1
                L12 /= np.sqrt(np.dot(L12, L12))
                L31 = p1-p3
                L31 /= np.sqrt(np.dot(L31, L31))
                L24 = p4 - p2
                L24 /= np.sqrt(np.dot(L24, L24))
                A312 = np.arccos(np.dot(L31, L12))
                A124 = np.arccos(np.dot(L12, L24))
                if(A312 > thetaMax or A124 > thetaMax):
                    self.smoothedNodes[inode] = (p1+p2) * 0.5
                    continue
                # print("%f %f"%(A312,A124))
                xs = np.array([p3[0], p1[0], p2[0], p4[0]])
                ys = np.array([p3[1], p1[1], p2[1], p4[1]])
                zs = np.array([p3[2], p1[2], p2[2], p4[2]])
                us = np.array([-1.5, -0.5, 0.5, 1.5])
                tck, u = scipy.interpolate.splprep([xs, ys, zs], u=us)
                pnew = scipy.interpolate.splev([0], tck)
                p = np.array([pnew[0][0], pnew[1][0], pnew[2][0]])
                self.smoothedNodes[inode] = p

            else:
                self.smoothedNodes[inode] = (p1+p2) * 0.5

        self.smoothedNodes = list(self.smoothedNodes.items())
        nodesOld = np.zeros((len(self.smoothedNodes), 3))
        nodesNew = nodesOld.copy()
        inode = 0
        for i in self.smoothedNodes:
            nodesOld[inode, :] = self.nodesO2All[i[0]]
            nodesNew[inode, :] = i[1]
            inode += 1
        nodesDisp = nodesNew-nodesOld
        WXY = getWXY(nodesOld, nodesDisp, crbf)

        for k in self.nodesO2All:
            self.nodesO2All[k] += RBFInterp(nodesOld,
                                            WXY, self.nodesO2All[k], crbf)

        self.smoothedNodes = dict(self.smoothedNodes)
        for inode in self.smoothedNodes:
            self.nodesO2All[inode] = self.smoothedNodes[inode]


def ExecuteO2Interp():
    modder = MeshO12DModder()
    modder.ReadFromGmshO1('NACA0012_WIDE_H3.msh')
    # modder.ReadFromGmshO1('Cylinder.msh')
    modder.GetMediumTopo()
    modder.AddO2Points()
    modder.SmoothEdge('bc-1', 1, np.pi/180.0 * 60)
    modder.PrintO2('NACA0012_WIDE_H3_O2.msh')
    # modder.PrintO2('CylinderO2.msh')


def ExecuteO1Convert():
    modder = MeshO12DModder()
    modder.ReadFromGmshO1('NACA0012_WIDE_H3.msh')
    modder.PrintO1As2dSCFVGrid(
        'NACA0012_WIDE_H3.grid.in', False, ia=3, ib=4, offset=np.array((-60, 0)))


if(__name__ == "__main__"):
    ExecuteO2Interp()
    # ExecuteO1Convert()

    # print(len(modder.faceDict))
    # print(len(modder.nodes))
    # print(modder.nnodes)