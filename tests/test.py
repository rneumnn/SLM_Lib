from src.Hologram import Hologram
from matplotlib import pyplot as plt


h = Hologram()
radia, OAM = h._linearInterpolRadius(5,20,1)

# print(radia,OAM)
# plt.figure()
# plt.plot(radia[1:], OAM, marker = "x")
# plt.show()
def testHolo():
    z = h.setZernikeCorrection()
    #h.parabolaOAM(11,20,75)
    h.stepFunctionOAM(10,5,70,40,-1,-1)
    h.plotHolo()
    h.plotModes()

def testHoloLS():
    z = h.setZernikeCorrection()
    #h.parabolaOAM(11,20,75)
    h.lightspring(50,1,10,5,6.5)
    h.plotHolo()
    h.plotModes()

def testDiffractionGratings():
    h.lightspring(50,1,5,3,6)
    h.applyGratingToSections(h.boundaries,h.holo,1,plot = True)
    h.plotHolo()
    
def testCalibrationHolo():
    h.calibrationHolo(4, 6)
    h.plotHolo()
