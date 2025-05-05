from Hologram import Hologram
from matplotlib import pyplot as plt

h = Hologram()
radia, OAM = h._linearInterpolRadius(5,20,1)

# print(radia,OAM)
# plt.figure()
# plt.plot(radia[1:], OAM, marker = "x")
# plt.show()

z = h.setZernikeCorrection()
h.parabolaOAM(11,20,75)
h.stepFunctionOAM(10,5,70,40,-1,-1)
h.plotHolo()
h.plotModes()
