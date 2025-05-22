import tests.test as t
from src.Hologram import Hologram
from src.DataProcessing import *

from src.Visualizing import plotReconstruction3D, showImage
from pathlib import Path
import numpy as np
import plotly

# t.testHolo()
# t.testHoloLS()
# t.testDiffractionGratings()
# t.testCalibrationHolo()

# h = Hologram()
# h.lightspring(50,1,10,5,6.2)
# h.plotHolo()
# h.plotModes()

testDataRoot = Path(r"C:\Projekte\SLM\testData OGV")
ellipse = (457,404,80,81, 14, 0)
crop = ((664,641),800)
# rec = reconstructData(testDataRoot/"lighspring_OAM50_n=9_sligthBiggerAngle"/"lighspring_tc50_n=9_sligthBiggerAngle_motorPos-114383.npy",
#                        *ellipse,
#                        crop,
#                        1,
#                        0,
#                        subtract_ref=testDataRoot/"lighspring_OAM50_n=9_sligthBiggerAngle"/"correctionData"/"correctionWave_Interf_-112800.npy")

# reconstructFromFolder(testDataRoot/"lighspring_OAM50_n=9_sligthBiggerAngle", 
#                     *ellipse,
#                     crop_pos=crop,
#                     plot_all=0,
#                     plot_exp_hist=1,
#                     subtract_ref=None)#testDataRoot/"lighspring_OAM50_n=9_sligthBiggerAngle"/"correctionData"/"correctionWave_Interf_-112800.npy")

data = makeVolumeArray(testDataRoot/"lighspring_OAM50_n=9_sligthBiggerAngle"/"reconstructionData")
# print(data.shape)
#plotReconstruction3D(np.abs(data[:,:,:]),surfaceCount=4,downscale=4,opacity=0.4)
# fig = plotly.colors.cyclical.swatches()
# fig.show()
# plotly.colors.sequential.swatches().show()
# print(removeDoubleLitted(data,True)[1])
slice = data[24]
# fig = showImage(slice)
print(getCurrentAngleLS(slice, 0.75, (346,306), 50, True)[0]*(180/np.pi))
# print(getCurrentAngleLS(slice, 0.95, (346,306), 50, True)[1]*(180/np.pi))

times = np.linspace(0,data.shape[0]-1,data.shape[0])
res = getOGV(data,times,(346,306), 50, threshhold=0.9, angleFittingAlgo=1)
plt.figure()
plt.plot(times,res[4]*(180/np.pi))
plt.xlabel("time/a.u.")
plt.ylabel("angle/°")
plt.show()