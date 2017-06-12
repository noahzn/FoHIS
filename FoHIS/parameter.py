"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""


import const

# atmosphere
const.VISIBILITY_RANGE_MOLECULE = 12  # m    12
const.VISIBILITY_RANGE_AEROSOL = 450  # m     450
const.ECM = 3.912 / const.VISIBILITY_RANGE_MOLECULE  # EXTINCTION_COEFFICIENT_MOLECULE /m
const.ECA = 3.912 / const.VISIBILITY_RANGE_AEROSOL  # EXTINCTION_COEFFICIENT_AEROSOL /m

const.FT = 70  # FOG_TOP m  31  70
const.HT = 34  # HAZE_TOP m  300    34

# camera
const.CAMERA_ALTITUDE = 1.8  #  m fog 50   1.8
const.HORIZONTAL_ANGLE = 0  #  °
const.CAMERA_VERTICAL_FOV = 64  #  °


