# Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng

License:
###
    This code is made publicly for research use only. 
    It may be modified and redistributed under the terms of the GNU General Public License.
    Please cite the paper and source code if you are using it in your work.
    
Instructions:  
###
    This code has been tested in Windows10-64bit with Python3.4 installed.  
    1. clone this project and put all the files in the same folder
    2. folder structure:
    
          FoHIS/const.py  # define const
                fog.py  # main
                parameter.py # all parameters used in simulating fog/haze are defined here.
                tool_kit.py # some useful functions
                
          AuthESI/compute_aggd.py
                  compute_authenticity.py  # main
                  guided_filter.py  # some functions
                  prisparam_16_hazeandfog.mat  # pre-trained model
                  
          img/img.jpg  # RGB image
              imgd.jpg  # depth image
              result.jpg  # simulation
              
    3. To simulate fog/haze effects:
        run python FoHIS/fog.py, the output 'result.jpg' will be saved in ../img/
          
    4. To evaluate the authenticity:
        run python compute_authenticity.py to evaluate 'result.jpg' in ../img/
                  
