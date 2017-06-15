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
                  

Dataset:
###
![image](https://github.com/noahzn/FoHIS/blob/master/img/dataset.png)

| Source Image  | Maximum Depth | Effect | Homogeneous | Particular Elevation|
|:-------------:|:-------------:|:------:|:-----------:|:-------------------:|
| (a)           |     150 m     | Haze   |      Yes    |         No          |
| (b)           |     400 m     | Haze   |      Yes    |         No          |
| (c)           |     800 m     | Haze   |      Yes    |         No          |
| (d)           |     30 m      | Fog    |      Yes    |         No          |
| (e)           |     150 m     | Fog    |      No     |         Yes         |
| (f)           |     30 m      |Fog+Haze|      No     |         No          |
| (g)           |     600 m     | Haze   |      Yes    |         No          |
| (h)           |     400 m     | Haze   |      Yes    |         No          |
| (i)           |     200 m     | Haze   |      Yes    |         No          |
| (j)           |     100 m     | Haze   |      Yes    |         No          |
| (k)           |     100 m     | Haze   |      Yes    |         No          |
| (l)           |     800 m     |Fog+Haze|      No     |         Yes         |
| (m)           |     300 m     | Haze   |      Yes    |         No          |
| (n)           |     60 m      | Haze   |      Yes    |         No          |
| (o)           |     300 m     | Haze   |      Yes    |         No          |
| (p)           |     1000 m    | Haze   |      Yes    |         No          |
| (q)           |     400 m     | Haze   |      Yes    |         No          |
| (r)           |     300 m     | Haze   |      Yes    |         No          |