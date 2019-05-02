# Shadow Matte Consistency
This program is a Python implementation of the image forgery detection approach proposed in the paper 
**"Identifying Image Composites Through Shadow Matte Consistency"** by **Liu et al.** 
which can be found here: http://ieeexplore.ieee.org/document/5743006/.

## Usage
**Requirements:** An image with a corresponding .csv file, containing xy-coordinates of all shadow boundary points inside the first two columns.

**Python Libraries:**

 - Numpy, SciPy and SymPy
 -	scitkit-learn
 -	matplotlib
 -	pandas

I also included the *ECCV 2010 Shadow Boundary Dataset* (http://vision.gel.ulaval.ca/~jflalonde/data.html) inside the repository. If you want to test the program with this set, just run

    python main.py
 The program will then list all the available image names, and you can pick the ones you want to test the program with.

If you want to run the program with your own images, place the image file and the corresponding boundary .csv file (for example img.jpg and img.csv) inside the main directory and run

    python main.py img.jpg
 If you want to test the approach with two different images, run
 

    python main.py img1.jpg img2.jpg

## Shadow Detector (https://github.com/jflalonde/shadowDetection)
I included the shadow detector used in the paper so you can create your own boundary .csv files.

**Instructions:**

1. Follow the compilation instructions of the original repository. Don't forget to also compile the 3rd party libraries already included inside *shadowDetection/pathUtils/*.
2. Inside the file **setPath.m,** you have to add the following lines:

    addpath(genpath(*'Path to shadowDetector/mycode'*));
    addpath(genpath(*'Path to shadowDetector/3rd_party\')*);
    addpath(genpath(*'Path to shadowDetector/pathUtils'*));

3. Add an image to *shadowDetection/mycode/data/img/*. Inside **demoShadowDetection.m**, replace the content of the variable *imgName* with the name of your image (without file extension).
4. Run **setPath()**, followed by **demoShadowDetection()**.
5. A file *'imgName.csv'* will be placed inside *shadowDetection/mycode*. This contains xy-coordinates of all shadow boundary points.

## Scripts
### main.<span></span>py
Self-explanatory. Here all the different scripts come together.
### shadowIntersection.<span></span>py
This script enables the user to draw in a line from the inside of a shadow to its outside. The nearest intersecting boundary point defines the later on boundary sampling location.
### identifyBoundary.<span></span>py
With this script a set of connected boundary points is computed. This happens based on a maximum distance from one boundary point to the next one.
### sampling.<span></span>py
This is where the boundary sampling happens. A boundary segment gets divided into groups of 10. To each group a line gets fitted via regression. Along every line, a certain number of pixels gets sampled along the lines normal directions. For this, biliniear interpolation gets used.
### sigmoid.<span></span>py
A sigmoid function gets fitted to each boundary patch. With this the so-called penumbra region gets identified.
### spline.<span></span>py
This is the core of the program. Here f_s and f_n get estimated. It also contains the actual code for shadow forgery detection.
### convert_database.<span></span>py
Converts the boundary point files of the used dataset from .xml to .csv file format.