from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import sys

line = np.array([-1, -1, -1, -1])
line_id = -1

# Saves line coordinates on click.
def clickEvent(event):
    if line[0] == -1:
        line[0] = event.x
        line[1] = event.y
    else:
        line[2] = event.x
        line[3] = event.y
        root.destroy()

# Draws a line from first click location to current mouse location.
def updateLine(event):
    global line_id
    if line[0] != -1:
        if line_id != -1:
            canvas.delete(line_id) 
        line_id = canvas.create_line(line[0], line[1], event.x, event.y, width=1, fill='red', smooth=True)
    

# When this method gets called, the specified image "imName" gets displayed in a canvas.
# The user draws in a line via two clicks.
# Start- and end-coordinates of the line get returned as a numpy array [x1, y1, x2, y2].
def intersectionLine_userInput(imName, boundary):
    # Initialize global variables
    global line
    line = np.array([-1, -1, -1, -1])
    global line_id
    line_id = -1
    
    # Create Tkinter instance.
    global root
    root = tk.Tk()
    
    # Load image
    imtk = ImageTk.PhotoImage(Image.open(imName))
    
    # Create canvas
    global canvas
    canvas = tk.Canvas(root, width=imtk.width(), height=imtk.height())
    canvas.pack()
    
    # Place image inside canvas
    canvas.create_image(0, 0, image=imtk, anchor='nw')
    
    # Paint boundary
    for i in range(0, boundary.shape[0]):
        canvas.create_oval(boundary[i, 0]-0.5, boundary[i, 1]-0.5, boundary[i, 0]+0.5, boundary[i, 1]+0.5, outline="red", fill="red", width=2)
    
    # Bind methods necessary for line drawing and start mainloop.
    canvas.bind("<Button-1>", clickEvent)
    canvas.bind("<Motion>", updateLine)
    root.mainloop()
    
    # Return line coordinates
    return line;


# Calculates smallest distance between line segment and point
def distancePointLine(point, line):
    # Source: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    v = np.array([line[0], line[1]])
    w = np.array([line[2], line[3]])
    l2 = np.sum(np.square(w-v))
    if l2 == 0.0: 
        return np.linalg.norm(point-v)
    
    t = max(0, min(1, np.dot(point - v, w - v))/l2)
    projection = v + t * (w - v)
    return np.linalg.norm(projection - point)


# Finds index of boundary point, which is closest to line segment
def nearestIntersectionPoint(boundaries, line):
    # Find boundary points which are near enough to line
    x_min = min(line[0], line[2])
    x_max = max(line[0], line[2])
    y_min = min(line[1], line[3])
    y_max = max(line[1], line[3])
    bitmask = (boundaries[:, 0] >= x_min) 
    bitmask = np.logical_and(bitmask, (boundaries[:, 0] <= x_max))
    bitmask = np.logical_and(bitmask, (boundaries[:, 1] >= y_min))   
    bitmask = np.logical_and(bitmask, (boundaries[:, 1] <= y_max))
    
    indices = np.nonzero(bitmask)
    smallestDistance = sys.float_info.max
    index = -1
    for i in indices[0]:
        distance = distancePointLine(boundaries[i], line)
        if distance < smallestDistance:
            smallestDistance = distance
            index = i
    return index
            
    