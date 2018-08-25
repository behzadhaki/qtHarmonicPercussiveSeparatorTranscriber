import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from PyQt5.QtCore import *


class DraggableHLine:

    # A class definition for a matplotlib horizontal line that can be interactively modified
    #
    # Manual:
    #       1. Each line is identified by an onset, an offset, and the height (y_value). To access these values,
    #          use the get_line() method
    #       2. Certain restrictions can be applied to the line:
    #               a. Snapping Vertically: height (y_value) can only have an integer value
    #               b. Snapping onset:  the onset can be snapped to a provided grid (horizontal_snap_grid)
    #                                   of vertical lines.
    #                                   (If a horizontal_snap_grid is provided, the onset is automatically snapped)
    #               c. Snapping offset: The onset can be snapped to a provided grid of vertical lines.
    #                                   (Offset snapping should explicitly be enabled using snap_offset_flag)
    #       2. A DraggableHLine object can be moved up and down by dragging and moving the object on the plot
    #       3. The onset of the object can be moved left or right by dragging the onset
    #       4. The offset of the object can be moved left or right by dragging the onset
    #       5. To delete a line, double click on the line. Without moving the mouse cursor,
    #          right-click to remove the line. After deleting onset, offset, y_val will be "None".
    #

    def __init__(self, fig, ax, onset, offset, y_value, **options):
        #
        #   Inputs:
        #       fig             :   matplotlib figure object
        #       ax              :   matplotlib axes object
        #       onset           :   onset location (x value of starting point of the line)
        #       offset          :   offset location (x value of ending point of the line)
        #       y_value         :   height of the line
        #
        #   Options:
        #       snapVerticallyFlag (bool)  :   Set to True, if height needs to be quantized to an integer value.
        #                                   (default value is False)
        #       horizontal_snap_grid    :   (1-D list or array) location of the grid to snap onsets and/or offsets
        #       snap_offset_flag (bool) :    Set True to enable snapping the offset to the grid
        #       x_sensitivity (float)   :   x distance of mouse cursor to a target for an event to be recognized
        #       y_sensitivity (float)   :   y distance of mouse cursor to a target for an event to be recognized
        #       line_width               :   width of the line to be plotted
        #       defaultColor            :   default color of the line when an event is not to be handled (default=blue)
        #       holdColor               :   color of line while being modified (default=red)
        #       doubleClickColor        :   color of line when double clicked on (default=green)
        #       y_isHz                  : True: if the main axis unit is Hz (default= False) (set true when you use a
        #                                       spectrogram with Hz as main ax and Midi as the second ax)

        self.onset = onset
        self.offset = offset
        self.y_value = y_value
        self.y_isHz = False

        self.ax = ax
        self.fig = fig

        self.snapVerticallyFlag = True
        self.horizontal_snap_grid = []
        self.snap_offset_flag = True

        #   Distance sensitivity for event occurrences
        self.x_sensitivity = .2  # sensitivity of mouse to object closeness
        self.y_sensitivity = .2  # sensitivity of mouse to object closeness

        #   line width and Colors used for different events
        self.line_width = 3
        self.defaultColor = "b"
        self.holdColor = "r"
        self.doubleClickColor = "g"

        #   Check optional parameters to be updated
        for option in options:
            if option == "snapVerticallyFlag":
                self.snapVerticallyFlag = options.get(option)
            if option == "horizontal_snap_grid":
                self.horizontal_snap_grid = np.array(options.get(option))
            if option == "snap_offset_flag":
                self.snap_offset_flag = options.get(option)
            if option == "x_sensitivity":
                self.x_sensitivity = options.get(option)
            if option == "y_sensitivity":
                self.y_sensitivity = options.get(option)
            if option == "line_width":
                self.line_width = options.get(option)
            if option == "defaultColor":
                self.defaultColor = options.get(option)
            if option == "holdColor":
                self.holdColor = options.get(option)
            if option == "doubleClickColor":
                self.doubleClickColor = options.get(option)
            if option == "y_isHz":
                self.y_isHz = options.get("y_isHz")

        # make sure a grid is provided for offset snapping, otherwise disable offset snapping
        if self.horizontal_snap_grid == []:
            self.snap_offset_flag = False

        #   Initialize Flags used for event handling
        self.holdFlag = False
        self.adjustLeftFlag = False
        self.adjustRightFlag = False
        self.cidClick = []

        #   set to true if double clicked on a line (right click after double clicking removes a line)
        self.doubleClickFlag = False



        #   Connect event handling functions
        self.connect()

        #   create the line
        self.line = Line2D([self.onset, self.offset], [self.y_value, self.y_value],
                           marker="o", markeredgecolor=self.defaultColor,
                           linewidth=self.line_width, color=self.defaultColor)
        self.ax.add_line(self.line)

        #   snap the line to grid if needed
        if self.snapVerticallyFlag:
            self.snap_vertically()
            
        if not self.horizontal_snap_grid == []:
            self.snap_onset_to_grid()
            if self.snap_offset_flag:
                self.snap_offset_to_grid()
        
        self.update_line()

    def connect(self):
        # connects the event handling functions
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        # disconnects the event handling functions
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)

        self.onset = None
        self.offset = None
        self.y_value = None

    def on_press(self, event):
        # event handling function for pressing mouse click
        if not (event.xdata and event.ydata):
            return
        #print(self.y_value)
        self.holdFlag = False   # initialize hold flag to default state

        if self.y_isHz:
            event.ydata = self.pitch2midi(event.ydata)

        if not(event.xdata < self.onset or event.xdata > self.offset or \
                event.ydata > (self.y_value + self.y_sensitivity) or event.ydata < (self.y_value - self.y_sensitivity)):
            # if right clicked after a double click remove the line
            if event.button == 3 and self.doubleClickFlag:
                self.onset = None
                self.offset = None
                self.y_value = None
                self.update_line()
                self.disconnect()

            if event.dblclick:
                #print("Can remove")
                self.doubleClickFlag = True  # set to true if double clicked for the first time
                self.line.set_color(self.doubleClickColor)
                self.update_line()
                return
            else:
                self.doubleClickFlag = False
                self.holdFlag = True
        else:
            self.doubleClickFlag = False

    def on_release(self, event):
        # event handling function for releasing mouse click
        if not (event.xdata and event.ydata):
            return

        self.holdFlag = False
        self.adjustLeftFlag = False
        self.adjustRightFlag = False

    def on_motion(self, event):
        # event handling function for moving mouse

        # ignore motion outside the ax area
        if not (event.xdata and event.ydata):
            return

        self.doubleClickFlag = False    # disable deleting line if mouse moved after double clicking

        #print(event.ydata)
        if self.y_isHz:
            event.ydata = self.pitch2midi(event.ydata)
        #print(event.ydata)
        # check whether (and which of) onset or offset is selected to be moved
        if (event.xdata <= self.onset + self.x_sensitivity) and event.xdata >= self.onset:
            self.adjustLeftFlag = True
            self.line.set_markeredgecolor(self.holdColor)
        elif (event.xdata >= self.offset - self.x_sensitivity) and event.xdata <= self.offset:
            self.adjustRightFlag = True
            self.line.set_markeredgecolor(self.holdColor)
        else:
            self.line.set_markeredgecolor(self.defaultColor)
            if not self.holdFlag:
                self.adjustLeftFlag = False
                self.adjustRightFlag = False

        # Adjust line based on the adjustment mode (identifiable through flags)
        if self.holdFlag:
            self.line.set_color(self.holdColor)

            if self.adjustLeftFlag:         # move onset if onset is selected to be moved
                self.onset = min(event.xdata, self.offset - self.x_sensitivity)

            elif self.adjustRightFlag:      # move offset if offset is selected to be moved
                self.offset = max(event.xdata, self.onset + self.x_sensitivity)

            else:
                if self.snapVerticallyFlag:  # move vertically if vertical movement is selected (quantize if required)
                    self.y_value = np.round(event.ydata, 0)
                else:
                    self.y_value = event.ydata

            # Snap onset and offset (optional) to the grid
            if (self.horizontal_snap_grid != []) and self.adjustLeftFlag:
                self.snap_onset_to_grid()  # snap to grid implementation here
            if (self.horizontal_snap_grid != []) and self.snap_offset_flag and self.adjustRightFlag:
                self.snap_offset_to_grid()  # snap to grid implementation here
        else:
            self.line.set_color(self.defaultColor)
        # update and redraw line
        self.update_line()

    def update_line(self):
        # redraws the line
        self.line.set_xdata([self.onset, self.offset])
        self.line.set_ydata([self.y_value, self.y_value])
        self.fig.canvas.draw()

    def draw(self):
        self.ax.add_line(self.line)

    def get_line(self):
        # returns the parameters characterizing the line
        return self.onset, self.offset, self.y_value

    def snap_vertically(self):
        self.y_value=np.round(self.y_value,0)
        return

    def snap_onset_to_grid(self):
        # Snaps onset to grid

        # identify the onset location with respect to grid
        _grid = self.horizontal_snap_grid
        lower_grids = _grid[_grid <= self.onset]
        higher_grids = _grid[_grid >= self.onset]

        # checks where to snap the onset
        if (self.onset - lower_grids[-1]) <= (higher_grids[0] - self.onset):
            snapped_onset = lower_grids[-1]
        else:
            if not higher_grids[0] >= self.offset:   # check that the snapped onset doesnt go above or over the offset
                snapped_onset = higher_grids[0]
            else:
                snapped_onset = lower_grids[-1]

        # update onset value
        self.onset = snapped_onset
        return

    def snap_offset_to_grid(self):
        # Snaps offset to grid

        # identify the offset location with respect to grid
        _grid = self.horizontal_snap_grid
        lower_grids = _grid[_grid <= self.offset]
        higher_grids = _grid[_grid >= self.offset]

        # checks where to snap the offset
        # doesn't allow offset to go below or over onset
        if (self.offset - lower_grids[-1]) <= (higher_grids[0] - self.onset) and (not lower_grids[-1] <= self.onset):
            snapped_offset = lower_grids[-1]
        else:
            snapped_offset = higher_grids[0]

        # update offset value
        self.offset = snapped_offset
        return

    def pitch2midi(self, pitch, quantizePitch=False):
        # converts Hz to Midi number
        midi = []

        if pitch == 0:
            midi = 0
        else:
            if quantizePitch:
                midi = (np.int(np.round(69 + 12 * np.math.log(pitch / 440.0, 2), 0)))
            else:
                midi = (69 + 12 * np.math.log(pitch / 440.0, 2))
        return midi

    def midi2pitch(self, midi):
        return 2**((midi-69)/12.0)*440



if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])

    grid = list(range(21))

    line1 = DraggableHLine(fig, ax, 1, 3, 10, snapVerticallyFlag=True)
    line2 = DraggableHLine(fig, ax, 2.3, 6, 12, horizontal_snap_grid=grid)
    line3 = DraggableHLine(fig, ax, 19.1, 19.3, 12, horizontal_snap_grid=grid, snap_offset_flag=False)
    line4 = DraggableHLine(fig, ax, 17.6, 17.9, 12, horizontal_snap_grid=grid)
    plt.show()
