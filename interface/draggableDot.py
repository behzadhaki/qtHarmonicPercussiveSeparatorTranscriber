import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from PyQt5.QtCore import *
from utils import pitch2chroma, midi2chroma, midi2pitch


class DraggableDot:

    # A class definition for a matplotlib horizontal line that can be interactively modified
    #
    # Manual:
    #       1. Each line is identified by an onset, an offset, and the height (midi_value). To access these values,
    #          use the get_line() method
    #       2. Certain restrictions can be applied to the line:
    #               a. Snapping Vertically: height (midi_value) can only have an integer value
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

    def __init__(self, fig, ax, onset, midi_value, **options):
        #
        #   Inputs:
        #       fig             :   matplotlib figure object
        #       ax              :   matplotlib axes object
        #       onset           :   onset location (x value of starting point of the line)
        #       midi_value         :   midi value of the line (either quantized or not quantized)
        #
        #   Options:
        #       snapVerticallyFlag (bool)  :   Set to True, if height needs to be quantized to an integer value.
        #                                   (default value is False)
        #       horizontal_snap_grid    :   (1-D list or array) location of the grid to snap onsets and/or offsets
        #       snap_offset_flag (bool) :    Set True to enable snapping the offset to the grid (default=False)
        #       x_sensitivity (float)   :   x distance of mouse cursor to a target for an event to be recognized
        #       y_sensitivity (float)   :   y distance of mouse cursor to a target for an event to be recognized
        #       marker_width            :   width of the line to be plotted
        #       defaultColor            :   default color of the line when an event is not to be handled (default=blue)
        #       holdColor               :   color of line while being modified (default=red)
        #       doubleClickColor        :   color of line when double clicked on (default=green)
        #

        self.ax = ax
        self.fig = fig

        self.onset = onset
        self.offset = onset
        self.midi_value = midi_value      # is the midi value

        self.y_isHz = False
        if "y_isHz" in options:
            self.y_isHz = options.get("y_isHz")


        # print(self.chroma_value)

        # fig and ax for plotting lines in a chromagram

        self.snapVerticallyFlag = False
        self.horizontal_snap_grid = []
        self.snap_offset_flag = True

        #   Distance sensitivity for event occurrences
        self.x_sensitivity = .2  # sensitivity of mouse to object closeness
        self.y_sensitivity = .2  # sensitivity of mouse to object closeness

        #   line width and Colors used for different events
        self.marker_width = 8
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
            if option == "marker_width":
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
        self.adjustFlag = False
        self.adjustFlag = False
        self.cidClick = []

        #   set to true if double clicked on a line (right click after double clicking removes a line)
        self.doubleClickFlag = False

        #   Connect event handling functions
        self.connect()

        #   create the line
        self.line = Line2D([self.onset, self.onset], [self.midi_value, self.midi_value],
                           marker="o", markeredgecolor=self.defaultColor,
                           markersize=self.marker_width, color=self.defaultColor)
        self.ax.add_line(self.line)

        #   snap the line to grid if needed
        if self.snapVerticallyFlag:
            self.snap_vertically()
            
        if not self.horizontal_snap_grid == []:
            # print("horizontal_snap_grid", self.horizontal_snap_grid)
            self.snap_onset_to_grid()
        
        self.update_line()

    def update_grid(self, _grid):
        self.horizontal_snap_grid = _grid

    def connect(self):
        # connects the event handling functions
        return
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
        self.midi_value = None

    def on_press(self, event):

        # event handling function for pressing mouse click
        if not (event.xdata and event.ydata):
            return

        # check whether (and which of) onset or offset is selected to be moved
        if (abs(event.xdata-self.onset) <= self.x_sensitivity):
            self.adjustFlag = True
            self.holdFlag = True
            self.line.set_markeredgecolor(self.holdColor)
            if event.dblclick:
                self.doubleClickFlag = True  # set to true if double clicked for the first time
                self.line.set_color(self.doubleClickColor)
                self.line.set_markeredgecolor(self.doubleClickColor)
                self.line.set_markerfacecolor(self.doubleClickColor)
                self.update_line()
                return
        else:
            self.line.set_markeredgecolor(self.defaultColor)
            if not self.holdFlag:
                self.adjustFlag = False

        if self.doubleClickFlag:
            # if right clicked after a double click remove the line
            if event.button == 3:
                self.onset = None
                self.offset = None
                self.midi_value = None
                self.update_line()
                self.disconnect()
            else:
                self.doubleClickFlag = False

        else:
            self.doubleClickFlag = False

    def on_release(self, event):
        # event handling function for releasing mouse click
        if not (event.xdata and event.ydata):
            return

        self.holdFlag = False
        self.adjustFlag = False
        self.adjustFlag = False

        self.line.set_color(self.defaultColor)
        # update and redraw line
        self.update_line()

    def on_motion(self, event):
        # event handling function for moving mous

        # ignore motion outside the ax area
        if not (event.xdata and event.ydata):
            return

        self.doubleClickFlag = False    # disable deleting line if mouse moved after double clicking

        # Adjust line based on the adjustment mode (identifiable through flags)
        if self.holdFlag:

            self.onset = event.xdata

            if self.y_isHz:
                cursor_y = self.pitch2midi(event.ydata)
            else:
                cursor_y = event.ydata

            if self.snapVerticallyFlag:  # move vertically if vertical movement is selected (quantize if required)
                self.midi_value = np.round(cursor_y, 0)
            else:
                self.midi_value = cursor_y

            # Snap onset and offset (optional) to the grid
            if (self.horizontal_snap_grid != []) and self.adjustFlag:
                self.snap_onset_to_grid()  # snap to grid implementation here

            # update and redraw line
            self.update_line()

    def update_line(self):
        # redraws the line
        self.line.set_xdata([self.onset, self.onset])

        self.fig.canvas.draw()

        if self.y_isHz:
            self.line.set_ydata([midi2pitch(self.midi_value), midi2pitch(self.midi_value)])
        else:
            self.line.set_ydata([self.midi_value, self.midi_value])

    def draw(self):
        self.ax.add_line(self.line)

    def get_line(self):
        # returns the parameters characterizing the line
        return self.onset, self.midi_value

    def snap_vertically(self):
        self.midi_value = np.round(self.midi_value, 0)
        return

    def snap_onset_to_grid(self):
        # Snaps onset to grid

        # identify the onset location with respect to grid
        _grid = self.horizontal_snap_grid
        lower_grids = _grid[_grid <= self.onset]
        higher_grids = _grid[_grid >= self.onset]

        # checks where to snap the onset
        if lower_grids != [] and higher_grids != []:
            if (self.onset - lower_grids[-1]) <= (higher_grids[0] - self.onset):
                snapped_onset = lower_grids[-1]
            else:
                snapped_onset = higher_grids[0]

            # update onset value
            self.onset = snapped_onset
            self.offset = snapped_onset
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

    def y2chroma(self, value):
        if self.y_isHz:
            return pitch2chroma(value)
        else:
            return midi2chroma(value)


if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])

    grid = list(range(21))

    line1 = DraggableDot(fig, ax, 1, 10, snapVerticallyFlag=True)
    line2 = DraggableDot(fig, ax, 2.3, 12, horizontal_snap_grid=grid)
    line3 = DraggableDot(fig, ax, 19.1, 12, horizontal_snap_grid=grid, snap_offset_flag=False)
    line4 = DraggableDot(fig, ax, 17.6, 12, horizontal_snap_grid=grid)
    plt.show()
