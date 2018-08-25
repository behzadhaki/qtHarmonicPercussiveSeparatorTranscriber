import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from PyQt5.QtCore import *
from utils import pitch2chroma, midi2chroma, midi2pitch


class DraggableHLine:

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

    def __init__(self, fig, ax, onset, offset, midi_value, **options):
        #
        #   Inputs:
        #       fig             :   matplotlib figure object
        #       ax              :   matplotlib axes object
        #       onset           :   onset location (x value of starting point of the line)
        #       offset          :   offset location (x value of ending point of the line)
        #       midi_value         :   midi value of the line (either quantized or not quantized)
        #
        #   Options:
        #       snapVerticallyFlag (bool)  :   Set to True, if height needs to be quantized to an integer value.
        #                                   (default value is False)
        #       horizontal_snap_grid    :   (1-D list or array) location of the grid to snap onsets and/or offsets
        #       snap_offset_flag (bool) :    Set True to enable snapping the offset to the grid (default=False)
        #       x_sensitivity (float)   :   x distance of mouse cursor to a target for an event to be recognized
        #       y_sensitivity (float)   :   y distance of mouse cursor to a target for an event to be recognized
        #       line_width               :   width of the line to be plotted
        #       defaultColor            :   default color of the line when an event is not to be handled (default=blue)
        #       holdColor               :   color of line while being modified (default=red)
        #       doubleClickColor        :   color of line when double clicked on (default=green)
        #       y_isHz                  : True: if the main axis unit is Hz (default= False) (set true when you use a
        #                                       spectrogram with Hz as main ax and Midi as the second ax)
        #       ax_chroma               :   axis for plotting line in a chroma gram
        #       fig_chroma              :   figure for plotting line in a chroma gram
        #

        self.ax = ax
        self.fig = fig

        self.onset = onset
        self.offset = offset
        self.midi_value = midi_value      # is the midi value

        self.y_isHz = False
        if "y_isHz" in options:
            self.y_isHz = options.get("y_isHz")


        # print(self.chroma_value)

        # fig and ax for plotting lines in a chromagram
        self.ax_chroma = None
        self.fig_chroma = None

        if "ax_chroma" in options:
            self.ax_chroma = options.get("ax_chroma")
            self.chroma_value = self.y2chroma(self.midi_value)

        if "fig_chroma" in options:
            self.fig_chroma = options.get("fig_chroma")

        # print(self.ax_chroma)

        self.snapVerticallyFlag = False
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
        # self.connect()    Remove #, if the line needs to be interactive from start

        #   create the line
        self.line = Line2D([self.onset, self.offset], [self.midi_value, self.midi_value],
                           marker="o", markeredgecolor=self.defaultColor,
                           linewidth=self.line_width, color=self.defaultColor)
        self.ax.add_line(self.line)

        if self.ax_chroma:
            self.line_chroma = Line2D([self.onset, self.offset], [self.chroma_value+.5, self.chroma_value],
                                      marker="o", markeredgecolor=self.defaultColor,
                                      linewidth=self.line_width, color=self.defaultColor)
            self.ax_chroma.add_line(self.line_chroma)
        else:
            self.line_chroma = None

        #   snap the line to grid if needed
        if self.snapVerticallyFlag:
            self.snap_vertically()
            
        if not self.horizontal_snap_grid == []:
            # print("horizontal_snap_grid", self.horizontal_snap_grid)
            self.snap_onset_to_grid()
            if self.snap_offset_flag:
                self.snap_offset_to_grid()
        
        self.update_line()

    def update_default_color(self, color="b"):
        self.defaultColor = color
        self.line.set_color(self.defaultColor)
        if self.line_chroma:
            self.line_chroma.set_color(self.defaultColor)
        # update and redraw line
        self.update_line()

    def update_grid(self, _grid):
        self.horizontal_snap_grid = _grid

    def connect(self):
        # connects the event handling functions
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        if self.ax_chroma:
            # print("connected")
            self.cidpress_chroma = self.fig_chroma.canvas.mpl_connect('button_press_event', self.on_press_chroma)
            self.cidrelease_chroma = self.fig_chroma.canvas.mpl_connect('button_release_event', self.on_release)
            self.cidmotion_chroma = self.fig_chroma.canvas.mpl_connect('motion_notify_event', self.on_motion_chroma)

    def disconnect(self):
        # disconnects the event handling functions
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)

        if self.fig_chroma:
            self.fig_chroma.canvas.mpl_disconnect(self.cidpress_chroma)
            self.fig_chroma.canvas.mpl_disconnect(self.cidrelease_chroma)
            self.fig_chroma.canvas.mpl_disconnect(self.cidmotion_chroma)

        # self.onset = None
        # self.offset = None
        # self.midi_value = None
        # self.chroma_value = None

    def on_press(self, event):

        # event handling function for pressing mouse click
        if not (event.xdata and event.ydata):
            return

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

        cursor_y = event.ydata
        if self.y_isHz:
            cursor_y = self.pitch2midi(cursor_y)
        else:
            cursor_y = event.ydata

        if (((event.xdata > self.onset) and (event.xdata < self.offset)) and
             (cursor_y < (self.midi_value + self.y_sensitivity)) and
             (cursor_y > (self.midi_value - self.y_sensitivity))):

            # if right clicked after a double click remove the line
            if event.button == 3 and self.doubleClickFlag:
                self.onset = None
                self.offset = None
                self.midi_value = None
                self.update_line()
                self.disconnect()

            if event.dblclick:
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

            if self.adjustLeftFlag:         # move onset if onset is selected to be moved
                self.onset = min(event.xdata, self.offset - self.x_sensitivity)

            elif self.adjustRightFlag:      # move offset if offset is selected to be moved
                self.offset = max(event.xdata, self.onset + self.x_sensitivity)

            else:
                if self.y_isHz:
                    cursor_y = self.pitch2midi(event.ydata)
                else:
                    cursor_y = event.ydata

                if self.snapVerticallyFlag:  # move vertically if vertical movement is selected (quantize if required)
                    self.midi_value = np.round(cursor_y, 0)
                else:
                    self.midi_value = cursor_y
                self.chroma_value = self.y2chroma(self.midi_value)

            # Snap onset and offset (optional) to the grid
            if (self.horizontal_snap_grid != []) and self.adjustLeftFlag:
                self.snap_onset_to_grid()  # snap to grid implementation here
            if (self.horizontal_snap_grid != []) and self.snap_offset_flag and self.adjustRightFlag:
                self.snap_offset_to_grid()  # snap to grid implementation here

            # update and redraw line
            self.update_line()

    def on_press_chroma(self, event):
        # event handling function for pressing mouse click

        if not (event.xdata and event.ydata):
            return

        self.holdFlag = False   # initialize hold flag to default state

        event.ydata = np.floor(event.ydata - .5)

        if not(event.xdata < self.onset or event.xdata > self.offset or
               event.ydata > (self.chroma_value + self.y_sensitivity) or
               event.ydata < (self.chroma_value - self.y_sensitivity)):

            # if right clicked after a double click remove the line
            if event.button == 3 and self.doubleClickFlag:
                self.onset = None
                self.offset = None
                self.midi_value = None
                self.chroma_value = None
                self.update_line()
                self.disconnect()
                return

            if event.dblclick:
                self.doubleClickFlag = True  # set to true if double clicked for the first time
                self.line_chroma.set_color(self.doubleClickColor)
                self.update_line()
                return
            else:
                self.doubleClickFlag = False
                self.holdFlag = True
        else:
            self.doubleClickFlag = False

    def on_motion_chroma(self, event):
        # event handling function for moving mouse

        # ignore motion outside the ax area
        if not (event.xdata and event.ydata):
            return

        self.doubleClickFlag = False    # disable deleting line if mouse moved after double clicking

        current_chroma_value = self.chroma_value

        # check whether (and which of) onset or offset is selected to be moved
        if (event.xdata <= self.onset + self.x_sensitivity) and event.xdata >= self.onset:
            self.adjustLeftFlag = True
            self.line_chroma.set_markeredgecolor(self.holdColor)
        elif (event.xdata >= self.offset - self.x_sensitivity) and event.xdata <= self.offset:
            self.adjustRightFlag = True
            self.line_chroma.set_markeredgecolor(self.holdColor)
        else:
            self.line_chroma.set_markeredgecolor(self.defaultColor)
            if not self.holdFlag:
                self.adjustLeftFlag = False
                self.adjustRightFlag = False

        # Adjust line based on the adjustment mode (identifiable through flags)
        if self.holdFlag:
            self.line_chroma.set_color(self.holdColor)

            if self.adjustLeftFlag:         # move onset if onset is selected to be moved
                self.onset = min(event.xdata, self.offset - self.x_sensitivity)

            elif self.adjustRightFlag:      # move offset if offset is selected to be moved
                self.offset = max(event.xdata, self.onset + self.x_sensitivity)

            else:
                # move vertically if vertical movement is selected (chroma is always quantized)
                self.chroma_value = np.round(event.ydata, 0)     # new chroma value
                self.midi_value = self.midi_value + (self.chroma_value - current_chroma_value)

            # Snap onset and offset (optional) to the grid
            if (self.horizontal_snap_grid != []) and self.adjustLeftFlag:
                self.snap_onset_to_grid()  # snap to grid implementation here

            # Snap offset (optional) to the grid
            if (self.horizontal_snap_grid != []) and self.snap_offset_flag and self.adjustRightFlag:
                self.snap_offset_to_grid()  # snap to grid implementation here
        else:
            self.line_chroma.set_color(self.defaultColor)

        # update and redraw line
        self.update_line()

    def update_line(self):
        # redraws the line
        self.line.set_xdata([self.onset, self.offset])

        self.fig.canvas.draw()

        if self.y_isHz:
            self.line.set_ydata([midi2pitch(self.midi_value), midi2pitch(self.midi_value)])
        else:
            self.line.set_ydata([self.midi_value, self.midi_value])

        if self.ax_chroma:
            self.line_chroma.set_xdata([self.onset, self.offset])
            if self.chroma_value:
                self.line_chroma.set_ydata([self.chroma_value+.5, self.chroma_value+.5])
            else:
                self.line_chroma.set_ydata([None, None])
            self.fig_chroma.canvas.draw()

    def draw(self):
        self.ax.add_line(self.line)
        self.ax_chroma.add_line(self.line_chroma)

    def get_line(self):
        # returns the parameters characterizing the line
        return self.onset, self.offset, self.midi_value

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
        if lower_grids.tolist() and higher_grids.tolist():
            if (self.onset - lower_grids[-1]) <= (higher_grids[0] - self.onset):
                snapped_onset = lower_grids[-1]
            else:
                if not higher_grids[0] >= self.offset:   # check that the snapped onset doesnt go above or over the offset
                    snapped_onset = higher_grids[0]
                else:
                    snapped_onset = lower_grids[-1]

            # update onset value
            self.onset = snapped_onset

        elif (lower_grids != []):     # means onset is before the first grid line
            self.onset = higher_grids[0]

        return

    def snap_offset_to_grid(self):
        # Snaps offset to grid

        # identify the offset location with respect to grid
        _grid = self.horizontal_snap_grid
        lower_grids = _grid[_grid <= self.offset]
        higher_grids = _grid[_grid >= self.offset]

        # checks where to snap the offset
        # doesn't allow offset to go below or over onset
        if lower_grids!=[] and higher_grids!=[]:
            if (self.offset - lower_grids[-1]) <= (higher_grids[0] - self.onset) and (not lower_grids[-1] <= self.onset):
                snapped_offset = lower_grids[-1]
            else:
                snapped_offset = higher_grids[0]
        else:
            snapped_offset = _grid[-1]
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
    for _grid in grid:
        ax.plot([_grid, _grid],[0,1000])
    line1 = DraggableHLine(fig, ax, .5, 3, 10, horizontal_snap_grid=grid, snapVerticallyFlag=True)
    #line2 = DraggableHLine(fig, ax, 2.3, 6, 12, horizontal_snap_grid=grid)
    #line3 = DraggableHLine(fig, ax, 19.1, 19.3, 12, horizontal_snap_grid=grid, snap_offset_flag=False)
    #line4 = DraggableHLine(fig, ax, 17.6, 17.9, 12, horizontal_snap_grid=grid)
    plt.show()
