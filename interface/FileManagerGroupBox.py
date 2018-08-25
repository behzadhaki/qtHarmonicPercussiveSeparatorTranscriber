from __future__ import unicode_literals
import sys
import os
import glob

from PyQt5 import QtWidgets

import json

progname = os.path.basename(sys.argv[0])


class FileManagerGroupBox(QtWidgets.QGroupBox):
    #   This widget is a QGroupBox used for navigating through files
    #       and looking at meta data of the current track to be analyzed
    #
    #   Manual:
    #       1.  Select the dataset folder format (* denotes the separate folder for each entry)
    #       2.  Find the tracks in the dataset by pressing "Find Tracks" push button
    #       3.  Use "Next" or "Previous" push buttons to navigate through files
    #
    #   Notes:
    #       1.  To be able to show Discogs and Youtube links, a json file should be located in the same folder as the
    #           audio file containing the info. json format: {"uri":<discogs link> , "youtube":<youtube link>
    #       2.  To go to a specific file, change the id number in the entry box and press "Enter" or "Return"
    #       3.  use self.set_current_id(new_id) to change the current id manually from your code
    #       4.  use self.set_current_id(new_id) to change the current id manually from your code
    #       4.  use self.get_id_and_filename() to get the existing id and filename (the filename includes the full
    #                                                                               address of the file)
    #
    #

    def __init__(self, parent=None, group_title="No Title", **options):

        #   Create QGroupBox and set the parent canvas (if any)
        QtWidgets.QGroupBox.__init__(self, group_title)
        self.setParent(parent)

        #   Create the grid within the group box
        self.layout = QtWidgets.QGridLayout()

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)
        self.layout.setColumnStretch(4, 4)
        self.layout.setColumnStretch(5, 4)

        #   QtWidget objects
        self.load_directory_line_edit = QtWidgets.QLineEdit(self)

        self.find_tracks_push_button = QtWidgets.QPushButton(' Find Tracks ')
        self.find_tracks_push_button.clicked.connect(self.find_tracks)

        self.next_push_button = QtWidgets.QPushButton('   Next   ')
        self.next_push_button.clicked.connect(self.next_track)
        self.next_push_button.setDisabled(True)                 # Next  should be disabled before finding tracks

        self.previous_push_button = QtWidgets.QPushButton('Previous')
        self.previous_push_button.clicked.connect(self.previous_track)
        self.previous_push_button.setDisabled(True)             # Previous  should be disabled before finding tracks

        self.id_label = QtWidgets.QLineEdit(self)   # change current id manually and press enter to go to another file
        self.id_label.returnPressed.connect(self.change_current_id_manually)
        self.filename_label = QtWidgets.QLineEdit(self)
        self.youtube_link_line_edit = QtWidgets.QLineEdit(self)
        self.discogs_link_line_edit = QtWidgets.QLineEdit(self)

        # Static Widgets and texts
        self.load_label = QtWidgets.QLabel(self)
        self.save_label = QtWidgets.QLabel(self)
        self.load_label.setText("Dataset Format: ")

        # Dataset related variables
        #   All entries must be located in a separate folder
        #   preferably, 1 entry in each folder
        self.load_directory_format = ""  # dataset location
        if "load_directory_format" in options:
            self.load_directory_format = options.get("load_directory_format")
        self.ids = []
        self.filenames = []
        self.youtube_links = []
        self.discogs_links = []

        # Track related variables
        self.current_id = "--"
        self.current_filename = "--"
        self.current_youtube_link = "youtube.com/??"
        self.current_discogs_link = "discogs.com/??"

        #   Set the texts in the widgets
        self.update_dataset_related_texts()
        self.update_current_file_texts()

        #   Organize the widgets in the grid
        self.lay_widgets_in_layout()

    def get_current_filename(self):
        return self.current_filename

    def update_dataset_related_texts(self):
        self.load_directory_line_edit.setText(self.load_directory_format)

    def update_current_file_texts(self):
        self.id_label.setText("id # "+str(self.current_id)+" out of "+str(len(self.filenames)))
        self.filename_label.setText(self.current_filename)
        self.youtube_link_line_edit.setText(self.current_youtube_link)
        self.discogs_link_line_edit.setText(self.current_discogs_link)

    def lay_widgets_in_layout(self):
        self.show()

        # Line 1 widgets
        self.layout.addWidget(self.load_label, 0, 0)
        self.layout.addWidget(self.load_directory_line_edit, 0, 1, 1, 2)
        self.layout.addWidget(self.find_tracks_push_button, 0, 3)

        # Line 2 widgets
        self.layout.addWidget(self.previous_push_button, 1, 0)
        self.layout.addWidget(self.next_push_button, 1, 1)
        self.layout.addWidget(self.id_label, 1, 2, 1, 2)
        self.layout.addWidget(self.filename_label, 1, 4)
        self.layout.addWidget(self.youtube_link_line_edit, 2, 5)
        self.layout.addWidget(self.discogs_link_line_edit, 1, 5)

        # Layout
        self.setLayout(self.layout)

    def find_tracks(self):

        # find all audio files
        if not(self.load_directory_line_edit.text()):
            self.load_directory_format = []
            return

        if self.load_directory_line_edit.text()[-1] != "/":
            self.load_directory_format = self.load_directory_line_edit.text()+"/"
        else:
            self.load_directory_format = self.load_directory_line_edit.text()

        self.filenames = glob.glob(self.load_directory_format+"*.mp3")
        wavfiles = glob.glob(self.load_directory_format + "*.wav")

        if wavfiles:
            self.filenames.extend(wavfiles)

        # sort filenames in alphabetical order
        self.filenames = sorted(self.filenames)

        # initialize current file
        if not self.filenames:      # if empty: disable buttons
            self.current_id = "??"
            self.current_filename = "No Files Found"
            self.current_youtube_link = "youtube.com/??"
            self.current_discogs_link = "discogs.com/??"
        else:
            self.current_id = 1
            self.current_filename = self.filenames[0]
            self.get_youtube_discogs_link()

        self.update_current_file_texts()

        self.set_button_states()

        self.create_analysis_folders()

    def get_youtube_discogs_link(self):
        # gets the youtube and discogs link from the json metadata file located in the same folder as audio
        jsonFolder = "/".join(self.current_filename.split("/")[:-1])  # assumes json file is in the same folder as audio
        jsonFile = glob.glob(jsonFolder+"/*.json")

        self.current_youtube_link = "youtube.com/??"
        self.current_discogs_link = "discogs.com/??"

        if jsonFile:
            json_data = json.load(open(jsonFile[0]))

            if "uri" in json_data:
                self.current_discogs_link = json_data["uri"]
            if "youtube" in json_data:
                self.current_discogs_link = json_data["youtube"]

    def create_analysis_folders(self):
        current_folder = os.path.dirname(self.current_filename)
        if not os.path.isdir(os.path.join(current_folder, "percussive")):
            os.mkdir(os.path.join(current_folder, "percussive"))
        if not os.path.isdir(os.path.join(current_folder, "harmonic")):
            os.mkdir(os.path.join(current_folder, "harmonic"))

    def next_track(self):
        # call back for "next" button
        current_track_index = self.filenames.index(self.current_filename)
        next_track_index = (current_track_index + 1)
        self.current_id = next_track_index+1
        self.set_track_data()
        self.create_analysis_folders()

    def previous_track(self):
        # call back for "previous" button
        current_track_index = self.filenames.index(self.current_filename)
        previous_track_index = (current_track_index - 1)
        self.current_id = previous_track_index+1
        self.set_track_data()
        self.create_analysis_folders()

    def set_track_data(self):
        #   After updating the self.current_id data, call this to load other relevant info
        self.current_filename = self.filenames[self.current_id-1]
        self.get_youtube_discogs_link()
        self.update_current_file_texts()
        self.set_button_states()

    def set_button_states(self):
        self.next_push_button.setDisabled(False)
        self.previous_push_button.setDisabled(False)

        if self.current_id == (len(self.filenames)):    # Disable Next if current track is the last track
            self.next_push_button.setDisabled(True)

        if self.current_id == 1:                        # Disable Previous if current track is the last track
            self.previous_push_button.setDisabled(True)

    def change_current_id_manually(self):
        # callback for changing current id manually and pressing enter  (connected to self.id_label)
        try:
            id_requested = int(self.id_label.text().split(" ")[2])
        except:
            self.set_track_data()
            return

        if 0 < id_requested <= len(self.filenames):
            self.set_current_id(int(id_requested))
        else:
            self.set_track_data()

    def set_current_id(self, current_id):
        # update id and set track data
        self.current_id = current_id
        self.set_track_data()
        return

    def get_id_and_filename(self):
        # use this to get the current id and filename (which includes the complete address to locate audio)
        return self.current_id, self.current_filename


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    aw = FileManagerGroupBox()
    aw.setWindowTitle("%s" % progname)
    aw.show()
    sys.exit(app.exec_())