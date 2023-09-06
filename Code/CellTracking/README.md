# Intestinal Stem Cells (ISCs) Tracking


Steps to follow one time

1. Install python >= 3.10
2. Create virtual environment at same location of Tracking.py file
    > virtualenv tracking-venv
3. Activate virtual environment
    > tracking-venv\Scripts\activate
4. Install all required libraries
    > pip install requirement.txt
5. Install git and follow command
    > git clone https://github.com/ifzhang/ByteTrack.git
6. Install ByteTrack dependecies
    > pip install -r ./ByteTrack/requirements.txt
7. Install requirements.txt file
    > pip install -r requirements.txt

Steps to follow each execution time

1. Change working directory to CellTracking
    > cd path_to_CellTracking_folder
2. Activate virtual environment
    > tracking-venv\Scripts\activate
3. Command to run algorithm
    > python tracking.py path_of_video_directory
    <br>
    Example:
    python tracking.py 1.avi