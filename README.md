# Green screen removal and image uploader

A bunch of scripts to automatically remove green screen from photos and upload them in Google Photos.

## Installation

(Optional but recommended) Create a python conda environment and activate it:
```
conda create -n bgremove python
conda activate bgremove
```
Clone the repository and install the requirements:
```
git clone https://github.com/puccj/bgremove-upload
pip install -r requirements.txt
```

## Usage

If you want upload capabilities, you first need to download OAuth client credentials from Google Cloud Console.

- The file `green_remover.py` contains basic functions to remove the background from an image. Use `--test` option to run the script with multiple option parameters to later visually see which parameters' set is better.

- (Optional) Use `create_album.py` to create an album in Google Photos where you can save the images. Note that only albums created this way (i.e. using Google Photos API) can be seen through Google Photos API.

- The script `main.py` continuously monitors a specified folder for new images, remove green backgrounds and uploads the processed images to Google Photos. It also provides a real-time preview of the latest image with the background removed, which can be desabled with `--no_preview`.

Set up your camera to save new photos in a specific folder, run the main script and you are ready to go!

### Typical execution

1. Download OAuth client credentials
2. (Optional) Create an album:

    ```
    python create_album.py <album_title>
    ```
3. Run tests and visually see which set of parameters works best in your light and setting (optional, you can just use the default ones)
    ```
    python green_remover.py --input <input_path> --bg <bg_path> --test
    ```
    Adjust the parameters accordingly, by editing `config.py`.
4. Run `main.py` to watch the folder where new photos are saved:
    ```
    python main.py --folder <folder_path>
    ```