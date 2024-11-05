import contextlib
import json
import logging
import os
import sys
import time
import urllib.request

import joblib
import numpy as np


def get_bounding_box(arr):
    """
    Crops the input 3D array so that no face of the array is entirely NaN.

    Parameters:
    arr (np.ndarray): Input 3D array to crop.

    Returns:
    np.ndarray: Cropped 3D array.
    """
    # Create a mask of where the non-NaN values are located
    mask = ~np.isnan(arr)

    if arr.ndim == 2:
        # Find the min and max indices along each dimension where non-NaN values exist
        y_non_nan = np.where(mask.any(axis=1))[0]
        x_non_nan = np.where(mask.any(axis=0))[0]

        # Determine the bounds for cropping
        ymin, ymax = y_non_nan[0], y_non_nan[-1] + 1
        xmin, xmax = x_non_nan[0], x_non_nan[-1] + 1

        # Crop the array using the determined bounds
        return arr[ymin:ymax, xmin:xmax]

    elif arr.ndim == 3:
        # Find the min and max indices along each dimension where non-NaN values exist
        z_non_nan = np.where(mask.any(axis=(1, 2)))[0]
        y_non_nan = np.where(mask.any(axis=(0, 2)))[0]
        x_non_nan = np.where(mask.any(axis=(0, 1)))[0]

        # Determine the bounds for cropping
        zmin, zmax = z_non_nan[0], z_non_nan[-1] + 1
        ymin, ymax = y_non_nan[0], y_non_nan[-1] + 1
        xmin, xmax = x_non_nan[0], x_non_nan[-1] + 1

        # Crop the array using the determined bounds
        return arr[zmin:zmax, ymin:ymax, xmin:xmax]


def get_logger(logger_date_time):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Ensure the logs directory exists
        logs_path = os.path.join(os.getcwd(), 'logs')
        os.makedirs(logs_path, exist_ok=True)

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(os.path.join(logs_path, f'{logger_date_time}.log'), encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = logging.getLogger()
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()


def close_all_loggers():
    # Close all handlers of the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Iterate over all loggers and close their handlers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def download_file(url, local_filename, verbose=False):
    """
    Downloads a file from the specified URL and saves it to the local file system.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path where the file should be saved.
        verbose (bool): If True, prints download status messages.

    Raises:
        URLError: If there's an issue with the network or URL.
        IOError: If there's an issue writing the file to the local system.
    """
    try:
        with urllib.request.urlopen(url) as response:
            with open(local_filename, 'wb') as out_file:
                out_file.write(response.read())
        if verbose:
            print(f"Downloaded {local_filename}")
    except urllib.error.URLError as e:
        print(f"Failed to download {url}. Error: {e}")
        raise
    except IOError as e:
        print(f"Failed to save {local_filename}. Error: {e}")
        raise


def fetch_github_directory_files(owner, repo, directory_path, save_path=None, token=None, verbose=False):
    """
    Fetches all files from a specified GitHub repository directory and saves them locally.

    Args:
        owner (str): The GitHub username or organization that owns the repository.
        repo (str): The name of the GitHub repository.
        directory_path (str): The path to the directory within the repository.
        save_path (str, optional): The local directory where the downloaded files will be saved.
        token (str, optional): GitHub personal access token for authentication.
        verbose (bool, optional): If True, print progress messages.

    Raises:
        ValueError: If an unsupported imaging format or chapter is specified.
        URLError: If there's an issue with the network or GitHub API.
        IOError: If there's an issue writing files to the local system.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory_path}"
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        with urllib.request.urlopen(urllib.request.Request(api_url, headers=headers)) as response:
            items = json.loads(response.read().decode())

            for item in items:
                item_path = item['path']  # Full path of the item in the repo
                relative_path = os.path.relpath(item_path,
                                                directory_path)  # Relative path within the specified directory

                if item['type'] == 'file':
                    download_url = item['download_url']
                    if save_path:
                        local_path = os.path.join(save_path, relative_path)
                    else:
                        local_path = os.path.join(directory_path, relative_path)

                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    if verbose:
                        print(f"Downloading {relative_path}...")
                    download_file(download_url, local_path, verbose)
                    if verbose:
                        print(f"{relative_path} downloaded.")
                elif item['type'] == 'dir':
                    # Recursive call to handle subdirectories
                    new_save_path = os.path.join(save_path, relative_path) if save_path else None
                    fetch_github_directory_files(owner, repo, item_path, new_save_path, token, verbose)

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            fetch_github_directory_files(owner, repo, directory_path, save_path, token, verbose)
        else:
            print(f"Failed to fetch directory contents from {api_url}. Error: {e}")
            raise
    except urllib.error.URLError as e:
        print(f"Failed to fetch directory contents from {api_url}. Error: {e}")
        raise
    except IOError as e:
        print(f"Failed to create directories or save files to {save_path}. Error: {e}")
        raise


def load_ibsi_phantom(chapter=1, phantom='ct_radiomics', imaging_format="dicom", save_path=None):
    """
    Downloads a specified IBSI Phantom dataset in the chosen imaging format and chapter from the IBSI GitHub repository.

    Args:
        chapter (int): The chapter number of the IBSI dataset. Supported values are 1 and 2.
        phantom (str): The type of phantom dataset to download. Options are "ct_radiomics" and "digital".
        imaging_format (str): The imaging format to download. Options are "dicom" and "nifti".
        save_path (str, optional): The local directory where the dataset will be saved.
                                   If None, the dataset is saved under the original directory structure.

    Raises:
        ValueError: If an unsupported chapter, phantom type, or imaging format is specified.
    """
    owner = "theibsi"
    repo = "data_sets"
    supported_chapters = [1, 2]
    supported_phantoms = ['ct_radiomics', 'digital']
    supported_formats = ["dicom", "nifti"]

    if chapter not in supported_chapters:
        raise ValueError(f"Unsupported chapter '{chapter}'. Supported chapters are: {supported_chapters}")

    if phantom not in supported_phantoms:
        raise ValueError(f"Unsupported phantom '{phantom}'. Supported phantoms are: {supported_phantoms}")

    if imaging_format.lower() not in supported_formats:
        raise ValueError(f"Unsupported imaging format '{imaging_format}'. Supported formats are: {supported_formats}")

    if chapter == 1 and phantom == 'digital' and imaging_format == "dicom":
        raise ValueError(f"The DICOM mask was deprecated due to incorrect image spacing. The phantom is available in NIfTI format and consists of the image itself (image) and its segmentation (mask).")

    directory_path = f"ibsi_{chapter}_{phantom}_phantom/{imaging_format.lower()}"
    fetch_github_directory_files(owner, repo, directory_path, save_path)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.
    source: https://stackoverflow.com/a/58936697/3859823
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
