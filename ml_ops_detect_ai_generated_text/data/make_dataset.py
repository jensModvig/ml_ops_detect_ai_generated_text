import subprocess

from ml_ops_detect_ai_generated_text.utilities import get_paths



def download_kaggle_data(raw_data_path, username, key):

    # Replace 'your_kaggle_username' and 'your_kaggle_key' with your actual Kaggle username and key.
    kaggle_username = username
    kaggle_key = key

    # Set the desired folder where you want to download the dataset.
    download_folder = raw_data_path

    # Configure Kaggle API credentials
    subprocess.run(['kaggle', 'config', 'set', '-n', 'username', '-v', kaggle_username])
    subprocess.run(['kaggle', 'config', 'set', '-n', 'key', '-v', kaggle_key])

    # Run the Kaggle competitions download command
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'llm-detect-ai-generated-text', '-p', download_folder])





if __name__ == '__main__':
    """

    """

    # Manage paths
    # ============
    repo_path, data_path, model_path = get_paths()

    # Check if the raw data exists
    # ================================
    raw_data_path = data_path / 'raw' / 'llm-detect-ai-generated-text'
    if not raw_data_path.exists():
        # Raw data does not exist, download it
        # ===================================
        print('Downloading raw data...')
        # First request Kaggle username and key
        print('Please enter your Kaggle username and API key.')
        username = input('Username: ')
        key = input('API key: ')
        # Download data
        download_kaggle_data(raw_data_path, username, key)

    

