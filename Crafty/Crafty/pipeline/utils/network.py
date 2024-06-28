import click
import requests


class NetworkUtil:

    @staticmethod
    def save_image_from_url(url, folder_path, file_name):
        """
        Downloads an image from a given URL and saves it to a specified folder with a specific file name.

        Parameters:
        - url (str): The URL of the image to download.
        - folder_path (str): The local folder path where the image should be saved.
        - file_name (str): The name of the file under which the image will be saved.

        Returns:
        - str: The path to the saved image file.
        """
        try:
            # Get the image content from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful

            # Construct the full path for the image
            full_path = f"{folder_path}{file_name}"

            # Write the image content to a file in the specified folder
            with open(full_path, 'wb') as image_file:
                image_file.write(response.content)
            click.echo(f"Image saved to: {full_path}")

            return full_path
        except requests.RequestException as e:
            return f"An error occurred: {e}"
