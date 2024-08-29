from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()

class AzureBlobHelper(object):
    def __init__(self):
        try:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            print("Successfully connected to Azure Blob Storage.")
        except Exception as e:
            print(f"Failed to connect to Azure Blob Storage: {e}")

    def download(self, blob_name: str, output_name: str, container_name: str):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(output_name, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloaded {blob_name} to {output_name}")
            return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        except Exception as e:
            print(f"Error downloading blob: {e}")
            raise

    def upload(self, blob_name: str, file_path: str, container_name: str):
        try:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            print(f"Uploading {file_path} to {blob_name} in container {container_name}")
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {file_path} to {blob_name} in Azure Blob Storage.")
        except Exception as e:
            print(f"Error uploading blob: {e}")
            raise

    def upload_directory(self, directory_path: str, container_name: str, blob_prefix: str = ""):
        try:
            if not os.path.exists(directory_path):
                print(f"Directory {directory_path} does not exist.")
                return
            
            print(f"Listing files in directory {directory_path}:")
            for root, dirs, files in os.walk(directory_path):
                # Handle empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):  # Directory is empty
                        keep_file_path = os.path.join(dir_path, ".keep")
                        if not os.path.exists(keep_file_path):
                            with open(keep_file_path, "w") as f:
                                pass  # Create an empty .keep file
                        placeholder_blob_name = os.path.join(blob_prefix, os.path.relpath(keep_file_path, directory_path)).replace("\\", "/")
                        print(f"Uploading placeholder for empty directory {dir_name} as {placeholder_blob_name}")
                        self.upload(placeholder_blob_name, keep_file_path, container_name)

                # Handle files
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    blob_name = os.path.join(blob_prefix, os.path.relpath(file_path, directory_path)).replace("\\", "/")
                    print(f"Uploading file {file_name} to blob {blob_name}")
                    self.upload(blob_name, file_path, container_name)

            print(f"Successfully uploaded directory {directory_path} to Azure Blob Storage.")
        except Exception as e:
            print(f"Error uploading directory: {e}")
            raise
    
    def download_directory(self, container_name: str, blob_prefix: str = "", local_base_dir: str = "outputs"):
        try:
            print(f"Listing blobs in container {container_name} with prefix {blob_prefix}:")
            container_client = self.blob_service_client.get_container_client(container=container_name)
            blobs = container_client.list_blobs(name_starts_with=blob_prefix)

            # Ensure the base local directory exists
            if not os.path.exists(local_base_dir):
                os.makedirs(local_base_dir)

            files_found = False
            for blob in blobs:
                files_found = True
                # Construct the local path by prepending the local_base_dir
                download_path = os.path.join(local_base_dir, blob.name)

                # Ensure the directory exists locally
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
                
                print(f"Downloading blob {blob.name} to {download_path}")
                self.download(blob_name=blob.name, output_name=download_path, container_name=container_name)

            if not files_found:
                print(f"No blobs found with prefix {blob_prefix} in container {container_name}.")
            else:
                print(f"Successfully downloaded directory {blob_prefix} to local path {local_base_dir}.")
        except Exception as e:
            print(f"Error downloading directory: {e}")
            raise

if __name__ == "__main__":
    azure_blob_helper = AzureBlobHelper()

    # Define your container name and blob prefix
    container_name = "craftybackendcontainer"
    blob_prefix = "c824aa1f9b"

    # Download the files to the local "outputs" directory
    azure_blob_helper.download_directory(container_name=container_name, blob_prefix=blob_prefix, local_base_dir="outputs")


# if __name__ == "__main__":
#     azure_blob_helper = AzureBlobHelper()

#     # Define your output directory and container name
#     output_dir = "c824aa1f9b"
#     container_name = "craftybackendcontainer"

#     print(f"Listing files in directory {output_dir}:")
#     for root, dirs, files in os.walk(output_dir):
#         for file_name in files:
#             print(f"Found file: {file_name}")

#     # Upload the entire directory to Azure Blob Storage
#     #azure_blob_helper.upload_directory(directory_path=output_dir, container_name=container_name, blob_prefix="outputs/4a23ca54ff")
    