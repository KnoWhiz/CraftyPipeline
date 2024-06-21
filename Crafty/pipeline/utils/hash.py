import hashlib


class HashUtil:
    @staticmethod
    def course_id(course_info: str) -> str:
        """
        Hash the course description with SHA-1 and truncate the hash to 10 characters.
        """
        # Initialize a hashlib object for SHA-1
        sha1_hash = hashlib.sha1()
        sha1_hash.update(course_info.encode("utf-8"))

        # Calculate the final hash and truncate it to 10 characters
        course_id = sha1_hash.hexdigest()[:10]

        return course_id
