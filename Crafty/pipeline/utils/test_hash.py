import unittest
from hash import HashUtil

class TestHashUtil(unittest.TestCase):

    def test_hash_course_info_with_valid_input(self):
        course_info = "Bob want to learn about the history of the United States!"
        result = HashUtil.hash_course_info(course_info)
        self.assertIsInstance(result, str)
        self.assertEqual(10, len(result))
        self.assertEqual('1c36d2796b', result)

    def test_hash_course_info_with_non_ascii_characters(self):
        course_info = "一堂介绍美国历史的课程！"
        result = HashUtil.hash_course_info(course_info)
        self.assertIsInstance(result, str)
        self.assertEqual(10, len(result))
        self.assertEqual('48592033fd', result)

if __name__ == '__main__':
    unittest.main()
