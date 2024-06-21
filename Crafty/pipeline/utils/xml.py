import re
import xml.etree.ElementTree as ET


class XmlUtil:
    @staticmethod
    def generate_xml_elements(section, elements):
        """
        Generate XML elements for the given list of elements.
        Take this as an example in the prompt.
        """
        section_string = re.sub(r'[^\w\s]', '', str(section))  # Remove any character that is not a word character or whitespace
        section_string = section_string.replace(':', '')  # Explicitly remove colons
        section_string = section_string.replace(' ', '_')  # Replace spaces with underscores
        root = ET.Element(str(section_string))  # Create a root element with the cleaned chapter name
        for element in elements:
            content = f'content for {element} here'
            child = ET.SubElement(root, element)
            child.text = content
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return xml_str

    @staticmethod
    def simple_dict_to_xml(tag, d):
        """
        Turn a simple dict of key/value pairs into XML
        """
        elem = ET.Element(tag)
        for key, val in d.items():
            if isinstance(val, dict):
                child = XmlUtil.simple_dict_to_xml(key, val)
                elem.append(child)
            elif isinstance(val, list):
                for sub_item in val:
                    if isinstance(sub_item, dict):
                        sub_elem = XmlUtil.simple_dict_to_xml(key, sub_item)
                        elem.append(sub_elem)
            else:
                child = ET.Element(key)
                child.text = str(val).strip()
                elem.append(child)

        return elem

    @staticmethod
    def nest_dict_to_xml(data):
        """
        Helper function dict_to_xml
        Convert a list of dictionaries to an XML tree.
        Handles nested dictionaries and lists.
        """
        final_strings = []
        final_roots = []

        root = ET.Element('root')
        for item in data:
            if isinstance(item, dict):
                for key, val in item.items():
                    if isinstance(val, dict):
                        elem = XmlUtil.simple_dict_to_xml(key, val)
                    elif isinstance(val, list):
                        elem = ET.Element(key)
                        for sub_item in val:
                            if isinstance(sub_item, dict):
                                sub_elem = XmlUtil.simple_dict_to_xml('item', sub_item)
                                elem.append(sub_elem)
                    else:
                        elem = ET.Element(key)
                        elem.text = str(val).strip()
                    root.append(elem)
                final_strings.append(ET.tostring(root, encoding='unicode'))
                final_roots.append(root)

        return final_roots

    @staticmethod
    def build_element(element, dictionary):
        """
        Build an XML element from a dictionary where keys are the XML element tags.
        """
        for key, val in dictionary.items():
            if isinstance(val, ET.Element):
                child = val
            else:
                child = ET.Element(key)
                if isinstance(val, dict):
                    XmlUtil.build_element(child, val)
                elif isinstance(val, list):
                    for sub_item in val:
                        if isinstance(sub_item, dict):
                            sub_child = ET.Element(key)
                            XmlUtil.build_element(sub_child, sub_item)
                            child.append(sub_child)
                        else:
                            sub_child = ET.Element('item')
                            sub_child.text = str(sub_item)
                            child.append(sub_child)
                else:
                    child.text = str(val)
            element.append(child)

    @staticmethod
    def dict_to_xml(tag, d):
        """
        Convert a dictionary into an XML tree where keys are the XML element tags.
        """
        elem = ET.Element(tag)
        XmlUtil.build_element(elem, d)
        return elem
