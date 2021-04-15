import json
from os import path
import xml.etree.ElementTree as ElementTree
from xml.parsers.expat import ExpatError
import xmltodict


class NxTomoXML:

    def __init__(self, xml_path):
        self._xml_path = xml_path
        self.nx_tomo_dict = {}
        self.json_template = {}

    def load_template_from_xml(self):
        """
        Load nexus template file from xml file and dump the content into a dictionary.
        Templates are found on: https://github.com/nexusformat/definitions
        """
        try:
            tree = ElementTree.parse(self._xml_path)
            xml_data = tree.getroot()
            xml_str = ElementTree.tostring(xml_data, encoding='utf-8',
                                           method='xml')
            self.nx_tomo_dict = dict(xmltodict.parse(xml_str))
            return True
        except (FileNotFoundError, ValueError, ExpatError) as e:
            print(e)
            return False

    def xml_to_json(self, save_path):
        """
        Save nexus template from xml format to json format.
        """
        status = True
        if not self.nx_tomo_dict:
            status = self.load_template_from_xml()

        if status:
            self._dict_to_json_format()
            with open(save_path, 'w') as json_file:
                json.dump(self.nx_tomo_dict, json_file)
        return status

    def _dict_to_json_format(self):
        pass


if __name__ == '__main__':

    file_dir = path.dirname(path.abspath(__file__))
    nx_tomo_xml_path = path.join(file_dir, "NXtomo.xml")
    tomo_xml = NxTomoXML(nx_tomo_xml_path)
    tomo_xml.xml_to_json(path.join(file_dir, "NXtomo.json"))





