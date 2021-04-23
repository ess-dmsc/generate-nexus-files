import json
from enum import Enum
from os import path
from lxml import etree
from xml.parsers.expat import ExpatError
import xmltodict


class FileWriterNexusConfigCreator:
    class ClassTypes(Enum):
        LIST, DICT, RAW = range(3)

    CHILDREN = 'children'
    GROUP = 'group'
    LINK = 'link'
    NAME = 'name'
    SOURCE = 'source'
    STREAM = 'stream'
    TARGET = 'target'
    TOPIC = 'topic'
    TYPE = 'type'
    WRITER_MODULE = 'writer_module'

    def __init__(self, nxs_definition_xml):
        self._nxs_definition_xml = nxs_definition_xml
        self.translator = self.get_translation()

    def get_translation(self):
        """
        Translation of nexus definition XML based dictionary to
        corresponding keys and types used in the file writer nexus configuration
        json.

        ::return:: returns the translation as a dictionary.
        """
        return {'@type': (self.TYPE, self.ClassTypes.RAW),
                '@name': (self.NAME, self.ClassTypes.RAW),
                '@target': (self.TARGET, self.ClassTypes.RAW),
                'group': (self.CHILDREN, self.ClassTypes.LIST),
                'link': (self.LINK, self.ClassTypes.RAW)}

    def nxs_config_object_factory(self, class_type, args):
        """
        Encapsulates argument in args using an object of type class_type.

        ::returns:: encapsulated argument or argument.
        """
        if class_type == self.ClassTypes.RAW:
            return args
        elif class_type == self.ClassTypes.LIST:
            return list(args)
        if class_type == self.ClassTypes.DICT:
            return dict(args)
        else:
            raise ValueError("Class type not supported.")

    def generate_nexus_file_writer_config(self):
        """
            Translate dictionary generated from xml file to a format that is
            consistent with nexus config json file used by the file writer.
        """
        data = self.edit_dict_key_value_pair(self._nxs_definition_xml)
        return {self.CHILDREN: [data]}

    def edit_dict_key_value_pair(self, sub_dict, parent=None):
        """
        Edits XML key-value pair to comply with expected format for
        file writer nexus configuration file.

        ::return:: returns modified sub dictionary.
        """
        data = {}
        for key in sub_dict:
            if key in self.translator:
                new_key = self.translator[key][0]
                class_type = self.translator[key][1]
                data[new_key] = self.nxs_config_object_factory(class_type,
                                                               sub_dict[key])
                if isinstance(sub_dict[key], list):
                    tmp_list = []
                    for item in sub_dict[key]:
                        tmp_list.append(self.edit_dict_key_value_pair(item,
                                                                      new_key))
                    data[new_key] = tmp_list
        if self.CHILDREN not in data and parent is not self.LINK:
            data[self.CHILDREN] = self.get_stream_information()

        return data

    def get_stream_information(self):
        """
        Get the stream information for the file writer to add in the
        file writer config json file.
        """
        stream_info = [{
            self.TYPE: self.STREAM,
            self.STREAM: {
                    self.TOPIC: '',
                    self.SOURCE: '',
                    self.WRITER_MODULE: '',
            },
        }]
        return stream_info


class NxApplicationXMLToJson:
    nodes_to_remove = ["field"]
    NAMESPACE = "{http://definition.nexusformat.org/nxdl/3.1}"

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
            tree = etree.parse(self._xml_path)
            xml_data = self._remove_nodes_from_tree(tree)
            xml_str = etree.tostring(xml_data, encoding='utf-8',
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
            dict_cont = self.nx_tomo_dict['definition']['group']
            nxs_config_creator = FileWriterNexusConfigCreator(dict_cont)
            data = nxs_config_creator.generate_nexus_file_writer_config()
            with open(save_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        return status

    def _remove_nodes_from_tree(self, tree):
        """
        Removes occurrences of specific nodes from tree.
        The undesired nodes are specified in nodes_to_remove.

        ::return:: Returns tree where undesired nodes are removed.
        """
        properties = [f'{self.NAMESPACE}{list_item}'
                      for list_item in self.nodes_to_remove]
        tree_root = tree.getroot()
        nodes_to_remove = [tree_root.findall(f'.//{prop}')
                           for prop in properties]
        for node in nodes_to_remove[0]:
            node.getparent().remove(node)

        return tree_root


if __name__ == '__main__':
    file_dir = path.dirname(path.abspath(__file__))
    nx_tomo_xml_path = path.join(file_dir, "NXtomo.xml")
    tomo_xml = NxApplicationXMLToJson(nx_tomo_xml_path)
    tomo_xml.xml_to_json(path.join(file_dir, "NXtomo.json"))
