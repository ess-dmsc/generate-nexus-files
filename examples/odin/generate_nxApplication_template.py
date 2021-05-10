import json

import pandas as pd
import xmltodict
from enum import Enum
from os import path
from lxml import etree
from xml.parsers.expat import ExpatError

ATTRIBUTES = 'attributes'
CHILDREN = 'children'
CONFIG = 'config'
DATA_TYPE = 'dtype'
GROUP = 'group'
LINK = 'link'
NAME = 'name'
NX_CLASS = 'NX_class'
SOURCE = 'source'
STREAM = 'stream'
TARGET = 'target'
TOPIC = 'topic'
TYPE = 'type'
VALUES = 'values'
VALUE_UNITS = 'value_units'
WRITER_MODULE = 'module'


class DeviceConfigurationFromXLS:

    list_excel_cols = [NAME, TYPE, TOPIC, SOURCE, WRITER_MODULE, DATA_TYPE, VALUE_UNITS]

    def __init__(self, file_path):
        self._load_configuration_file(file_path)

    def _load_configuration_file(self, file_path):
        """
        Loads excel configuration file to get device data stream information.
        """
        try:
            self.configuration = pd.read_excel(file_path)
        except FileNotFoundError as err:
            self.configuration = None
            print(err)

    def get_configuration_as_dict(self):
        """
        Translates the configuration raw file content to a organized dictionary.

        ::return:: returns a dictionary containing information about data stream
        that the kafka-to-nexus file writer will use.
        """
        data_in_dict = {}
        if self.configuration is not None:
            used_keys = []
            for key in self.configuration:
                if key in self.list_excel_cols:
                    used_keys.append(key)
            if set(used_keys) == set(self.list_excel_cols):
                data_in_dict = self._construct_config_dict()
            else:
                raise KeyError('Missing columns in excel configuration file.')

        return data_in_dict

    def _construct_config_dict(self):
        """
        Constructs the dictionary from the raw file content/

        ::return:: returns the processed configuration data in a structured
        dictionary.
        """
        data = self.configuration
        config_dict = {}
        for idx, name in enumerate(data[NAME]):
            tmp_dict = {}
            for key in self.list_excel_cols:
                tmp_dict[key] = data[key][idx]
            config_dict.update({name: tmp_dict})
        return config_dict

    def _replace_nans(self):
        for key in self.configuration:
            for idx, item in enumerate(self.configuration[key]):
                item = str(item)
                if 'nan' in item or 'NaN' in item:
                    self.configuration[key][idx] = None


class FileWriterNexusConfigCreator:
    class ClassTypes(Enum):
        LIST, DICT, RAW = range(3)

    XML_NAME = '@name'
    XML_TYPE = '@type'
    XML_TARGET = '@target'
    XML_GROUP = 'group'
    XML_LINK = 'link'

    def __init__(self, nxs_definition_xml, xls_path):
        self._nxs_definition_xml = nxs_definition_xml
        self.translator = self.get_translation()
        self.configuration = DeviceConfigurationFromXLS(xls_path).get_configuration_as_dict()

    def get_translation(self):
        """
        Translation of nexus definition XML based dictionary to
        corresponding keys and types used in the file writer nexus configuration
        json.

        ::return:: returns the translation as a dictionary.
        """
        return {self.XML_TYPE: (TYPE, self.ClassTypes.RAW),
                self.XML_NAME: (NAME, self.ClassTypes.RAW),
                self.XML_TARGET: (TARGET, self.ClassTypes.RAW),
                self.XML_GROUP: (CHILDREN, self.ClassTypes.LIST),
                self.XML_LINK: (LINK, self.ClassTypes.RAW)}

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
        return {CHILDREN: [data]}

    def edit_dict_key_value_pair(self, sub_dict, parent=None):
        """
        Edits XML key-value pair to comply with expected format for
        file writer nexus configuration file.

        ::return:: returns modified sub dictionary.
        """
        data = {}
        name = NAME
        for key in sub_dict:
            if key == self.XML_NAME:
                name = sub_dict[self.XML_NAME]
            if key in self.translator:
                new_key = self.translator[key][0]
                class_type = self.translator[key][1]
                data[new_key] = self.nxs_config_object_factory(class_type,
                                                               sub_dict[key])
                if new_key == CHILDREN or new_key == LINK:
                    tmp_list = []
                    for item in sub_dict[key]:
                        tmp_list.append(self.edit_dict_key_value_pair(item,
                                                                      new_key))
                    data[new_key] = tmp_list
        if CHILDREN not in data and parent is not LINK:
            data[CHILDREN] = self.get_stream_information(name)
        if TYPE in data and NAME in data:
            data[ATTRIBUTES] = [{NAME: NX_CLASS,
                                 DATA_TYPE: 'string',
                                 VALUES: data[TYPE]}]
            data[TYPE] = GROUP

        return data

    def get_stream_information(self, name):
        """
        Get the stream information for the file writer to add in the
        file writer config json file.
        """
        stream_info = [{
            WRITER_MODULE: '',
            CONFIG: {
                SOURCE: '',
                TOPIC: '',
                DATA_TYPE: '',
                VALUE_UNITS: '',
            },
        }]

        if name in self.configuration:
            if self._item_is_string(name, WRITER_MODULE):
                stream_info[0][WRITER_MODULE] = self.configuration[name][
                    WRITER_MODULE]
            if self._item_is_string(name, SOURCE):
                stream_info[0][CONFIG][SOURCE] = self.configuration[name][SOURCE]
            if self._item_is_string(name, TOPIC):
                stream_info[0][CONFIG][TOPIC] = self.configuration[name][TOPIC]
            if self._item_is_string(name, DATA_TYPE):
                stream_info[0][CONFIG][DATA_TYPE] = self.configuration[name][DATA_TYPE]
            if self._item_is_string(name, VALUE_UNITS):
                stream_info[0][CONFIG][VALUE_UNITS] = self.configuration[name][VALUE_UNITS]


        return stream_info

    def _item_is_string(self, name, kind):
        return isinstance(self.configuration[name][kind], str)


class NxApplicationXMLToJson:
    NAMESPACE = '{http://definition.nexusformat.org/nxdl/3.1}'

    def __init__(self, xml_path, xls_path, nodes_to_remove=['field']):
        self._xml_path = xml_path
        self._config_xls_path = xls_path
        self.nx_tomo_dict = {}
        self.json_template = {}
        self.nodes_to_remove = nodes_to_remove

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
            nxs_config_creator = FileWriterNexusConfigCreator(dict_cont, self._config_xls_path)
            data = nxs_config_creator.generate_nexus_file_writer_config()
            with open(save_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        return status

    def _remove_nodes_from_tree(self, tree):
        """
        Removes occurrences of specific nodes from tree.
        The undesired nodes are specified in self.nodes_to_remove.

        ::return:: Returns tree where undesired nodes are removed.
        """
        properties = [f'{self.NAMESPACE}{list_item}'
                      for list_item in self.nodes_to_remove]
        tree_root = tree.getroot()
        remove_nodes = [tree_root.findall(f'.//{prop}')
                        for prop in properties]
        for node in remove_nodes[0]:
            node.getparent().remove(node)

        return tree_root


if __name__ == '__main__':
    file_dir = path.dirname(path.abspath(__file__))
    nx_tomo_xml_path = path.join(file_dir, 'NXtomo.xml')
    config_xls_file = path.join(file_dir, 'config.xlsx')
    tomo_xml = NxApplicationXMLToJson(nx_tomo_xml_path, config_xls_file)
    tomo_xml.xml_to_json(path.join(file_dir, 'NXtomo.json'))
