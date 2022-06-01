import json
import pandas as pd
import xmltodict
from enum import Enum
from os import path
from lxml import etree
from xml.parsers.expat import ExpatError

ARRAY_SIZE = 'array_size'
ATTRIBUTES = 'attributes'
CHILDREN = 'children'
CUSTOM_FIELD = 'custom_field'
CONFIG = 'config'
DATA_NAME = 'data_name'
DATASET = 'dataset'
DATA_TYPE = 'dtype'
GROUP = 'group'
KIND = 'kind'
LINK = 'link'
NAME = 'name'
NX_CLASS = 'NX_class'
NX_DATA = 'NXdata'
SOURCE = 'source'
STATIC_DATA = 'static_data'
STATIC_VALUE = 'static_value'
STREAM = 'stream'
TARGET = 'target'
TOPIC = 'topic'
TYPE = 'type'
VALUES = 'values'
VALUE_UNITS = 'value_units'
WRITER_MODULE = 'module'


class DeviceConfigurationFromXLS:

    list_excel_cols = [NAME, TOPIC, SOURCE, WRITER_MODULE, DATA_NAME, KIND,
                       DATA_TYPE, VALUE_UNITS, ARRAY_SIZE, CUSTOM_FIELD,
                       STATIC_VALUE]

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
            if name in config_dict:
                config_dict[name].append(tmp_dict)
            else:
                config_dict.update({name: [tmp_dict]})
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

    nexus_instance_name = {'NXentry': 'entry',
                           'NXdetector': 'detector',
                           'NXinstrument': 'instrument',
                           'NXsample': 'sample',
                           'NXmonitor': 'control',
                           'NXdata': 'data',
                           'NXsource': 'source'}

    def __init__(self, nxs_definition_xml, xls_path):
        self._nxs_definition_xml = nxs_definition_xml
        self.translator = self.get_translation()
        self.configuration = DeviceConfigurationFromXLS(xls_path).\
            get_configuration_as_dict()
        self._data = {}
        self._data_fields = None

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
        self._data = self.edit_dict_key_value_pair(self._nxs_definition_xml)
        return {CHILDREN: [self._data]}

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
                if new_key == CHILDREN:
                    tmp_list = []
                    for item in sub_dict[key]:
                        tmp_list.append(self.edit_dict_key_value_pair(item,
                                                                      new_key))
                    data[new_key] = tmp_list
                elif new_key == LINK:
                    tmp_list = []
                    for item in sub_dict[key]:
                        tmp_list.append(self.get_link(item))
                    data[CHILDREN] = tmp_list
                    del data[LINK]
        if TYPE in data and data[TYPE] in self.nexus_instance_name:
            data[NAME] = self.nexus_instance_name[data[TYPE]]
            data[ATTRIBUTES] = [{NAME: NX_CLASS,
                                 DATA_TYPE: 'string',
                                 VALUES: data[TYPE]}]
            data[TYPE] = GROUP
        if CHILDREN not in data and parent is not LINK:
            data[CHILDREN] = self.get_stream_information(data[NAME])
        elif parent is not LINK:
            for item in self.get_stream_information(data[NAME]):
                data[CHILDREN].append(item)
        return data

    def get_link(self, data):
        """
        Returns a nexus link dictionary.
        """
        return {
            WRITER_MODULE: LINK,
            CONFIG: {
                NAME: data[self.XML_NAME],
                SOURCE: self._translate_link(data[self.XML_TARGET])
            }
        }

    def _translate_link(self, link):
        """
        Translates target link to something the file-writer will understand.
        """
        nexus_instances = link.split('/')
        translated_nexus_list = []
        for nexus_instance in nexus_instances:
            nxs_item = nexus_instance.split(':')[-1]
            if nxs_item in self.nexus_instance_name:
                translated_nexus_list.append(
                    self.nexus_instance_name[nxs_item])
            else:
                translated_nexus_list.append(nexus_instance)
        return '/'.join(translated_nexus_list)

    def get_stream_information(self, name):
        """
        Get the stream information for the file writer to add in the
        file writer config json file.
        """
        stream_info_aggr = []
        if name in self.configuration:
            for item in self.configuration[name]:
                stream_info = {}
                if item[KIND] == GROUP:
                    stream_info = {
                        WRITER_MODULE: '',
                        CONFIG: {
                            SOURCE: '',
                            TOPIC: '',
                            DATA_TYPE: '',
                        },
                    }
                    if self._item_is_string(WRITER_MODULE, item):
                        stream_info[WRITER_MODULE] = \
                            item[WRITER_MODULE]
                    if self._item_is_string(SOURCE, item):
                        stream_info[CONFIG][SOURCE] = \
                            item[SOURCE]
                    if self._item_is_string(TOPIC, item):
                        stream_info[CONFIG][TOPIC] = \
                            item[TOPIC]
                    if self._item_is_string(DATA_TYPE, item):
                        stream_info[CONFIG][DATA_TYPE] = \
                            item[DATA_TYPE]
                    if self._item_is_string(VALUE_UNITS, item):
                        stream_info[CONFIG][VALUE_UNITS] = \
                            item[VALUE_UNITS]
                    if self._item_is_string(ARRAY_SIZE, item):
                        str_values = item[ARRAY_SIZE].split(',')
                        int_values = [int(val) for val in str_values]
                        stream_info[CONFIG][ARRAY_SIZE] = int_values
                    if self._item_is_string(DATA_NAME, item):
                        stream_info = {
                            TYPE: GROUP,
                            NAME: item[DATA_NAME],
                            CHILDREN: [stream_info],
                            ATTRIBUTES: [
                                {
                                    NAME: NX_CLASS,
                                    DATA_TYPE: "string",
                                    VALUES: "NXlog"
                                }
                            ]
                        }
                elif item[KIND] == STATIC_DATA:
                    config = {}
                    if self._item_is_string(DATA_TYPE, item):
                        config[DATA_TYPE] = item[DATA_TYPE]
                    if self._item_is_string(STATIC_VALUE, item):
                        config[VALUES] = item[STATIC_VALUE]
                    if self._item_is_string(DATA_NAME, item):
                        config[NAME] = item[DATA_NAME]
                        stream_info = {
                            WRITER_MODULE: DATASET,
                            CONFIG: config
                        }
                if stream_info:
                    stream_info_aggr.append(stream_info)
        return stream_info_aggr

    @staticmethod
    def _item_is_string(kind, item):
        return isinstance(item[kind], str)


class NxApplicationXMLToJson:
    NAMESPACE = '{http://definition.nexusformat.org/nxdl/3.1}'

    def __init__(self, xml_path, xls_path, nodes_to_remove=None):
        if nodes_to_remove is None:
            nodes_to_remove = ['field']
        self._xml_path = xml_path
        self._config_xls_path = xls_path
        self.nx_tomo_dict = {}
        self.json_template = {}
        self.nodes_to_remove = nodes_to_remove

    def load_template_from_xml(self):
        """
        Load nexus template file from xml file and dump the content
        into a dictionary.
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

    def xml_to_json(self):
        """
        Save nexus template from xml format to json format.
        """
        status = True
        if not self.nx_tomo_dict:
            status = self.load_template_from_xml()
        if status:
            dict_cont = self.nx_tomo_dict['definition']['group']
            nxs_config_creator = \
                FileWriterNexusConfigCreator(dict_cont, self._config_xls_path)
            self.json_template = \
                nxs_config_creator.generate_nexus_file_writer_config()
        return status

    def save_json_file(self, save_path):
        with open(save_path, 'w') as json_file:
            json.dump(self.json_template, json_file, indent=4)

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
    tomo_xml.xml_to_json()
    tomo_xml.save_json_file(path.join(file_dir, 'NXtomo.json'))
