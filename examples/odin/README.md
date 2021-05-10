# Odin scripts for data storing and streaming

This sub-folder is used to store the scripts that will be used to generate
different part of the NeXuS file format and data streaming for ODIN.

At present state it contains code to automatically generate a NeXuS JSON
template file for NICOS file writing based on the nxTomo from nexus-definitions
github:
https://github.com/nexusformat/definitions.

## Generate NeXus template JSON file for the NICOS file writer

The generate_nXApplication_template.py is a 
runnable python file used to generate the template JSON file.
It uses NXtomo.xml from nexus-definitions to read out the necessary tags and removes any other
tags that are not needed for the JSON based template file.
All the needed readout of the XML file is done in the class 
NxApplicationXMLToJson. It uses the lxml Python package to seemingly remove all
nodes in the XML data not necessary in the creation of the NeXuS JSON template
file. It should of course be able to work with other application classes
in the NeXuS definition repository, but it has not been tried extensively.
The constructor call to NXApplicationXMLToJson class, as seen in the main method,
is supplied two file paths: a path to the config.xlsx (described below), and a path
to the xml file from nexus-definition, in this case NXtomo.xml.
Calling the xml_to_json function in NXApplicationXMLToJson will create a
NeXuS template JSON configuration file for the NICOS file writer. In the example
provided it will be called NXtomo.json and created directly in the odin sub-folder.

The NXapplicationXMLToJson has in its class attribute list a reference to an object
of type FileWriterNexusConfigCreator. The constructor of this class is provided two
parameters. One of them is the Excel file path specifying the data streaming to 
each part of the template JSON file.
Example of such an Excel file is provided in config.xlsx. It defines the relevant
Kafka topics, writer modules and schemas that will be used by the file writer
to correctly create a NeXuS data file, as desired by the user.
The second parameter to the constructor is supplied directly from NXApplicationToJson
and contains the raw XML content from NXMtomo.json, with the exception that any
unnecessary tags are removed that are not needed in the creation of the template file.

A third class, DeviceConfigurationFromXLS, is used to populate the JSON template
with the correct information regarding the kafka topics, schemas and writer module.
This class is an attribute of the FileWriterNexusConfigCreator and is only used
internally by it. The user should never have to be worried about it, more than
providing a correct style config.xlsx with the column names as given in the
example file.
