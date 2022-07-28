import sys
from unittest.mock import Mock

module_name = 'arcpy'
arcpy = Mock(name=module_name)
sys.modules[module_name] = arcpy
