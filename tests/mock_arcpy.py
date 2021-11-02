import sys
import types
from unittest.mock import Mock

module_name = 'arcpy'
arcpy = types.ModuleType(module_name)
sys.modules[module_name] = arcpy
arcpy.da = Mock(name=module_name + '.da')
