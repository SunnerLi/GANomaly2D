"""
    This script defines some general functions which will be used widely in other scripts

    Author: SunnerLi
"""

def INFO(string):
    print("[ GANomaly2D ] %s" % (string))

def showParameters(args_dict):
    """
        Print the parameters setting line by line
        
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("==================== Parameters ====================")
    for key in sorted(args_dict.keys()):
        INFO("{:>25} : {}".format(key, args_dict[key]))
    INFO("====================================================")
