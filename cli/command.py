import argparse
from typing import Callable, Dict, List, Union
from cinput import cinput, keyValidator, menuArgsInput, Validator, Menu

class Command:
    def __init__(self, name:str, dependencies:List[Callable[[str], None]], data):
        self.name = name
        self.data = data
        self.parser = argparse.ArgumentParser(name)
        
    
    

