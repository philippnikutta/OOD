import argparse
from time import strftime

import torch




class Main(object):
    def __init__(self, args):
        self.args = args
        self.timestamp = strftime('%Y-%m-%d-%H.%M.%S')
        for k, v in args.__dict__.items():
            setattr(self, k, v)

    def __str__(self):
        return json.dumps(self.args.__dict__)

    def __repr__(self):
        return 'python main.py ' + ' '.join(sys.argv[1:])
            
    def __call__(self):
        if (self.command[0] == '_'
            or not hasattr(self, self.command)
            or not callable(getattr(self, self.command))):
          raise RuntimeError(f"bad command: {self.command}")
        getattr(self, self.command)()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='command(s) to run')
    args = parser.parse_args()
    Main(args)()