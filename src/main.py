from parse_arguments import parse_arguments
from experiments import Experiment
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
def main():
    args = parse_arguments()
    Experiment(args=args).run()

if __name__ == '__main__':
   main()