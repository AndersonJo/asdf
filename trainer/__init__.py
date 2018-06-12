import os
import sys

print(os.path.abspath(os.path.join(__file__, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))
