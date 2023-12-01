# Allow users to not install Tensorflow if they only want to use non-tf architecures
try:
    import tensorflow as tf
except ImportError:
    Warning("Tensorflow not installed. Some functionality may not work.")
