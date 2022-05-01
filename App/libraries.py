import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import threading
from typing import Union
from PIL import Image
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import av
from streamlit_webrtc import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True