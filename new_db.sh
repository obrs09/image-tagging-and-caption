#!/bin/sh

rm -rf images.db
python main.py
python gui_image_finder.py

