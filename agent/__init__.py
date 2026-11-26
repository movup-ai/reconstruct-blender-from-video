# -*- coding: utf-8 -*-
"""Blender Agent - local white-model production app.

Stage 1 Movement, fully automatic: video clip -> local models (people, joints, camera, metric scale, floor)
-> Blender build with the sample character -> white-model and comparison videos -> checked v2.1 zip.
Stage 2 Background (later): objects placed from Stage 1's saved depth.

Start:  "Start_Blender_Agent.bat"   (or  .venv\\Scripts\\python -m agent)
"""
__version__ = "1.0.0"
APP_NAME = "Blender Agent"
