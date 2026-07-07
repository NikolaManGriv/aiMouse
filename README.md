# aiMouse
Virtual mouse powered with computer vision

# Why?
This virtual mouse aims to be used while reading a pdf and you don't want to hold the mouse

# Functions
It uses four fingers (index, middle, ring and pinky). You can think pinky as a shift.

* for scrolling down--> rise index only
* for scrolling up--> rise index and pinky only
* for moving cursor as you wish--> rise index and middle
* for activating drag action--> rise index, middle and pinky, then use moving cursor action
* for deactivating drag (and activating highlighting)--> rise index, middle, and ring

# Instalation
run uv sync and be happy. 
installing uv is quite simple, since you might have pip its just
````bash
pip intall uv
````
Once that is done, run uv sync. This will create a virtual enviroment .venv

If you are using **LINUX/MacOs**
```bash
source .venv/bin/activate
python3 main.py
```

If you are using **Windows**
```bash
.venv\Scripts\activate.bat
```
