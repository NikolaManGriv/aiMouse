# aiMouse
Virtual mouse powered with computer vision

# Why?
This virtual mouse aims to be used while reading a pdf and you don't want to hold the mouse

# Functions
It uses four fingers (index, middle, ring and pinky). You can think pinky as a shift.

You have different actions:
* Scrolling up/down
* ctrl c/v
* clicking (left click)
---
---

* For scrolling down--> rise index only
* For scrolling up--> rise index and pinky only
* For moving cursor as you wish--> rise index and middle
* For activating drag action--> rise index, middle and ring fingers, then use moving cursor action (ie, rise only index and middle fingers)
* For deactivating drag (and activating ctrl-c)--> rise index, middle, and pinky fingers.
* In order to paste-->
    * after deactivating drag:
        * rise all fingers except thumb.
    * didn't activate drag:
        * perform deactivating drag action and then paste what you copied as already explained. 
* for clicking--> to perform one click, rise once your middle finger only. When using moving cursor action, to click looks like bending your index finger

# Instalation
run uv sync and be happy. 
installing uv is quite simple, since you might have pip its just
````bash
pip intall uv
uv sync
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
