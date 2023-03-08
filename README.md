Discovering and rectangular tiling of [polyominoes](https://en.wikipedia.org/wiki/Polyomino)
---


### Quick guide

To find all polyominoes of size 5:
```bash
python polyomino.py 5 --rect None
```

To further find all solutions to the 6×10 tiling:
```bash
python polyomino.py 5 --rect 6x10
```

To find all solutions to all tilings:
```bash
python polyomino.py 5
```

To see all available command-line arguments:
```bash
python polyomino.py -h
```


### Dependencies
The code requires Python >= 3.8 and NumPy.

For colour output, install [Blessings](https://github.com/erikrose/blessings):
```bash
python -m pip install blessings
```
If only a few colours appear, try running with
```bash
TERM=xterm-256color python polyomino.py
```

To significantly accelerate the code, install [Numba](https://github.com/numba/numba):
```bash
python -m pip install numba
```

