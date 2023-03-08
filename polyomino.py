#!/usr/bin/env python3
import argparse
import collections
import functools
import itertools
import multiprocessing
import os
import random
import re
import sys

import numpy as np

# Import Numba, if available
try:
    import numba
except ModuleNotFoundError:
    numba = lambda : None
    numba.jit = lambda f: f
    print(
        f'To accelerate the computation, please install Numba:\n'
        f'  {sys.executable} -m pip install numba',
        file=sys.stderr,
    )

# Global constants
cache_size = 2**20
terminal_width = 80
height_termchar = 2
seed_color = 8



#####################
# Generating pieces #
#####################

# Function for finding all possible pieces of a given size
@functools.lru_cache
def generate_pieces(size):
    if size == 1:
        return [((0, 0), )]
    pieces_new = set()
    for piece in generate_pieces(size - 1):
        for block in piece:
            for sign_x, sign_y in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                block_new = (block[0] + sign_x, block[1] + sign_y)
                if block_new in piece:
                    continue
                piece_new = piece + (block_new, )
                pieces_new.add(canonicalize(piece_new))
    return sorted(pieces_new)

# Function returning the fully canonical form of a piece
@functools.lru_cache(cache_size)
def canonicalize(piece):
    return min(generate_symmetries(piece))

# Function returning all symmetric versions
# (up to 8) of a piece, in random order.
@functools.lru_cache(cache_size)
def generate_symmetries(piece):
    def generate_symmetries_offset(piece):
        for parity in range(2):
            yield piece
            for n in range(3):
                piece = rot(piece)
                yield piece
            piece = flip(rot(piece))
    symmetries = set()
    for piece in generate_symmetries_offset(piece):
        dx = -min(block[0] for block in piece)
        dy = -min(block[1] for block in piece)
        symmetries.add(tuple(sorted(
            (block[0] + dx, block[1] + dy)
            for block in piece
        )))
    return list(symmetries)
@functools.lru_cache(cache_size)
def flip(piece):
    return tuple((-block[0], block[1]) for block in piece)
@functools.lru_cache(cache_size)
def rot(piece):
    return tuple((-block[1], block[0]) for block in piece)



#####################
# Tiling rectangles #
#####################

# Class representing a piece, including all symmetries
class Piece:
    def __init__(self, id, blocks):
        self.id = id
        self.canonical = canonicalize(blocks)
        self.size = len(self.canonical)
        self.syms = []
        for blocks in generate_symmetries(self.canonical):
            width, height = self.get_shape(blocks)
            arr = np.zeros((width, height), dtype=np.int32)
            pos = np.empty((self.size, 2), dtype=np.int32)
            for i, block in enumerate(blocks):
                arr[block] = self.id
                pos[i] = block
            self.syms.append(Sym(width, height, arr, pos))
    def get_shape(self, blocks):
        width  = 1 + max(block[0] for block in blocks)
        height = 1 + max(block[1] for block in blocks)
        return width, height
    def prune_syms(self, keep):
        while len(self.syms) > keep:
            self.syms.pop()
Sym = collections.namedtuple('Sym', ('width', 'height', 'arr', 'pos'))

# Function for finding possible rectangles to place the pieces within
def factorize(pieces):
    def get_xy(piece, dim):
        for block in piece:
            yield block[dim]
    size = len(pieces[0])
    width_min = 0
    for piece in pieces:
        width_piece = size
        for dim in range(2):
            width_dim = 1 + max(get_xy(piece, dim)) - min(get_xy(piece, dim))
            width_piece = min(width_piece, width_dim)
        width_min = max(width_min, width_piece)
    area = size*len(pieces)
    factorizations = [
        (i, area//i)
        for i in range(int(area**0.5), width_min - 1, -1)
        if not area%i
    ]
    return factorizations

# Entry function for solving the puzzle
def solve(pieces_in, width, height, max_num_sol, id_index=None):
    # Initialize empty rectangle
    rect = np.zeros((width, height), dtype=np.int32)
    # Transform pieces to 2D NumPy arrays
    pieces = {
        id: Piece(id, blocks)
        for id, blocks in enumerate(pieces_in, 1)
    }
    # Remove symmetry versions of one of the pieces with 8 symmetries,
    # reducing the search space and guaranteeing unique solutions.
    # - For square boards a single symmetry version is needed.
    # - For rectangular boards we need two symmetry versions,
    #   related by a single rotation.
    for piece in pieces.values():
        if len(piece.syms) == 8:
            piece.prune_syms(1 + (width != height))
            break
    # Initialize recursive search
    ids = tuple(pieces.keys())
    if id_index is not None:
        ids = (ids[id_index], )
    solutions = solve_recursively(rect, pieces, 0, 0, [], max_num_sol, ids)
    return solutions

# Recursive puzzle solution
def solve_recursively(rect, pieces, x, y, solutions, max_num_sol, ids):
    if len(solutions) == max_num_sol:
        return solutions
    width, height = rect.shape
    for id in reversed(ids):  # iterating in reversed insertion order speeds up the search
        piece = pieces.pop(id)
        for sym in piece.syms:
            width_sym  = sym.width
            height_sym = sym.height
            arr = sym.arr
            pos = sym.pos
            dx_min = x - width_sym + 1
            if dx_min < 0:
                dx_min = 0
            dx_max_max = width - width_sym
            if x > dx_max_max:
                dx_max = 1 + dx_max_max
            else:
                dx_max = 1 + x
            dy_min = y - height_sym + 1
            if dy_min < 0:
                dy_min = 0
            dy_max_max = height - height_sym
            dy_max = y
            if y > dy_max_max:
                dy_max = 1 + dy_max_max
            else:
                dy_max = 1 + y
            for dx in range(dx_min, dx_max):
                for dy in range(dy_min, dy_max):
                    # Check if piece may be placed
                    if not check_place_piece(rect, x, y, arr, pos, dx, dy):
                        continue
                    # Place piece
                    place_piece(rect, id, pos, dx, dy)
                    if len(pieces) == 0:
                        # All pieces placed
                        solutions.append(rect.copy())
                        rank = multiprocessing.current_process()._identity[0]
                        if nprocs == 1 or rank == 1:
                            # Print out solution only for the first processes
                            print(f'  Solution #{len(solutions)}:')
                            draw_rect(rect, indent=2)
                        # Backtrack
                        place_piece(rect, 0, pos, dx, dy)
                        pieces[id] = piece
                        return solutions
                    # Call recursively
                    x_next, y_next = x, y
                    while rect[x_next, y_next]:
                        # For performance, let the fastest changing
                        # dimension correspond to the shortest side
                        # of the rectangle.
                        x_next = (x_next + 1)%rect.shape[0]
                        y_next = y_next + (x_next == 0)
                    ids = tuple(pieces.keys())
                    solve_recursively(
                        rect, pieces, x_next, y_next,
                        solutions, max_num_sol, ids,
                    )
                    # Backtrack
                    place_piece(rect, 0, pos, dx, dy)
        # Did not succeed in covering (x, y) with this piece
        pieces[id] = piece
    return solutions
nprocs = 1  # so that solve_recursively() may be called without multiprocessing

# Helper functions for solve_recursively(),
# accelerated using Numba.
@numba.jit
def check_place_piece(rect, x, y, arr, pos, dx, dy):
    # Check whether the shifted piece covers (x, y)
    if not arr[x - dx, y - dy]:
        return
    # Check whether the shifted piece covers already occupied terrain
    for i in range(pos.shape[0]):
        if rect[pos[i, 0] + dx, pos[i, 1] + dy]:
            return
    # Piece can be placed
    return True
@numba.jit
def place_piece(rect, id, pos, dx, dy):
    for i in range(pos.shape[0]):
        rect[pos[i, 0] + dx, pos[i, 1] + dy] = id



#####################
# Terminal printing #
#####################

# For printing the pieces
def draw_pieces(pieces):
    rect = []
    sep = np.zeros([0, 0], dtype=np.int32)
    height_max = max(
        Piece(id, piece).syms[0].arr.shape[0]
        for id, piece in enumerate(pieces, 1)
    )
    for id, piece in enumerate(pieces, 1):
        arr = Piece(id, piece).syms[0].arr.copy()
        arr.resize((height_max, arr.shape[1]), refcheck=False)
        rect.append(arr)
        rect.append(sep)
    rect.pop()
    sep.resize((height_max, 1), refcheck=False)
    sep[...] = 0
    rect = np.concatenate(rect, axis=1)
    chunksize = terminal_width//height_termchar
    i = 0
    while True:
        j = i + chunksize
        if j < rect.shape[1]:
            for j in range(j, -1, -1):
                if (rect[:, j] == 0).all():
                   break
        draw_rect(rect[:, i:j])
        i = j + 1
        if i >= rect.shape[1]:
            break
        print()

# For printing a rectangle (puzzle solution)
def draw_rect(rect, indent=0):
    colorfuncs = {id: get_colorfunc(id) for id in sorted(np.unique(rect))}
    rect_strrep = []
    for row in rect:
        rect_strrep.append(' '*indent)
        for id in row:
            if id == 0:
                rect_strrep.append(' '*height_termchar)
            else:
                rect_strrep.append(colorfuncs[id]('█'*height_termchar))
        rect_strrep.append('\n')
    rect_strrep.pop()
    print(''.join(rect_strrep), flush=True)

# Function enabling color output
@functools.lru_cache(cache_size)
def get_colorfunc(id, warning_emitted=[]):
    try:
        import blessings
    except ModuleNotFoundError:
        colorfunc = lambda s: s
        if not warning_emitted:
            print(
                f'For color output, please install Blessings:\n'
                f'  {sys.executable} -m pip install blessings\n'
                f'It may also be preferable to run with TERM=xterm-256color',
                file=sys.stderr,
            )
        warning_emitted.append(True)
    else:
        colors = sorted(
            color
            for color in blessings.COLORS
            if not color.startswith('on_')
        )
        random.seed(seed_color)
        random.shuffle(colors)
        colors = itertools.cycle(colors)
        terminal = blessings.Terminal(force_styling=True)
        for i in range(1 + id):
            color = next(colors)
        colorfunc = getattr(terminal, next(colors))
    return colorfunc



#################
# Run as script #
#################

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'size',
        help='number of blocks in each polyomino',
        default=5,
        nargs='?',
    )
    parser.add_argument(
        '-m', '--max-num-sol',
        help='maximum number of solutions to find',
        default=float('inf'),
    )
    parser.add_argument(
        '-n', '--nprocs',
        help='number of processes',
        default=1,
    )
    parser.add_argument(
        '-r', '--rectangle',
        help='width and height of the rectangle',
    )
    args = parser.parse_args()
    size = int(args.size)
    max_num_sol = float(args.max_num_sol)
    nprocs = int(args.nprocs)
    rectangle = args.rectangle
    if rectangle is not None:
        rectangle = set(map(int, re.findall(r'\d+', rectangle)))
    # Generate all pieces of the given size
    pieces = generate_pieces(size)
    print(f'{len(pieces)} pieces of size {size}:')
    draw_pieces(pieces)
    # Solve tilings
    solutions_all = {}
    for width, height in factorize(pieces):
        if rectangle is not None and {width, height} != rectangle:
            continue
        print(f'Rectangle {width}×{height}:')
        solve_partial = functools.partial(solve, pieces, width, height, max_num_sol)
        with multiprocessing.Pool(nprocs) as pool:
            solutions = pool.map(
                solve_partial,
                [None] if nprocs == 1 else range(len(pieces)),
            )
        solutions = list(itertools.chain(*solutions))
        if not solutions:
            print(f'  No solutions')
        solutions_all[width, height] = solutions
    if solutions_all:
        print('Summary:')
        n = len(str(max(map(len, solutions_all.values()))))
        for (width, height), solutions in solutions_all.items():
            print(f'  Rectangle {width}×{height}: {len(solutions):>{n}} solutions')
