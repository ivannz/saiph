import gym
import nle
import time

# from nle.scripts import play
# from nle.scripts.play import main

from nle.nethack import Nethack

from nle_toolbox.wrappers.replay import ReplayToFile

import sys


class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()

        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


def render(obs):
    tty_chars = obs["tty_chars"]
    tty_colors = obs["tty_colors"]
    r, c = obs["tty_cursor"]

    rows, cols = tty_chars.shape
    ansi = '\033[1;1H\033[2J'
    for i in range(rows):
        for j in range(cols):
            cl, ch = tty_colors[i, j], tty_chars[i, j]
            # use separate SGR CSI escapes for attr and color only if necessary
            # ansi += f'\033[{bool(cl&8):d}m\033[3{cl&7:d}m{ch:c}'
            if not cl:
                ansi += chr(ch)
            else:
                # use separate SGR CSI escapes: attr and color
                ansi += f'\033[{bool(cl&8):d}m\033[3{cl&7:d}m{ch:c}'

        ansi += '\n'
    ansi += f'\033[m\033[{1+r};{1+c}H'

    sys.stdout.write(ansi)
    sys.stdout.flush()

    return ansi


if __name__ == '__main__':
    from collections import deque

    getch = _Getch()
    with open('log.txt', 'wt') as log, \
        ReplayToFile(
            gym.make('NetHackChallenge-v0'),  # XXX saiph needs options!
            folder='./replays', save_on='done',
    ) as env:
        log.write('Started \n')

        chr2id = {chr(act.value): i for i, act in enumerate(env.unwrapped._actions)}

        # XXX disable the state machine by initting to `-1`.
        state, buffer = 0, deque([])
        obs, done = env.reset(), False
        log.write(str(len(render(obs))) + '\n')

        while not done:
            # recv character from term

            action = getch()
            action = '\r' if action == '\n' else action

            buffer.append(action)
            if state == 0 and action == '#':
                state = 1
                continue

            elif state == 1:
                if action not in ('\033', '\015'):
                    continue

                state = 0

            # buffer is ready
            log.write(f'state {state} {buffer}\n')
            while buffer:
                action = buffer.popleft()
                obs, reward, done, info = env.step(chr2id[action])

                log.write(f'action: "{action}", "{ord(action)}", "{chr2id.get(action)}"\n')

            log.write((b'\n'.join(map(bytes, obs['tty_chars']))).decode('ascii') + '\n')

            log.write(str(len(render(obs))) + '\n')
            log.flush()
