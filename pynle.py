import gym
import nle
import time

import numpy as np
import pprint as pp
from textwrap import wrap

# from nle.scripts import play
# from nle.scripts.play import main

from nle.nethack import Nethack

from nle_toolbox.wrappers.replay import ReplayToFile

import sys
from collections import deque, namedtuple
from nle_toolbox.utils.obs import BLStats


Misc = namedtuple('Misc', 'in_yn_function,in_getlin,xwaitingforspace')


try:
    class _Getch:
        """Gets a single character from standard input in MS Windows."""
        from msvcrt import getch as __call__

except ImportError:
    class _Getch:
        def __call__(self):
            import sys
            import tty
            import termios

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return sys.stdin.read(1)

            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def render(tty_colors, tty_chars, tty_cursor, **ignore):
    r, c = tty_cursor

    rows, cols = tty_chars.shape
    ansi = '\033[1;1H\033[2J'
    for i in range(rows):
        for j in range(cols):
            cl, ch = tty_colors[i, j], tty_chars[i, j]
            # use escapes only if necessary
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


def rebuild_tty_chars(obs):
    """Restore broken `tty_chars` in case of mutliline message with `--More--`.
    """

    misc = Misc(*obs['misc'].astype(bool))
    message = obs['message'].view('S256')[0].decode('ascii')
    if '\n' in message and misc.xwaitingforspace:
        # replace lf-s with whitespace and reconstruct the tty_chars
        message = message.replace('\n', ' ')
        screen = np.copy(obs['tty_chars'])

        # FYI tty_chars[1:-2, :-1] == chars on no-message screens
        screen[1:-2, :-1] = obs['chars']

        # properly wrap the text at 80
        pieces = wrap(message + '--More--', 80)
        assert len(pieces) == 2, "Should never be >= 2 by design."

        vw_screen = screen.view('S80')
        for j, line in enumerate(pieces):
            vw_screen[j] = bytes(f'{line:<80s}', encoding='ascii')

        # overwrite the tty with the screen and patch the cursor
        obs['tty_cursor'][:] = 1, len(pieces[1])
        obs['tty_chars'] = screen

        # fix the message: replace '\n' (\x0a) with space ` ` (\x20)
        obs['message'] = np.where(message == 0x0A, 0x20, message)

    return obs


def main(log, *, seed=None):
    getch = _Getch()
    with ReplayToFile(
        gym.make('NetHackChallenge-v0'),  # XXX saiph needs options!
        folder='./replays',
        save_on='done',
    ) as env:
        env.seed(seed=None if seed is None else tuple(seed))

        log.write("python pynle.py --seed " + ' '.join(map(str, env._seed)) + '\n')

        # XXX disable the state machine by initting to `-1`.
        state, buffer = 0, deque([])

        chr2id = {chr(act.value): i for i, act in enumerate(env.unwrapped._actions)}
        obs, done = env.reset(), False
        while not done:
            obs = rebuild_tty_chars(obs)

            # some flags and computed fields
            message = obs['message'].view('S256')[0].decode('ascii')
            msc = Misc(*obs['misc'].astype(bool))

            # NB NLE does no report game turn counter in the bottom line stats.
            bls = BLStats(*obs['blstats'])

            # log stuff
            log.write(f"{state=} `{message=}`\n")
            log.write(pp.pformat(dict(bls=bls._asdict(), msc=msc._asdict())) + '\n')
            log.write(f'sent {len(render(**obs)):d} to stdout:\n')
            log.write((b'\n'.join(map(bytes, obs['tty_chars']))).decode('ascii') + '\n')

            log.flush()

            # recv character from term
            action = getch()
            action = '\r' if action == '\n' else action

            buffer.append(action)

            # case 1: an extended command r"#\w+[\015\033]"
            if state == 0:
                if action == '#':
                    state = 1
                    continue

            elif state == 1:
                if action not in ('\033', '\015'):
                    continue

                state = 0

            assert state <= 0  # can use -1 to disable buffering

            # buffer is ready
            log.write(">>>>> Taking " + repr(tuple(buffer))[1:-1] + "\n")
            while buffer:
                action = buffer.popleft()
                obs, reward, done, info = env.step(chr2id[action])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Saiph-NLE interface.',
        add_help=True)

    parser.add_argument(
        '--seed', type=int, nargs=2, required=False, dest='seed',
        help='The seed pair to use. Ignore for unseeded run.',
    )

    parser.set_defaults(seed=None)
    args, _ = parser.parse_known_args()

    with open('log.txt', 'wt') as log:
        try:
            main(log, **vars(args))

        except Exception as e:
            log.write(f'{type(e).__name__}({str(e)})')

    # import gym
    # import nle

    # with gym.make('NetHackChallenge-v0') as env:
    #     obs = env.reset()
