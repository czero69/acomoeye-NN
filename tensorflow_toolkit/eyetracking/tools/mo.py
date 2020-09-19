#!/usr/bin/env python2.7
#
#   Copyright (C) 2013, Alethea Butler.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

"""
mo.py
A simple utility to organize music files into directories based on tags.
Requires Python 2.7 and the Mutagen tagging library.
"""

import os
import unicodedata
import shutil
import argparse
import mutagen


def main():
    parser = argparse.ArgumentParser(
        prog='MO',
        description='A simple utility to organize music files into \
            directories based on tags.')
    parser.set_defaults(mode='web',
                        max_length=None,
                        overwrite='no',
                        artist='albumartist')

    parser.add_argument('-m', '--move', action='store_true',
                        help='move files instead of copying them')
    parser.add_argument('-a', '--artist', action='store_const',
                        const='artist',
                        help='prefer artist tag over album artist tag')

    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        '-f', '--force', dest='overwrite',
        action='store_const', const='yes',
        help='overwrite existing files without prompting')
    overwrite_group.add_argument(
        '-i', '--interactive', dest='overwrite',
        action='store_const', const='ask',
        help='ask before overwriting files')

    parser.add_argument(
        'sources', metavar='SOURCE', nargs="+",
        help='files to examine')
    parser.add_argument(
        'target', metavar='TARGET',
        help='destination for new directory structure')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-u', '--upper', dest='mode',
        action='store_const', const='clean',
        help='strip non-alphanumeric characters from filenames, leave \
        whitespace and uppercase characters')
    mode_group.add_argument(
        '-s', '--shorten', dest='mode',
        action='store_const', const='short',
        help='shorten filename to its lowercase initials')
    mode_group.add_argument(
        '-l', '--length', dest='max_length', type=int,
        help='shorten names if longer than LENGTH, use web names otherwise')

    parser.add_argument(
        '-t', '--format',
        help=('format string for new directory. Valid tags are {artist}, \
              {album}, {title}, {track}, {disc}, {year} Ex:' +
              repr(default_formats['web'])))

    args = parser.parse_args()
    if args.max_length is not None:
        args.mode = 'mixed'
    if args.format is None:
        args.format = default_formats[args.mode]

    directories = set()
    filepairs = {}

    for source in args.sources:
        try:
            metadata = mutagen.File(source, easy=True)
        except IOError as error:
            metadata = error.strerror
        if not isinstance(metadata, mutagen.FileType):
            if metadata is None:
                metadata = 'Could not extract tags'
            if args.overwrite == 'no':
                parser.error('{0}: {1}'.format(metadata, source))
            elif args.overwrite == 'ask':
                print('{0}: {1}'.format(metadata, source))
                if not ask('Skip it and continue?'):
                    parser.exit()
            continue

        tags = {
            'album': process_name(' '.join(metadata.get('album',
                                                        ['Unknown'])), args),
            'title': process_name(' '.join(metadata.get('title',
                                                        ['Unknown'])), args),
            'track': process_number(metadata.get('tracknumber', [0])[0]),
            'disc': process_number(metadata.get('discnumber', [0])[0]),
            'year': str(process_number(metadata.get('date',
                                                    ['Unknown'])[0], 4))}
        if args.artist == 'albumartist':
            tags['artist'] = process_name(' '.join(
                metadata.get('albumartist',
                             metadata.get('artist', ['Unknown']))), args)
        else:
            tags['artist'] = process_name(' '.join(
                metadata.get('artist',
                             metadata.get('albumartist', ['Unknown']))), args)

        ext = os.path.splitext(source)[1].lower()
        dest = os.path.join(args.target, args.format.format(**tags) + ext)
        filepairs[source] = dest
        directories.add(os.path.dirname(dest))

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for source, dest in filepairs.viewitems():
        if os.path.exists(dest) and args.overwrite != 'yes':
            if args.overwrite == 'no':
                parser.error('File exists: {0}'.format(dest))
            elif args.overwrite == 'ask':
                responce = None
                print('File exists: {0}'.format(dest))
                if not ask('Overwrite it?'):
                    continue
        try:
            if args.move:
                shutil.move(source, dest)
            else:
                shutil.copy(source, dest)
        except IOError as error:
            if args.overwrite == 'no':
                parser.error('{0}: {1}'.format(error.strerror,
                                               error.filename))
            elif args.overwrite == 'ask':
                print('{0}: {1}'.format(error.strerror, error.filename))
                if not ask('Skip it and continue?'):
                    parser.exit()


def process_name(name, args):
    if args.mode == 'none':
        return unicode(name)
    splitnames = name.split()
    subnames = []
    for splitname in splitnames:
        normname = unicodedata.normalize('NFKD', splitname)
        subname = u''.join(char for char in normname if char.isalnum())
        if len(subname) > 0:
            subnames.append(subname)
    if args.mode == 'clean':
        return u' '.join(subnames)
    web = u'-'.join(subname.lower() for subname in subnames)
    if args.mode == 'short' or (args.mode == 'mixed' and
                                len(web) > args.max_length):
        return u''.join(subname.lower()[0] for subname in subnames)
    if args.mode == 'web' or args.mode == 'mixed':
        return web


def process_number(number, length=None):
    if number is None or number == 'Unknown' or isinstance(number, int):
        return number
    digits = []
    found_digit = False
    for char in number:
        if char.isdigit():
            digits.append(char)
            found_digit = True
        elif found_digit:
            break
    if length is not None and len(digits) != length:
        return 'Unknown'
    return int(''.join(digits))


def ask(prompt):
    while True:
        responce = raw_input('{0} [y/n] '.format(prompt))
        if responce.lower() == 'n':
            return False
        elif responce.lower() == 'y':
            return True


default_formats = {
    'none': os.path.join('{artist}', '{album}', '{track:02} {title}'),
    'clean': os.path.join('{artist}', '{album}', '{track:02} {title}'),
    'short': os.path.join('{artist}', '{album}', '{track:02}{title}'),
    'mixed': os.path.join('{artist}', '{album}', '{track:02}-{title}'),
    'web': os.path.join('{artist}', '{album}', '{track:02}-{title}')}

if __name__ == '__main__':
    main()