from PIL import Image
import io
import os
import configparser
from pathlib import Path
import numpy

def save_config_ini(values, section="LAST"):
    inifile = str(Path(__file__).parent.resolve()) + "\\default.ini"
    config = configparser.ConfigParser()
    config.read(inifile)

    try:
        config.add_section(section)
    except configparser.DuplicateSectionError:
        pass

    for k in values:
        config.set(section, k, str(values[k]))

    with open(inifile, "w") as config_file:
        config.write(config_file)


def get_config_ini(section="LAST"):
    inifile = str(Path(__file__).parent.resolve()) + "\\default.ini"
    config = configparser.ConfigParser()
    config.read(inifile)

    section = "DEFAULT" if not config.has_section(section) else section

    rconfig = {}
    try:
        config.get(section, "prompt")
        for key in config[section]:
            rconfig[key] = config.get(section, key)
        return rconfig
    except configparser.NoOptionError:
        pass
    except configparser.NoSectionError:
        pass
    return {}


def get_template_file():
    file = str(Path(__file__).parent.resolve()) + "/templates.csv"
    if os.path.isfile(file):
        return file
    return None


def templates_list():
    file = get_template_file()
    if file is not None:
        with open(file, "r+") as csv:
            return csv.readlines()
    return None


def templates_update(deleted_indexes, lines):
    if deleted_indexes:
        file = get_template_file()
        if file is not None:
            with open(file, "r+") as csv:
                for index in sorted(deleted_indexes, reverse=True):
                    del lines[index]
                csv.seek(0)
                csv.truncate()
                csv.writelines(lines)


def get_img_ratio(curr_width, curr_height):
    def calculate_gcd (w, h): return w if h == 0 else calculate_gcd(h, w % h)
    gcd = calculate_gcd(int(curr_width), int(curr_height))
    return (int(curr_width/gcd), int(curr_height/gcd))


def resize_img(path, size):
    img = Image.open(path)
    img.putalpha(255)
    img.thumbnail(size)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return (numpy.frombuffer(img.tobytes(), dtype=numpy.uint8) / 255.0), img.size
